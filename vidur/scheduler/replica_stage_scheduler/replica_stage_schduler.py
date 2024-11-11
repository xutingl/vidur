from typing import Tuple

from vidur.entities import Batch, BatchStage, ExecutionTime
from vidur.execution_time_predictor import BaseExecutionTimePredictor
from vidur.config import EarlyExitType, BaseReplicaSchedulerConfig
import random
import heapq


class ReplicaStageScheduler:
    def __init__(
        self,
        replica_id: int,
        stage_id: int,
        is_last_stage: bool,
        execution_time_predictor: BaseExecutionTimePredictor,
        replica_scheduler_config: BaseReplicaSchedulerConfig,
        num_stages: int=-1
    ) -> None:
        self._replica_id = replica_id
        self._stage_id = stage_id
        self._is_last_stage = is_last_stage
        self._execution_time_predictor = execution_time_predictor

        self.enable_priority_queue = replica_scheduler_config.enable_priority_queue

        self._batch_queue = []
        self.queue_length_samples = [] # Store the length of current queue each time a batch is added
        self._is_busy = False

        self._num_stages: int = num_stages
        self.early_exit_type: EarlyExitType = execution_time_predictor._config.early_exit_type

    @property
    def is_last_stage(self) -> bool:
        return self._is_last_stage

    def is_empty(self) -> bool:
        return len(self._batch_queue) == 0

    def add_batch(self, batch: Batch) -> None:
        self.queue_length_samples.append(len(self._batch_queue))
        if self.enable_priority_queue:
            heapq.heappush(self._batch_queue, batch)
        else:
            self._batch_queue.append(batch)
        

    def on_stage_end(self) -> None:
        self._is_busy = False
    
    """
    Returns the fraction of layers that are skipped in the current stage
    """
    def get_fraction_skipped(self, num_layers_per_stage=4, skip_chance=0.5) -> float:
        if self.early_exit_type == EarlyExitType.MOD:
            num_skipped_layers = 0
            for i in range(num_layers_per_stage):
                if random.random() < skip_chance:
                    num_skipped_layers += 1
            return num_skipped_layers / num_layers_per_stage
        if self.early_exit_type == EarlyExitType.EE:
            assert self._num_stages >= 1
            total_layers = num_layers_per_stage * self._num_stages
            fraction_skipped = 0.0
            for i in range(num_layers_per_stage):
                skip_chance = 0.8 / total_layers * (self._stage_id + 1) * (i + 1)
                if random.random() < skip_chance:
                    fraction_skipped = (num_layers_per_stage - i) / num_layers_per_stage
                    break
            return fraction_skipped
        
        if self.early_exit_type == EarlyExitType.NO_EE:
            return 0.0

        raise Exception(f"Expecting early exit type = 1 or 2 but got {self.early_exit_type}")
    
    def on_schedule(self) -> Tuple[Batch, BatchStage, ExecutionTime]:
        if self._is_busy or not self._batch_queue:
            return None, None, None

        self._is_busy = True
        if self.enable_priority_queue:
            batch = heapq.heappop(self._batch_queue)
        else:
            batch = self._batch_queue.pop(0)

        fraction_skipped = 0
        if self.early_exit_type != 0:
            fraction_skipped = self.get_fraction_skipped(skip_chance=0.5)
        execution_time = self._execution_time_predictor.get_execution_time(
            batch,
            self._stage_id,
            fraction_skipped=fraction_skipped
        )
        total_execution_time = execution_time.total_time
        model_execution_time = execution_time.model_time
        batch_stage = BatchStage(
            batch.id,
            self._replica_id,
            self._stage_id,
            total_execution_time,
            model_execution_time,
            batch.requests,
            batch.num_tokens,
        )

        return batch, batch_stage, execution_time
