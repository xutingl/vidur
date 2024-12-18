from abc import ABC, abstractmethod
from vidur.config import EarlyExitType

from vidur.config import (
    BaseExecutionTimePredictorConfig,
    BaseReplicaSchedulerConfig,
    MetricsConfig,
    ReplicaConfig,
)
from vidur.entities import Batch, ExecutionTime


class BaseExecutionTimePredictor(ABC):
    def __init__(
        self,
        predictor_config: BaseExecutionTimePredictorConfig,
        replica_config: ReplicaConfig,
        replica_scheduler_config: BaseReplicaSchedulerConfig,
        metrics_config: MetricsConfig,
    ) -> None:
        self._config = predictor_config
        self._replica_config = replica_config
        self._model_config = replica_config.model_config

        # get configs
        self._replica_scheduler_provider = str(replica_scheduler_config.get_type())
        self._block_size = replica_scheduler_config.block_size
        self._cache_dir = metrics_config.cache_dir
        self._num_layers_per_pipeline_stage = (
            self._model_config.num_layers // self._replica_config.num_pipeline_stages
        )

        print("----------")
        print(f"Execution Time Predictor initialized with Early-Exit type={self._config.early_exit_type}")
        print("----------")

    def get_execution_time(self, batch: Batch, pipeline_stage: int, fraction_skipped: float = 0.0) -> ExecutionTime:
        if pipeline_stage == self._replica_config.num_pipeline_stages - 1:
            pipeline_parallel_communication_time = 0
        else:
            pipeline_parallel_communication_time = (
                self._get_pipeline_parallel_communication_time(batch)
            )

        if self._replica_config.tensor_parallel_size == 1:
            tensor_parallel_communication_time = 0
        else:
            tensor_parallel_communication_time = (
                self._get_tensor_parallel_communication_time(batch)
            )
        
        if self._config.early_exit_type != 0: 
            if fraction_skipped > 0 or batch.exited:
                batch.exited = True
                if self._config.early_exit_type == EarlyExitType.MOD or pipeline_stage == self._replica_config.num_pipeline_stages - 1: # Reset exited status at the end of iteration; For MoD (type 2), always reset so that the models continues to execute the next layer
                    batch.exited = False
                return ExecutionTime(
                    self._num_layers_per_pipeline_stage,
                    self._get_attention_rope_execution_time(batch) * (1 - fraction_skipped),
                    self._get_attention_kv_cache_save_execution_time(batch) * (1 - fraction_skipped),
                    self._get_attention_decode_execution_time(batch) * (1 - fraction_skipped),
                    self._get_attention_prefill_execution_time(batch) * (1 - fraction_skipped),
                    self._get_attention_layer_pre_proj_execution_time(batch) * (1 - fraction_skipped),
                    self._get_attention_layer_post_proj_execution_time(batch) * (1 - fraction_skipped),
                    self._get_mlp_layer_up_proj_execution_time(batch) * (1 - fraction_skipped),
                    self._get_mlp_layer_down_proj_execution_time(batch) * (1 - fraction_skipped),
                    self._get_mlp_layer_act_execution_time(batch) * (1 - fraction_skipped),
                    self._get_attn_norm_layer_act_execution_time(batch) * (1 - fraction_skipped),
                    self._get_mlp_norm_layer_act_execution_time(batch) * (1 - fraction_skipped),
                    self._get_add_layer_act_execution_time(batch) * (1 - fraction_skipped),
                    tensor_parallel_communication_time,
                    pipeline_parallel_communication_time,
                    self._get_schedule_time(batch),
                    self._get_sampler_e2e_time(batch),
                    self._get_prepare_inputs_e2e_time(batch),
                    self._get_process_model_outputs_time(batch),
                    self._get_ray_comm_time(batch),
                )


        return ExecutionTime(
            self._num_layers_per_pipeline_stage,
            self._get_attention_rope_execution_time(batch),
            self._get_attention_kv_cache_save_execution_time(batch),
            self._get_attention_decode_execution_time(batch),
            self._get_attention_prefill_execution_time(batch),
            self._get_attention_layer_pre_proj_execution_time(batch),
            self._get_attention_layer_post_proj_execution_time(batch),
            self._get_mlp_layer_up_proj_execution_time(batch),
            self._get_mlp_layer_down_proj_execution_time(batch),
            self._get_mlp_layer_act_execution_time(batch),
            self._get_attn_norm_layer_act_execution_time(batch),
            self._get_mlp_norm_layer_act_execution_time(batch),
            self._get_add_layer_act_execution_time(batch),
            tensor_parallel_communication_time,
            pipeline_parallel_communication_time,
            self._get_schedule_time(batch),
            self._get_sampler_e2e_time(batch),
            self._get_prepare_inputs_e2e_time(batch),
            self._get_process_model_outputs_time(batch),
            self._get_ray_comm_time(batch),
        )

    @abstractmethod
    def _get_attention_layer_pre_proj_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_attention_layer_post_proj_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_attention_rope_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_attention_kv_cache_save_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_attention_decode_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_attention_prefill_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_mlp_layer_up_proj_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_mlp_layer_down_proj_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_mlp_layer_act_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_tensor_parallel_communication_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_pipeline_parallel_communication_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_schedule_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_sampler_e2e_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_prepare_inputs_e2e_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_process_model_outputs_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_ray_comm_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_mlp_norm_layer_act_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_attn_norm_layer_act_execution_time(self, batch: Batch) -> float:
        pass

    @abstractmethod
    def _get_add_layer_act_execution_time(self, batch: Batch) -> float:
        pass
