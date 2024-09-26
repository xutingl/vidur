# Default
python -m vidur.main  \
--replica_config_device a100 \
--replica_config_model_name meta-llama/Llama-2-7b-hf  \
--cluster_config_num_replicas 1 \
--replica_config_tensor_parallel_size 1 \
--replica_config_num_pipeline_stages 8 \
--request_generator_config_type synthetic \
--length_generator_config_type trace \
--interval_generator_config_type static \
--trace_request_length_generator_config_max_tokens 4096 \
--trace_request_length_generator_config_trace_file ./data/processed_traces/arxiv_summarization_stats_llama2_tokenizer_filtered_v2.csv \
--synthetic_request_generator_config_num_requests 2048  \
--replica_scheduler_config_type vllm  \
--vllm_scheduler_config_batch_size_cap 256  \
--vllm_scheduler_config_max_tokens_in_batch 4096


## batch size = 2048
(8196 seems to be too large)

python -m vidur.main  \
--replica_config_device a100 \
--replica_config_model_name meta-llama/Llama-2-7b-hf  \
--cluster_config_num_replicas 1 \
--replica_config_tensor_parallel_size 1 \
--replica_config_num_pipeline_stages 8 \
--request_generator_config_type synthetic \
--length_generator_config_type trace \
--interval_generator_config_type static \
--trace_request_length_generator_config_max_tokens 2048 \
--trace_request_length_generator_config_trace_file ./data/processed_traces/arxiv_summarization_stats_llama2_tokenizer_filtered_v2.csv \
--synthetic_request_generator_config_num_requests 128  \
--replica_scheduler_config_type vllm  \
--vllm_scheduler_config_batch_size_cap 128  \
--vllm_scheduler_config_max_tokens_in_batch 2048




python -m vidur.main  \
--replica_config_device a100 \
--replica_config_model_name meta-llama/Llama-2-7b-hf  \
--cluster_config_num_replicas 1 \
--replica_config_tensor_parallel_size 4 \
--replica_config_num_pipeline_stages 2 \
--request_generator_config_type synthetic \
--length_generator_config_type trace \
--interval_generator_config_type static \
--trace_request_length_generator_config_max_tokens 4096 \
--trace_request_length_generator_config_trace_file ./data/processed_traces/arxiv_summarization_stats_llama2_tokenizer_filtered_v2.csv \
--synthetic_request_generator_config_num_requests 2048  \
--replica_scheduler_config_type vllm  \
--vllm_scheduler_config_batch_size_cap 256  \
--vllm_scheduler_config_max_tokens_in_batch 4096