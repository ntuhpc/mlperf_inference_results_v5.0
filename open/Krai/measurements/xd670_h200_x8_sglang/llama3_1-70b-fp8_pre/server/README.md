
# MLPerf Inference v5.0 - Open - Krai

To run experiments individually, use the following commands.

## xd670_h200_x8_sglang - llama3_1-70b-fp8_pre - server

### Accuracy  

```
axs byquery loadgen_output,task=llama2,framework=openai,loadgen_mode=AccuracyOnly,loadgen_scenario=Server,loadgen_dataset_size=24576,loadgen_buffer_size=24576,num_openai_workers=96,num_loadgen_workers=1,tp=1,pp=1,dp=8,num_gpus=8,max_num_seqs=700,max_seq_len_to_capture=1024,max_num_batched_tokens=8192,gpu_memory_utilization=0.95,loadgen_target_qps=85,openai_client_max_retries=0,openai_max_connections=801,openai_max_keepalive_connections=801,openai_retry_delay_ms=2000,model_path=/scratch/krai/models/Meta-Llama-3.1-70B-Instruct-FP8/
```

### Performance 

```
axs byquery loadgen_output,task=llama2,framework=openai,loadgen_mode=PerformanceOnly,loadgen_scenario=Server,loadgen_dataset_size=24576,loadgen_buffer_size=24576,num_openai_workers=96,num_loadgen_workers=1,tp=1,pp=1,dp=8,num_gpus=8,max_num_seqs=700,max_seq_len_to_capture=1024,max_num_batched_tokens=8192,gpu_memory_utilization=0.95,loadgen_target_qps=90,openai_client_max_retries=0,openai_max_connections=801,openai_max_keepalive_connections=801,openai_retry_delay_ms=2000,model_path=/scratch/krai/models/Meta-Llama-3.1-70B-Instruct-FP8/
```

### Compliance TEST06

```
axs byquery loadgen_output,task=llama2,framework=openai,loadgen_mode=PerformanceOnly,loadgen_scenario=Server,loadgen_dataset_size=24576,loadgen_buffer_size=24576,num_openai_workers=96,num_loadgen_workers=1,tp=1,pp=1,dp=8,num_gpus=8,max_num_seqs=700,max_seq_len_to_capture=1024,max_num_batched_tokens=8192,gpu_memory_utilization=0.95,loadgen_target_qps=85,openai_client_max_retries=0,openai_max_connections=801,openai_max_keepalive_connections=801,openai_retry_delay_ms=2000,model_path=/scratch/krai/models/Meta-Llama-3.1-70B-Instruct-FP8/,loadgen_compliance_test=TEST06,inference_server=sglang,server_docker_image=lmsysorg/sglang,server_docker_image_tag=latest
```

