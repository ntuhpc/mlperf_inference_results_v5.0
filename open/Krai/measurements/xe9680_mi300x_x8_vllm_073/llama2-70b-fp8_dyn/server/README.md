
# MLPerf Inference v5.0 - Open - Krai

To run experiments individually, use the following commands.

## xe9680_mi300x_x8_vllm_073 - llama2-70b-fp8_dyn - server

### Accuracy  

```
axs byquery loadgen_output,task=llama2,framework=openai,loadgen_mode=AccuracyOnly,loadgen_scenario=Server,loadgen_dataset_size=24576,loadgen_buffer_size=24576,num_openai_workers=32,num_loadgen_workers=1,backend=rocm,tp=1,pp=1,dp=8,num_gpus=8,quantization=fp8,max_num_seqs=512,max_seq_len_to_capture=2048,max_num_batched_tokens=65536,gpu_memory_utilization=0.99,loadgen_target_qps=45,openai_client_max_retries=0,openai_max_connections=192,openai_max_keepalive_connections=192,uvloop+
```

### Performance 

```
axs byquery loadgen_output,task=llama2,framework=openai,loadgen_mode=PerformanceOnly,loadgen_scenario=Server,loadgen_dataset_size=24576,loadgen_buffer_size=24576,num_openai_workers=32,num_loadgen_workers=1,backend=rocm,tp=1,pp=1,dp=8,num_gpus=8,quantization=fp8,max_num_seqs=512,max_seq_len_to_capture=2048,max_num_batched_tokens=65536,gpu_memory_utilization=0.99,loadgen_target_qps=45,openai_client_max_retries=0,openai_max_connections=192,openai_max_keepalive_connections=192,uvloop+
```

### Compliance TEST06

```
axs byquery loadgen_output,task=llama2,framework=openai,loadgen_mode=PerformanceOnly,loadgen_scenario=Server,loadgen_dataset_size=24576,loadgen_buffer_size=24576,num_openai_workers=32,num_loadgen_workers=1,backend=rocm,tp=1,pp=1,dp=8,num_gpus=8,quantization=fp8,max_num_seqs=512,max_seq_len_to_capture=2048,max_num_batched_tokens=65536,gpu_memory_utilization=0.99,loadgen_target_qps=45,openai_client_max_retries=0,openai_max_connections=192,openai_max_keepalive_connections=192,uvloop+,loadgen_compliance_test=TEST06
```

