
# MLPerf Inference v5.0 - Open - Krai

To run experiments individually, use the following commands.

## xe9680_mi300x_x8_vllm_073 - llama3.1-405b - offline

### Accuracy  

```
axs byquery loadgen_output,task=llama3_1,framework=openai,loadgen_mode=AccuracyOnly,loadgen_scenario=Offline,loadgen_dataset_size=8313,loadgen_buffer_size=8313,num_openai_workers=1,num_loadgen_workers=1,backend=rocm,tp=8,pp=1,dp=1,num_gpus=8,quantization=fp8,model_path=/mnt/llm_data/krai/lg4/work_collection/downloaded_Llama-3.1-405b-Instruct_model,loadgen_target_qps=1
```

### Performance 

```
axs byquery loadgen_output,task=llama3_1,framework=openai,loadgen_mode=PerformanceOnly,loadgen_scenario=Offline,loadgen_dataset_size=8313,loadgen_buffer_size=8313,num_openai_workers=1,num_loadgen_workers=1,backend=rocm,tp=8,pp=1,dp=1,num_gpus=8,quantization=fp8,max_num_seqs=256,max_seq_len_to_capture=32768,max_num_batched_tokens=131072,gpu_memory_utilization=0.97,model_path=/mnt/llm_data/krai/lg4/work_collection/downloaded_Llama-3.1-405b-Instruct_model,min_tokens=2,max_tokens=2000,max_model_len=131072,temperature=1,openai_max_connections=256,loadgen_target_qps=1
```

