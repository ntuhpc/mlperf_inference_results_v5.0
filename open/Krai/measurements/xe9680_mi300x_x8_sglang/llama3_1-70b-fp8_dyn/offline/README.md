
# MLPerf Inference v5.0 - Open - Krai

To run experiments individually, use the following commands.

## xe9680_mi300x_x8 - llama3.1-70b-99.9 - offline

### Accuracy  

```
axs byquery loadgen_output,task=llama2,framework=openai,loadgen_mode=AccuracyOnly,loadgen_scenario=Offline,loadgen_dataset_size=24576,loadgen_buffer_size=24576,num_openai_workers=32,num_loadgen_workers=1,backend=rocm,tp=1,pp=1,dp=8,num_gpus=8,quantization=fp8,openai_max_connections=192,model_path=/mnt/llm_data/krai/models/Llama-3.1-70B-Instruct,inference_server=sglang,server_docker_image=lmsysorg/sglang,server_docker_image_tag=v0.4.3.post2-rocm630-srt
```

### Performance 

```
axs byquery loadgen_output,task=llama2,framework=openai,loadgen_mode=PerformanceOnly,loadgen_scenario=Offline,loadgen_dataset_size=24576,loadgen_buffer_size=24576,num_openai_workers=64,num_loadgen_workers=1,backend=rocm,tp=1,pp=1,dp=8,num_gpus=8,quantization=fp8,openai_max_connections=384,model_path=/mnt/llm_data/krai/models/Llama-3.1-70B-Instruct,inference_server=sglang,server_docker_image=lmsysorg/sglang,server_docker_image_tag=v0.4.3.post2-rocm630-srt,loadgen_target_qps=60
```

