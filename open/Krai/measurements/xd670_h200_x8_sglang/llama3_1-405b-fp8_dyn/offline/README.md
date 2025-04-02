
# MLPerf Inference v5.0 - Open - Krai

To run experiments individually, use the following commands.

## xd670_h200_x8_sglang - llama3.1-405b - offline

### Accuracy  

```
axs byquery loadgen_output,task=llama3_1,framework=openai,loadgen_mode=AccuracyOnly,loadgen_scenario=Offline,loadgen_dataset_size=8313,loadgen_buffer_size=8313,num_openai_workers=1,num_loadgen_workers=1,backend=default,tp=8,pp=1,dp=1,num_gpus=8,quantization=fp8,max_num_seqs=128,inference_server=sglang,server_docker_image=lmsysorg/sglang,server_docker_image_tag=latest
```

### Performance 

```
axs byquery loadgen_output,task=llama3_1,framework=openai,loadgen_mode=PerformanceOnly,loadgen_scenario=Offline,loadgen_dataset_size=8313,loadgen_buffer_size=8313,num_openai_workers=1,num_loadgen_workers=1,backend=default,tp=8,pp=1,dp=1,num_gpus=8,quantization=fp8,max_num_seqs=128,inference_server=sglang,server_docker_image=lmsysorg/sglang,server_docker_image_tag=latest,loadgen_target_qps=None
```

