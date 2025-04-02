
# MLPerf Inference v5.0 - Open - Krai

To run experiments individually, use the following commands.

## xe9680_mi300x_x8_sglang - deepseek-v3 - offline

### Accuracy  

```
axs byquery loadgen_output,task=llama3_1,framework=openai,loadgen_mode=AccuracyOnly,loadgen_scenario=Offline,num_openai_workers=1,num_loadgen_workers=1,backend=rocm,tp=8,pp=1,dp=1,quantization=fp8,model_path=/mnt/llm_data/krai/models/DeepSeek-V3,loadgen_target_qps=1,inference_server=sglang,attempt=2
```

### Performance 

```
axs byquery loadgen_output,task=llama3_1,framework=openai,loadgen_mode=PerformanceOnly,loadgen_scenario=Offline,num_openai_workers=1,num_loadgen_workers=1,backend=rocm,tp=8,pp=1,dp=1,quantization=fp8,model_path=/mnt/llm_data/krai/models/DeepSeek-V3,loadgen_target_qps=1,inference_server=sglang
```

### Compliance TEST06

```
axs byquery loadgen_output,task=llama3_1,framework=openai,loadgen_mode=PerformanceOnly,loadgen_scenario=Offline,num_openai_workers=1,num_loadgen_workers=1,backend=rocm,tp=8,pp=1,dp=1,quantization=fp8,model_path=/mnt/llm_data/krai/models/DeepSeek-V3,loadgen_target_qps=1,inference_server=sglang,loadgen_compliance_test=TEST06
```

