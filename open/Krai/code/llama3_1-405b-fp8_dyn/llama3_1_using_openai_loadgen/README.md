# MLPerf Inference - Llama3_1 task

## Prerequisites
* Docker must be installed.
* The user running benchmarks must be incuded in the `docker` group.

## Define a local workspace directory
```
export WORKSPACE_DIR=/workspace
mkdir -p ${WORKSPACE_DIR}
mkdir -p ${WORKSPACE_DIR}/${USER}
```

## Install KRAI [AXS](https://github.com/krai/axs)

### Clone

Clone the AXS repository under `${WORKSPACE_DIR}/$USER`:
```
git clone https://github.com/krai/axs ${WORKSPACE_DIR}/$USER/axs
```

### Init

Define environment variables in your `~/.bashrc`:
```
echo "

# AXS.
export WORKSPACE_DIR=/workspace
export PATH=${WORKSPACE_DIR}/${USER}/axs:${PATH}
export AXS_WORK_COLLECTION=${WORKSPACE_DIR}/${USER}/work_collection

" >> ~/.bashrc
```

### Test
```
source ~/.bashrc
axs version
```

## Import public AXS repositories

Import the required public repos into your work collection:

```
axs byquery git_repo,collection,repo_name=axs2mlperf
axs byquery git_repo,collection,repo_name=axs2vllm
```

## Download artifacts

Use a [HuggingFace access token](https://huggingface.co/docs/hub/en/security-tokens) (`export HF_TOKEN=...`) to download the model and its tokenizer.

### Model
The benchmark supports the following models:
* Llama3.1-405b
To download the model run
```
axs byquery downloaded,hf_model,model_family=llama3_1,variant=405b,hf_token=${HF_TOKEN}
```

You can also provide your own model by adding `,model_path=...` to the commands below.

### Tokenizer
```
axs byquery downloaded,hf_tokeniser,model_family=llama3_1,variant=405b,total_samples=8313,hf_token=${HF_TOKEN}
```

### Dataset (LongBench, LongDataCollections, Ruler, GovReport)
```
axs byquery downloaded,dataset_name=llrg,model_family=llama3_1,variant=405b,desired_python_version===3.9
```

## Benchmark

See below example commands for the Server/Offline scenarios and under the Accuracy/Performance/Compliance modes.

See commands used for actual submission runs under the corresponding subdirectories of `measurements/`.

The `quantization` parameter can be set to `None` if you want to use a non-quantized or prequantized model.
If it is set to `fp8` - a dynamic fp8 quantisation would be applied by an inference server at model loading.

The `backend` parameter supports options `cuda` and `rocm`.

The benchmark supports teh following inference servers: vLLM (by default), SGLang and NIM.
To activate not default ones, specify `inference_server=sglang` or `inference_server=nim`.

### Offline
#### Accuracy
```
axs byquery loadgen_output,task=llama3_1,framework=openai,loadgen_mode=AccuracyOnly,loadgen_scenario=Offline,backend=cuda,quantization=None
```

#### Performance
```
axs byquery loadgen_output,task=llama3_1,framework=openai,loadgen_mode=PerformanceOnly,loadgen_scenario=Offline,backend=cuda,quantization=None,loadgen_target_qps=<desired qps>
```

### Server
#### Accuracy
```
axs byquery loadgen_output,task=llama3_1,framework=openai,loadgen_mode=AccuracyOnly,loadgen_scenario=Server,backend=cuda,quantization=None,loadgen_target_qps=<desired qps>
```

#### Performance
```
axs byquery loadgen_output,task=llama3_1,framework=openai,loadgen_mode=PerformanceOnly,loadgen_scenario=Server,backend=cuda,quantization=None,loadgen_target_qps=<desired qps>
```
