# MLPerf Inference using vLLM

## Define workspace directory

```
export WORKSPACE_DIR=/local/mnt/workspace
mkdir -p ${WORKSPACE_DIR}/${USER}
```

## Install `axs`

Install `axs` under `${WORKSPACE_DIR}/$USER`:

```
git clone https://github.com/krai/axs ${WORKSPACE_DIR}/$USER/axs
```

Define environment variables in your `~/.bashrc`:
```
echo "

# AXS.
export WORKSPACE_DIR=/local/mnt/workspace
export PATH=${WORKSPACE_DIR}/${USER}/axs:${PATH}
export AXS_WORK_COLLECTION=${WORKSPACE_DIR}/${USER}/work_collection

" >> ~/.bashrc
```

### Test
```
source ~/.bashrc
axs version
```

## Import public `axs` repositories

Import the required public repos into your work collection:

```
axs byquery git_repo,collection,repo_name=axs2mlperf
axs byquery git_repo,collection,repo_name=axs2vllm

```

## Download model, tokenizer, and dataset

```
axs byquery downloaded,hf_model,model_family=llama3_1,variant=405b,revision=be673f326cab4cd22ccfef76109faf68e41aa5f1,hf_token=$HF_TOKEN
```
```
axs byquery downloaded,hf_tokeniser,model_family=llama3_1,variant=405b,total_samples=8313,hf_token=$HF_TOKEN
```
```
axs byquery downloaded,dataset_name=llrg,model_family=llama3_1,variant=405b,desired_python_version===3.9
```

## Run Benchmarks

### Offline Accuracy
```
axs byquery loadgen_output,task=llama3_1,framework=openai,loadgen_mode=AccuracyOnly,loadgen_scenario=Offline,backend=cuda,loadgen_dataset_size=8313,loadgen_buffer_size=8313,num_openai_workers=1,num_loadgen_workers=1,tp=1,pp=1,dp=1,num_gpus=8,quantization=None,max_num_seqs=256,max_seq_len_to_capture=32768,max_num_batched_tokens=131072,max_model_len=131072,gpu_memory_utilization=0.95,openai_max_connections=256,min_tokens=2,max_tokens=2000,temperature=1,
```

### Offline Performance
```
axs byquery loadgen_output,task=llama3_1,framework=openai,loadgen_mode=PerformanceOnly,loadgen_scenario=Offline,backend=cuda,loadgen_dataset_size=8313,loadgen_buffer_size=8313,num_openai_workers=1,num_loadgen_workers=1,tp=1,pp=1,dp=1,num_gpus=8,quantization=None,max_num_seqs=256,max_seq_len_to_capture=32768,max_num_batched_tokens=131072,max_model_len=131072,gpu_memory_utilization=0.95,openai_max_connections=256,min_tokens=2,max_tokens=2000,temperature=1,loadgen_target_qps=<desired qps>
```