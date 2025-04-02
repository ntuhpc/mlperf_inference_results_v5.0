# MLPerf Inference

This folder contains all of the code necessary to run:
 - MLPerf Inference Multi-node "Offline"
 - MLPerf Inference Multi-node "Server"   

We use **FP8 quantized llama2-70b** model inference to generate all the results.  Please ensure that all software dependecies: ROCm 6.3.0, LLMBoost-0.5.2, Python 3.12 are installed. The benchmarking is run on **MI300X GPUs**. (four 8-MI300X nodes for multinode benchmarking).

The following steps outline the process of setting up a Docker environment and the details to reproduce our MLPerf V5.0 inference results.   

---

## 1. Docker Preparation

### Model and Dataset

Build the docker image for the benchmark by running the below command

```bash
bash setup/build_model_and_dataset_env.sh
```

Start the docker container for the benchmark by running the below command

```bash
bash setup/start_model_and_dataset_env.sh
```

Inside the docker, download the model with

```bash
# Generate an access token on huggingface and set it here
HUGGINGFACE_ACCESS_TOKEN="<your HF token goes here>" python download_model.py
```

Inside the docker, download the dataset with

```bash
bash download_llama2_70b.sh
```

Inside the docker, quantize the model with

```bash
bash quantize_llama2_70b.sh
```

## 2. Multi-Node Benchmarking (Default 4 Nodes):

We provide docker for the multi-node benchmarking. Please get our docker from docker hub by:

```bash
docker pull llmboost/mb-llmboost:mlperf-5.0
```   
(This docker is specifically for MLPerf Inference Submission with limited funtionality of LLMBoost. If you are interested in the production version of our LLM platform, please visit our LLMBoost product page: https://www.mangoboost.io/products/software/mango-llmboost-tm. We support on-premise and on cloud deployment (AWS, Azure, and GCP) running on both Nvidia and AMD GPUs.)   

Then, please download LLMBoost wheel from our MLPerf submission repo: `closed/MangoBoost/code/llama2-70b-99.9/llmboost-0.5.2%2Bpy3.12-py3-none-any.whl`. After getting the wheel, please start the docker by:   

```bash
docker run -it --rm \
  --device=/dev/dri:/dev/dri \
  --device=/dev/kfd:/dev/kfd \
  -v <path to quantized models>:/models \
  -v <path to llmboost wheel>:/workspace/llmboost-0.5.2+py3.12-py3-none-any.whl \
  llmboost/mb-llmboost:mlperf-5.0
```   

Next, a `README.md` can be found within the docker `/workspace`. Please follow that README to finish the rest of the multi-node benchmarking.

----

## 3. Packing the Submission & Conducting Validation Checking

After the commands above, all the results are ready to be packaged. Please run the commands below to package the results and doing the submission checking for validation.

```bash
cd /workspace
bash submission_package.sh
```

+ If all the results are valid and all the submission directory structure is satisfied, it is ready to be zipped and submitted!


## Appendix

### Multi Node Benchmarking - Code Explanation

The command to run the `server` is `python server.py` and its arguments are:
```
       USAGE: server.py [flags]
flags:

server.py:
  --beam_width: beam width if used
    (default: '0')
    (an integer)
  --dp: number of workers
    (default: '8')
    (an integer)
  --host: Host address
  --kv_cache_dtype: <auto|fp8>: KV cache dtype
    (default: 'auto')
  --load_format: <bitsandbytes|bitsandbytesint8|auto>: Quantized model to use
    (default: 'auto')
  --max_num_seqs: max number of sequences to process in parallel
    (default: '0')
    (an integer)
  --max_tokens: max tokens to generate
    (default: '1024')
    (an integer)
  --model_name: <gptj|llama2-70b|llama2-7b|mixtral-8x7b>: Model name
    (default: 'llama2-70b')
  --port: Port number
    (default: '8000')
    (an integer)
  --quantization: <bitsandbytes|fp8|None>: Quantization scheme to use
    (default: 'None')
  --quantization_param_path: Path to quantization parameters
  --quantization_weight_path: Path to quantization weights
  --[no]streaming: enable streaming
    (default: 'false')
  --tp: tensor parallelism
    (default: '1')
    (an integer)
  --[no]vllm: Turn on/off vLLM engine
    (default: 'true')
  --gpu_batch_size: The max number of queries to be sent to each worker as a batch.
  	(default: '48')
  --batcher_threshold: The max duration (in sec) that we wait to send the pending queries.
  	(default: '0.4')
```

The command to run the `client` is `python client.py` and its arguments are:
```
       USAGE: client.py [flags]
flags:

client.py:
  --[no]accuracy_test: Perform accuracy test
    (default: 'false')
  --batched_queries: Number of batched requests
    (default: '10')
    (an integer)
  --mlperf_conf: Path to mlperf.conf
    (default: 'mlperf.conf')
  --model_name: <gptj|llama2-70b|llama2-7b|mixtral-8x7b>: Model name
    (default: 'llama2-70b')
  --parallel_requests: Number of parallel requests
    (default: '10')
    (an integer)
  --sut_server_addr: List of server address
    (default: 'http://localhost:8000')
  --test_mode: <Offline|Server>: type of test to perform
    (default: 'Server')
  --user_conf: Path to user.conf
    (default: 'user.conf')
```
