# MLPerf Inference 5.0

## Setup

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

Exit the docker image, because a different image is needed for inference

## Inference

### Runtime tunables

To boost the machine's performance further, execute the following script before any performance test (should be set once after a reboot):

```bash
bash setup/runtime_tunables.sh
```

### Docker

NOTE: The script below will use a pre-built docker base image. If it is not available online, or you want to build it from scratch, use rocm dockerfiles in setup/vllm/ to create it.

Build the docker image for the benchmark by running the below command

```bash
bash setup/build_submission_env.sh
```

Start the docker container for the benchmark by running the below command

```bash
bash setup/start_submission_env.sh
```

### Running the benchmark

Run the following commands inside the docker container

``` bash
### MI325x Power Setup
sudo rocm-smi --setperfdeterminism 1700
sudo amd-smi set --soc-pstate 0 -g all
```

### Running the benchmark and submission packaging

```bash
COMPANY="GigaComputing" SYSTEM_NAME="G893-ZX1-AAX2" GPU_NAME="mi325x" bash /lab-mlperf-inference/submission/llama2_70b.sh
```
