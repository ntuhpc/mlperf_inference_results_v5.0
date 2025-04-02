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
sudo rocm-smi --setperfdeterminism 1600
sudo amd-smi set --soc-pstate 0 -g all

## Performance
python /lab-mlperf-inference/code/llama2-70b-99.9/main.py \
   --config-path /lab-mlperf-inference/code/llama2-70b-99.9/harness_llm/models/llama2-70b/ \
   --config-name server_mi325x \
   test_mode=performance \
   harness_config.data_parallel_size=8 \
   harness_config.user_conf_path=/lab-mlperf-inference/code/llama2-70b-99.9/user_mi325x.conf \
   harness_config.output_log_dir=/lab-mlperf-inference/results/llama2-70b/Server/performance/run_1

## Accuracy
python /lab-mlperf-inference/code/llama2-70b-99.9/main.py \
   --config-path /lab-mlperf-inference/code/llama2-70b-99.9/harness_llm/models/llama2-70b/ \
   --config-name server_mi325x \
   test_mode=accuracy \
   harness_config.data_parallel_size=8 \
   harness_config.user_conf_path=/lab-mlperf-inference/code/llama2-70b-99.9/user_mi325x.conf \
   harness_config.output_log_dir=/lab-mlperf-inference/results/llama2-70b/Server/accuracy

### Evaluate accuracy
bash /lab-mlperf-inference/code/llama2-70b-99.9/scripts/check_llama2_accuracy_scores.sh \
   /lab-mlperf-inference/results/llama2-70b/Server/accuracy/mlperf_log_accuracy.json
```
