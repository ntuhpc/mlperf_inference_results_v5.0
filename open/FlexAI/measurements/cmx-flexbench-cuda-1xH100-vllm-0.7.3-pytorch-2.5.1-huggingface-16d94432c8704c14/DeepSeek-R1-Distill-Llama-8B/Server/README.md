Please go to [this directory](../../../../code/DeepSeek-R1-Distill-Llama-8B/src) and run the following commands
to reproduce the results using the [standard MLPerf OpenOrca dataset](https://github.com/mlcommons/inference/tree/master/language/llama2-70b#get-dataset).

# Install FlexBench

1. Install uv (optional):
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install dependencies:
```sh
uv sync
source .venv/bin/activate
```

# Download dataset

Please follow [this guide](https://github.com/mlcommons/inference/tree/master/language/llama2-70b#get-dataset)
to obtain `open_orca_gpt4_tokenized_llama.sampled_24576.pkl` file (MLPerf OpenOrca dataset).

# Start vLLM server (terminal 1)

```sh
CUDA_VISIBLE_DEVICES=0 vllm serve "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
    --disable-log-requests \
    --max-model-len=2048
```

# Run MLPerf loadgen in Server mode (terminal 2)

You can run FlexBench with a different number of samples to analyze its impact on accuracy and performance.

## Performance (4000 samples)

```sh
python main.py \
    --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --dataset-path open_orca_gpt4_tokenized_llama.sampled_24576.pkl \
    --scenario Server \
    --total-sample-count 4000 \
    --target-qps 4.5
```

## Performance (full dataset - 24576 samples)

```sh
python main.py \
    --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --dataset-path open_orca_gpt4_tokenized_llama.sampled_24576.pkl \
    --scenario Server \
    --total-sample-count 24576 \
    --target-qps 4.5
```

## Accuracy (2000 samples)

```sh
python main.py \
    --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --dataset-path open_orca_gpt4_tokenized_llama.sampled_24576.pkl \
    --scenario Server \
    --total-sample-count 2000 \
    --target-qps 4.5 \
    --accuracy
```

## Accuracy (full dataset - 24576 samples)

```sh
python main.py \
    --model-path deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --dataset-path open_orca_gpt4_tokenized_llama.sampled_24576.pkl \
    --scenario Server \
    --total-sample-count 24576 \
    --target-qps 4.5 \
    --accuracy
```
