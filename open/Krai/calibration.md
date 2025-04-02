# MLPerf Inference v5.0 - Krai

## Calibration (Quantization) Details

Submissions from Krai on NVIDIA H200 and AMD MI300X GPUs either use dynamic FP8 quantization with vLLM and SGLang (`llama2-70b-fp8_dyn`, `llama3_1-70b-fp8_dyn`, `llama3_1-405b-fp8_dyn`), or FP8 models pre-quantized by NVIDIA (`llama3_1-70b-fp8_nim`), Neural Magic (`llama3_1-70b-fp8_pre`), Meta (`llama3_1-405b-fp8_pre`) and DeepSeek (`deepseek-v3`).

Model name | Model URL<br>(`starting_weights_filename`) | Weights data type<br>(`weight_data_types`) | Weights transformations<br>(`weight_transformations`)
-|-|-|-
`llama2-70b` | https://huggingface.co/meta-llama/Llama-2-70b-chat-hf | fp16 | none
`llama2-70b-fp8_dyn` | https://huggingface.co/meta-llama/Llama-2-70b-chat-hf | fp16 | fp8 quantization (dynamic)
`llama3_1-70b` | https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct | fp16 | none
`llama3_1-70b-fp8_dyn` | https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct | fp16 | fp8 quantization (dynamic)
`llama3_1-70b-fp8_pre` | https://huggingface.co/neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8 | fp8 | none
`llama3_1-70b-fp8_nim` | https://docs.nvidia.com/nim/large-language-models/1.5.0/supported-models.html#llama-3-1-70b-instruct | fp8 | none
`llama3_1-405b` | https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct | fp16 | none
`llama3_1-405b-fp8_dyn` | https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct | fp16 | fp8 quantization (dynamic)
`llama3_1-405b-fp8_pre` | https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct-FP8 | fp8 | none
`deepseek-v3` | https://huggingface.co/deepseek-ai/DeepSeek-V3 | fp8 | none
