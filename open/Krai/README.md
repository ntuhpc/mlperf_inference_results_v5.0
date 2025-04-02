# MLPerf Inference v5.0 - Krai

Krai present LLM submissions using vLLM, SGLang and NIM on HPE Cray XD670
servers with NVIDIA H200 GPUs and on Dell PowerEdge XE9680 servers with AMD
MI300X GPUs.

The submissions use the [KRAI](https://krai.ai) [KISS](http://github.com/krai/axs2kiss)
(KRAI Inference Serving Solution) for fast, efficient and scalable inference, and the
[KRAI X](http://github.com/krai/axs) technology for workflow automation.

Detailed setup instructions per workload are provided in README files under the
[code](code) directory.  Individual benchmarking commands per system,
workload, scenario and mode are provided in README files under the respective
[measurements](measurements) directories.
Calibration (quantization) details are provided in [calibration.md](calibration.md).

The source code has been released under the permissive MIT license across
several public repositories (under the `mlperf_5.0` branches created by the
v5.0 submission deadline):

- https://github.com/krai/axs (KRAI X Workflow Automation Technology)
- https://github.com/krai/axs2kiss (KRAI Inference Serving Solution)
- https://github.com/krai/axs2mlperf

## Contact

For any queries please contact info@krai.ai.
