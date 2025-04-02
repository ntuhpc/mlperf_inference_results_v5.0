*Check [CM MLPerf docs](https://docs.mlcommons.org/inference) for more details.*

## Host platform

* OS version: Linux-6.8.0-1021-aws-aarch64-with-glibc2.39
* CPU version: aarch64
* Python version: 3.12.3 (main, Feb  4 2025, 14:48:35) [GCC 13.3.0]
* MLC version: unknown

## CM Run Command

See [CM installation guide](https://docs.mlcommons.org/inference/install/).

```bash
pip install -U mlcflow

mlc rm cache -f

mlc pull repo gateoverflow@mlperf-automations --checkout=8bc932351c07e1239299d1ce761f3b43e2181ede


```
*Note that if you want to use the [latest automation recipes](https://docs.mlcommons.org/inference) for MLPerf,
 you should simply reload gateoverflow@mlperf-automations without checkout and clean MLC cache as follows:*

```bash
mlc rm repo gateoverflow@mlperf-automations
mlc pull repo gateoverflow@mlperf-automations
mlc rm cache -f

```

## Results

Platform: aws_c7g.4xlarge-ctuning_cpp_tflite-cpu-armnn_tflite_vv23.11-using_neon_with_live_preprocessing

Model Precision: fp32

### Accuracy Results 
`acc`: `70.624`, Required accuracy for closed division `>= 75.6954`

### Performance Results 
`90th percentile latency (ns)`: `3019452.0`
