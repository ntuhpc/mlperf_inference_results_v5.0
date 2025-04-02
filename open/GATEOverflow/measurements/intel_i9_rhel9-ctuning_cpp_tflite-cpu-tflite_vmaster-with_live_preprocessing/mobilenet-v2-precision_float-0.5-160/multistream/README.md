*Check [CM MLPerf docs](https://docs.mlcommons.org/inference) for more details.*

## Host platform

* OS version: Linux-6.8.0-53-generic-x86_64-with-glibc2.34
* CPU version: x86_64
* Python version: 3.9.21 (main, Dec  5 2024, 00:00:00) 
[GCC 11.5.0 20240719 (Red Hat 11.5.0-2)]
* MLC version: unknown

## CM Run Command

See [CM installation guide](https://docs.mlcommons.org/inference/install/).

```bash
pip install -U mlcflow

mlc rm cache -f

mlc pull repo gateoverflow@mlperf-automations --checkout=ca73f3b42b6a8af79ddd71f2175ee016d2afdd30


```
*Note that if you want to use the [latest automation recipes](https://docs.mlcommons.org/inference) for MLPerf,
 you should simply reload gateoverflow@mlperf-automations without checkout and clean MLC cache as follows:*

```bash
mlc rm repo gateoverflow@mlperf-automations
mlc pull repo gateoverflow@mlperf-automations
mlc rm cache -f

```

## Results

Platform: intel_i9_rhel9-ctuning_cpp_tflite-cpu-tflite_vmaster-with_live_preprocessing

Model Precision: fp32

### Accuracy Results 
`acc`: `61.668`, Required accuracy for closed division `>= 75.6954`

### Performance Results 
`90th percentile latency (ns)`: `1120557.0`
