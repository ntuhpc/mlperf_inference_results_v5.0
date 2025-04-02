*Check [CM MLPerf docs](https://docs.mlcommons.org/inference) for more details.*

## Host platform

* OS version: Linux-6.8.0-53-generic-x86_64-with-glibc2.39
* CPU version: x86_64
* Python version: 3.9.11 (main, Feb 22 2025, 10:32:50) 
[GCC 13.3.0]
* MLC version: unknown

## CM Run Command

See [CM installation guide](https://docs.mlcommons.org/inference/install/).

```bash
pip install -U mlcflow

mlc rm cache -f

mlc pull repo mlcommons@mlperf-automations --checkout=af51c745ce2a498c95423591e173cc693e60ecba


```
*Note that if you want to use the [latest automation recipes](https://docs.mlcommons.org/inference) for MLPerf,
 you should simply reload mlcommons@mlperf-automations without checkout and clean MLC cache as follows:*

```bash
mlc rm repo mlcommons@mlperf-automations
mlc pull repo mlcommons@mlperf-automations
mlc rm cache -f

```

## Results

Platform: ideapadG3-reference-cpu-deepsparse_v1.8.0-default_config

Model Precision: int8

### Accuracy Results 
`F1`: `90.81269`, Required accuracy for closed division `>= 89.96526`

### Performance Results 
`90th percentile latency (ns)`: `56181306.0`
