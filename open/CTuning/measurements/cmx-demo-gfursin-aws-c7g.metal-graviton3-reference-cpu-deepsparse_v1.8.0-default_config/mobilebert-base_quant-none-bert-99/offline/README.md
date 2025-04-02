The sparsified bert submissions are using the [MLCommons reference implementation](https://github.com/mlcommons/cm4mlops/tree/main/script/app-mlperf-inference-mlcommons-python) extended by NeuralMagic to add the [deepsparse backend](https://github.com/neuralmagic/inference/blob/deepsparse/language/bert/deepsparse_SUT.py).

Please follow [this script](https://github.com/mlcommons/cm4mlops/blob/main/script/run-all-mlperf-models/run-pruned-bert.sh) for generating an end to end submission for the sparsified bert models. 

## Host platform

* OS version: Linux-6.8.0-1021-aws-aarch64-with-glibc2.35
* CPU version: aarch64
* Python version: 3.10.12 (main, Feb  4 2025, 14:57:36) [GCC 11.4.0]
* MLC version: unknown

## CMX Run Command

See [CMX installation guide](https://access.cknowledge.org/playground/?action=install).

```bash
pip install -U cmind
pip install -U cmx4mlperf
pip install -U mlcflow

cmx test core

cmlc rm cache -f

cmlc pull repo mlcommons@mlperf-automations --checkout=c8cb2c378fcc84d44fe20a81ef24956bc93dffc0


```
*Note that if you want to use the [latest automation recipes](https://docs.mlcommons.org/inference) for MLPerf,
 you should simply reload mlcommons@mlperf-automations without checkout and clean MLC cache as follows:*

```bash
cmlc rm repo mlcommons@mlperf-automations
cmlc pull repo mlcommons@mlperf-automations
cmlc rm cache -f

