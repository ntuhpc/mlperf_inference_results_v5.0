# Instructions on how to run SDXL on TPU


## Setup
### Create 1 TPU Trillium-4 VM.
Follow these [instructions](https://cloud.google.com/tpu/docs/v6e-intro#provision-queued-resource) to create TPUv6e-4 VMs.

### Setup Code & Dependencies
ssh into the VM

git clone https://github.com/mlcommons/inference.git

sudo apt install python3.10-venv
python -m venv venv-maxtext
source venv-maxtext/bin/activate

cd `inference/text_to_image/` and pip install -r requirements.txt

From the Readme here, install and setup loadgen

Copy the `sdlx_tpu` found in the submission package, into the cloned repo `inference/text_to_image/` directory.

Then make sure you're in the directory  `inference/text_to_image/sdxl_tpu`.
`pip install -r requirements.txt`

`pip install jax[tpu]==0.5.0 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html`

`pip install flax==0.10.2`

-------

## Offline Scenario 

Make sure per_device_batch_size in `base_xl.yml` is `16` for Offline mode performance and accuracy

### Performance

```python
python3 main.py --config='{ROOT_DIRECTORY}/inference/text_to_image/sdxl_tpu/configs/base_xl.yml' --latents=<'absolute path to numpy latents file'> --threads=80 --scenario=Offline --threshold-time=8 --threshold-queue-length=64 --max-batchsize=64 --output={OUTPUT_PATH} 

```


### Accuracy
```python
python3 main.py --config='{ROOT_DIRECTORY}/inference/text_to_image/sdxl_tpu/configs/base_xl.yml' --latents=<'absolute path to numpy latents file'> --threads=80 --scenario=Offline --threshold-time=8 --threshold-queue-length=64 --max-batchsize=64 --output={OUTPUT_PATH} --accuracy
```



-----

## Server Scenario 

Make sure per_device_batch_size in `base_xl.yml` is `8` for Server mode performance and accuracy


## Performance
```python
python3 main.py --config='{ROOT_DIRECTORY}/inference/text_to_image/sdxl_tpu/configs/base_xl.yml' --latents=<'absolute path to numpy latents file'> --threads=80 --scenario=Server --threshold-time=10 --threshold-queue-length=32 --max-batchsize=32 --output={OUTPUT_PATH}

```


## Accuracy 
```python
python3 main.py --config='{ROOT_DIRECTORY}/inference/text_to_image/sdxl_tpu/configs/base_xl.yml' --latents=<'absolute path to numpy latents file'> --threads=80 --scenario=Server --threshold-time=10 --threshold-queue-length=32 --max-batchsize=32 --output={OUTPUT_PATH} --accuracy

```


----




