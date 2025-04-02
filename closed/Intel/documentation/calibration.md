## Intel MLPerf Inference Calibration and Quantization Details

### ResNet-50 Quantization
Model Source: https://zenodo.org/record/4588417/files/resnet50-19c8e357.pth

Model Quantization: FP32 -> INT8

Steps: /closed/Intel/code/resnet50/pytorch-cpu/scripts/run_calibration.sh

### RetinaNet Quantization
Model Source: https://zenodo.org/record/6617981/files/resnext50_32x4d_fpn.pth

Model Quantization: FP32 -> INT8

Steps: /closed/Intel/code/retinanet/pytorch-cpu/scripts/run_calibration.sh

### 3D-UNet Quantization
Model Source: https://zenodo.org/record/5597155/files/3dunet_kits19_pytorch_checkpoint.pth

Model Quantization: FP32 -> MIXED (FP32 + INT8)

Steps: /closed/Intel/code/3d-unet-99.9/pytorch-cpu/scripts/run_calibration.sh

### DLRMv2 Quantization
Model Source: https://zenodo.org/record/5597155

Model Quantization: FP32 -> INT8

Steps: /closed/Intel/code/dlrm-v2-99.9/pytorch-cpu/scripts/run_calibration.sh

### GPT-J Quantization
Model Source: https://cloud.mlcommons.org/index.php/s/QAZ2oM94MkFtbQx/download

Model Quantization: FP32 -> INT8

Steps: /closed/Intel/code/gptj-99.9/pytorch-cpu/scripts/run_calibration.sh

### R-GAT Quantization
Model Source: https://github.com/IllinoisGraphBenchmark/IGB-Datasets/

Model Quantization: FP32 -> INT8

Implementation: /closed/Intel/code/rgat/pytorch-cpu/backend.py
