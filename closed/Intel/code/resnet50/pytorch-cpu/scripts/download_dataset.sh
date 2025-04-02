#!/bin/bash

CHECKSUMS=(
            /data,0eb6aa7d356fabd0429ea46659ef4ec1
          )

echo "Downloading dataset..."
export DATA_DIR=/data
cd ${DATA_DIR}
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
tar -xf ILSVRC2012_img_val.tar
rm ILSVRC2012_img_val.tar
wget https://raw.githubusercontent.com/mlcommons/inference_results_v4.0/main/closed/Intel/code/resnet50/pytorch-cpu/val_data/val_map.txt

echo "Begining validation..."
cd /workspace
for ITEM in "${CHECKSUMS[@]}"; do
    FILENAME=$(echo ${ITEM} | cut -d',' -f1)
    CHECKSUM=$(echo ${ITEM} | cut -d',' -f2)
    bash run_validation.sh ${FILENAME} ${CHECKSUM}
done
