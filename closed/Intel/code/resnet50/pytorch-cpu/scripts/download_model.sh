#!/bin/bash

CHECKSUMS=(
            /model/resnet50-fp32-model.pth,9e9c86b324d80e65229fab49b8d9a8e8
          )

echo "Downloading model..."
export MODEL_DIR=/model
wget --no-check-certificate https://zenodo.org/record/4588417/files/resnet50-19c8e357.pth -O "${MODEL_DIR}/resnet50-fp32-model.pth"

echo "Begining validation..."
cd /workspace
for ITEM in "${CHECKSUMS[@]}"; do
    FILENAME=$(echo ${ITEM} | cut -d',' -f1)
    CHECKSUM=$(echo ${ITEM} | cut -d',' -f2)
    bash run_validation.sh ${FILENAME} ${CHECKSUM}
done
