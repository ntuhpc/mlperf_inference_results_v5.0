#!/bin/bash

CHECKSUMS=(
           /model/3dunet_kits19_pytorch_checkpoint.pth,09c696e3ec13d83c628498bcd831eb5b
          )

echo "Downloading model..."
export MODEL_DIR=/model
ZENODO_PYTORCH="https://zenodo.org/record/5597155/files/3dunet_kits19_pytorch_checkpoint.pth?download=1"
PYTORCH_MODEL="${MODEL_DIR}/3dunet_kits19_pytorch_checkpoint.pth"
wget -O ${PYTORCH_MODEL} ${ZENODO_PYTORCH}

echo "Begining validation..."
cd /workspace
for ITEM in "${CHECKSUMS[@]}"; do
    FILENAME=$(echo ${ITEM} | cut -d',' -f1)
    CHECKSUM=$(echo ${ITEM} | cut -d',' -f2)
    bash run_validation.sh ${FILENAME} ${CHECKSUM}
done
