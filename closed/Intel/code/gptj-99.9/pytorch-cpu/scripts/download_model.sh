#!/bin/bash

CHECKSUMS=(
            /model/gpt-j-checkpoint,4ff5e3910d41750fe3ca64af07d9306d
          )

echo "Downloading model..."
export MODEL_DIR=/model
cd ${MODEL_DIR}
wget https://cloud.mlcommons.org/index.php/s/QAZ2oM94MkFtbQx/download -O "${MODEL_DIR}/gpt-j-checkpoint.zip"
unzip -j gpt-j-checkpoint.zip "gpt-j/checkpoint-final/*" -d "${MODEL_DIR}/gpt-j-checkpoint"
rm -f gpt-j-checkpoint.zip

echo "Begining validation..."
cd /workspace
for ITEM in "${CHECKSUMS[@]}"; do
    FILENAME=$(echo ${ITEM} | cut -d',' -f1)
    CHECKSUM=$(echo ${ITEM} | cut -d',' -f2)
    bash run_validation.sh ${FILENAME} ${CHECKSUM}
done
