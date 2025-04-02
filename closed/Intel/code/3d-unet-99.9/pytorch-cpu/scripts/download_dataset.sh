#!/bin/bash

CHECKSUMS=(
           /data/kits19/data,ffe23bbbe5dc5e8581750c425d7f814e
          )

echo "Downloading dataset..."
export DATA_DIR=/data
cd ${DATA_DIR}
git clone https://github.com/neheller/kits19
cd kits19
python3 -m starter_code.get_imaging

echo "Begining validation..."
cd /workspace
for ITEM in "${CHECKSUMS[@]}"; do
    FILENAME=$(echo ${ITEM} | cut -d',' -f1)
    CHECKSUM=$(echo ${ITEM} | cut -d',' -f2)
    bash run_validation.sh ${FILENAME} ${CHECKSUM}
done
