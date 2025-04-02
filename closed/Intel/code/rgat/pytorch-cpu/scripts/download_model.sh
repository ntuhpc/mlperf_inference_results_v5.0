#!/bin/bash
# Redirect all output to 'output.log'
exec > /logs/data_model_output.log 2>&1
#download and prepare data

echo "Downloading model..."
export MODEL_DIR=/model
rclone config create mlc-inference s3 provider=Cloudflare access_key_id=f65ba5eef400db161ea49967de89f47b secret_access_key=fbea333914c292b854f14d3fe232bad6c5407bf0ab1bebf78833c2b359bdfd2b endpoint=https://c2686074cb2caf5cbaf6d134bdba8b47.r2.cloudflarestorage.com
rclone copy mlc-inference:mlcommons-inference-wg-public/R-GAT/RGAT.pt $MODEL_DIR -P

CHECKSUMS=(
            /model/RGAT.pt,81225f862db03e6042f68b088b84face
          )
echo "Verifying md5sum of the model"
pushd /workspace
    for ITEM in "${CHECKSUMS[@]}"; do
        FILENAME=$(echo ${ITEM} | cut -d',' -f1)
        CHECKSUM=$(echo ${ITEM} | cut -d',' -f2)
        bash run_validation.sh ${FILENAME} ${CHECKSUM}
    done
popd