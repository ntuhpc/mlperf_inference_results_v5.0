#!/bin/bash
set -e

SETUP_DIR=$(dirname -- $0)
VLLM_IMAGE_NAME=rocm/vllm-dev:nightly_main_20250203_hipblaslt_6b6a724
MLPERF_IMAGE_NAME=llmboost_mlperf_inference:5.0-base
MLPERF_FINAL_IMAGE_NAME=llmboost_mlperf_inference:5.0

docker build --no-cache --build-arg BASE_IMAGE=${VLLM_IMAGE_NAME} -f "$SETUP_DIR/Dockerfile.mlperf" -t ${MLPERF_IMAGE_NAME} "$SETUP_DIR/.."
docker build --no-cache --build-arg BASE_IMAGE=${MLPERF_IMAGE_NAME} --build-arg VLLM_DIR="$SETUP_DIR/vllm" -f "$SETUP_DIR/Dockerfile.vllm" -t ${MLPERF_FINAL_IMAGE_NAME} "$SETUP_DIR/.."
