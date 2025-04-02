#!/bin/bash

VERSION=ww05
export IMAGE_NAME="gptj_vllm:${VERSION}"

echo "Building CPU MLPerf workflow container"

DOCKER_BUILDKIT=1 docker build --no-cache --progress=plain -f docker/Dockerfile -t ${IMAGE_NAME} .
rm -rf third_party
