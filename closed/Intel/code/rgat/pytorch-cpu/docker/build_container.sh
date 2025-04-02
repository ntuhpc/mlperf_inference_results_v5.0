#!/bin/env bash

WORKLOAD="r-gat"
ARCH="cpu"
RELEASE="5.0_mk"
REGISTRY="amr-registry.caas.intel.com/aiops/mlperf"

export DOCKER_BUILD_ARGS="--build-arg ftp_proxy=${ftp_proxy} --build-arg FTP_PROXY=${FTP_PROXY} --build-arg http_proxy=${http_proxy} --build-arg HTTP_PROXY=${HTTP_PROXY} --build-arg https_proxy=${https_proxy} --build-arg HTTPS_PROXY=${HTTPS_PROXY} --build-arg no_proxy=${no_proxy} --build-arg NO_PROXY=${NO_PROXY} --build-arg socks_proxy=${socks_proxy} --build-arg SOCKS_PROXY=${SOCKS_PROXY}"

VERSION=${ARCH}_${WORKLOAD}_${RELEASE}
export IMAGE_NAME="${REGISTRY}:${VERSION}"

echo "Building ${WORKLOAD} ${ARCH} MLPerf workflow container"
DOCKER_BUILDKIT=1 docker build --progress=plain --network="host" ${DOCKER_BUILD_ARGS} -f docker/Dockerfile -t ${IMAGE_NAME} .
