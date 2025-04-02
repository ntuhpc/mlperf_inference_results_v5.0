#!/bin/bash

export DATA_DIR="${DATA_DIR:-${PWD}/data}"
export MODEL_DIR="${MODEL_DIR:-${PWD}/model}"
export LOG_DIR="${LOG_DIR:-${PWD}/logs}"

docker run --privileged -it --rm \
        --ipc=host --net=host --cap-add=ALL \
        --device /dev/dri:/dev/dri \
        -v /dev/dri/by-path:/dev/dri/by-path \
        -v /lib/modules:/lib/modules \
        -v ${DATA_DIR}:/data \
        -v ${MODEL_DIR}:/model \
	-v ${LOG_DIR}:/logs \
        amr-registry.caas.intel.com/aiops/mlperf:cpu_r-gat_5.0 /bin/bash
