#!/bin/bash
set -xeu

# Timestamp
export LAB_TS=`date +%m%d-%H%M`

# Host side
export LAB_MLPINF=$(dirname $(dirname $(readlink -fm -- $0)))

export LAB_MODEL="${LAB_MODEL:-/models/}"

# Docker
export LAB_DKR_CTNAME_BASE=mlperf.vllm.$(whoami)
export LAB_DKR_CTNAME=${LAB_DKR_CTNAME_BASE}.${LAB_TS}

docker run --rm -it --ipc=host --network=host --privileged --cap-add=CAP_SYS_ADMIN --device=/dev/kfd --device=/dev/dri --device=/dev/mem \
        --group-add render --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
        --name=${LAB_DKR_CTNAME} \
        -v ${LAB_MODEL}:/models/ \
        -v ${LAB_MLPINF}:/workspace/ \
        llmboost_mlperf_inference:5.0
