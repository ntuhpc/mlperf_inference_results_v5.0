#!/bin/bash

CPUS_PER_INSTANCE=${CPUS_PER_INSTANCE:-8}
START_CORE=${START_CORE:-0}
NUM_WARMUP=${NUM_WARMUP:-100}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_INSTANCES=${NUM_INSTANCES:-32}

if [ -z "${DATA_DIR}" ]; then
    echo "Path to dataset not set. Please set it:"
    echo "export DATA_DIR=</path/to/openimages>"
    exit 1
fi

if [ -z "${MODEL_PATH}" ]; then
    echo "Path to trained checkpoint not set. Please set it:"
    echo "export MODEL_PATH=</path/to/retinanet-int8-model.pth>"
    exit 1
fi

export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"

export LD_PRELOAD=${CONDA_PREFIX}/lib/libjemalloc.so

export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
export KMP_BLOCKTIME=1
export $KMP_SETTING

CUR_DIR=${PWD}
APP=${CUR_DIR}/build/bin/mlperf_runner

if [ -e "mlperf_log_summary.txt" ]; then
    rm mlperf_log_summary.txt
fi

USER_CONF=user.conf

${APP} --scenario Server \
    --mode Performance \
    --user_conf ${USER_CONF} \
    --model_name retinanet \
    --model_path ${MODEL_PATH} \
    --data_path ${DATA_DIR} \
    --num_instance ${NUM_INSTANCES} \
    --start_core ${START_CORE} \
    --warmup_iters ${NUM_WARMUP} \
    --cpus_per_instance $CPUS_PER_INSTANCE \
    --total_sample_count 24781 \
    --batch_size $BATCH_SIZE


if [ -e "mlperf_log_summary.txt" ]; then
    cat mlperf_log_summary.txt
fi
