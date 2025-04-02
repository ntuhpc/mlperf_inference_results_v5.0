#!/bin/bash

if [ -z "${DATA_DIR}" ]; then
    echo "Path to dataset not set. Please set it:"
    echo "export DATA_DIR=</path/to/openimages>"
    exit 1
fi

if [ -z "${RN50_START}" ]; then
    echo "Path to resnet50_start model not set. Please set it:"
    export "RN50_START=</path/to/resnet50_start.pth>"
    exit 1
fi

if [ -z "${RN50_END}" ]; then
    echo "Path to resnet50_end model not set. Please set it:"
    export "RN50_END=</path/to/resnet50_end.pth>"
    exit 1
fi

if [ -z "${RN50_FULL}" ]; then
    echo "Path to resnet50_full model not set. Please set it:"
    export "RN50_FULL=</path/to/resnet50_full.pth>"
    exit 1
fi

CONDA_ENV_NAME=rn50-mlperf

export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"

export LD_PRELOAD=${CONDA_PREFIX}/lib/libjemalloc.so

export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
export KMP_BLOCKTIME=1
export $KMP_SETTING

CUR_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
APP=${CUR_DIR}/build/bin/mlperf_runner

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib

if [ -e "mlperf_log_summary.txt" ]; then
    rm mlperf_log_summary.txt
fi

export number_cores=`lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l`
export number_numa=`lscpu | grep "NUMA node(s)" | rev | cut -d' ' -f1 | rev`
if [ "${number_cores}" == "256" ] && [ "${number_numa}" == "6" ]; then
        EXCLUDE="--exclude_cores 0,42,43,128,171"
        APP="-C 0-41,44-85,86-127,129-170,172-213,214-255 -m 0,1,2,3,4,5 ${APP}"
elif [ "${number_cores}" == "172" ] && [ "${number_numa}" == "4" ]; then
        EXCLUDE="--exclude_cores 0,42,43,86,129"
        APP="-C 0-41,44-85,87-128,130-171 -m 0,1,2,3 ${APP}"
else
        EXCLUDE=""
fi

numactl ${APP} --scenario Server  \
    --mode Performance  \
    --user_conf ${USER_CONF} \
    --model_name resnet50 \
    --rn50-part1 ${RN50_START} \
    --rn50-part3 ${RN50_END} \
    --rn50-full-model ${RN50_FULL} \
    --data_path ${DATA_DIR} \
    --num_instance $num_instances \
    --warmup_iters 50 \
    --cpus_per_instance $CPUS_PER_INSTANCE \
    --total_sample_count 50000 \
    --batch_size 4 \
    ${EXCLUDE}

if [ -e "mlperf_log_summary.txt" ]; then
    cat mlperf_log_summary.txt
fi
