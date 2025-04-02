CHECKPOINT_PATH="${CHECKPOINT_PATH:-/model/gpt-j-checkpoint-mlperf-autoround-w4g128-gpu}"
DATASET_PATH="${DATASET_PATH:-/data/cnn_dailymail/validation-data/cnn_dailymail_validation.json}"

export CORES_PER_INST=$CORES_PER_INST_SERVER
BATCH_SIZE=$BATCH_SIZE_SERVER
# In case cores per node is not divisable by cores per inst. E.g.: 30 cores/node, 8 cores/inst
# It can't handle the case where different nodes have different number of insts
INSTS_PER_NODE=$(($CORES_PER_NODE / $CORES_PER_INST))
NUM_INSTS=$(($INSTS_PER_NODE * $NUM_NUMA_NODES))

TOTAL_SAMPLE_COUNT=13368
OUTPUT_DIR=/workspace/output_logs

mkdir -p $OUTPUT_DIR

VLLM_CPU_KVCACHE_SPACE=$MEM_CACHE_SERVER \
OMP_NUM_THREADS=$CORES_PER_INST \
NUM_NUMA_NODES=$NUM_NUMA_NODES \
NUM_INSTS=$NUM_INSTS \
INSTS_PER_NODE=$INSTS_PER_NODE \
python -u main.py --scenario Server \
        --model-path ${CHECKPOINT_PATH} \
        --workload-name gptj \
        --accuracy \
        --user-conf user.conf \
        --total-sample-count ${TOTAL_SAMPLE_COUNT} \
        --dataset-path ${DATASET_PATH} \
        --output-log-dir ${OUTPUT_DIR} \
        --dtype "bfloat16" \
        --batch-size $BATCH_SIZE \
        --num-workers $NUM_INSTS \
	--quantized \
        --warmup \
        --device cpu 2>&1 | tee ${OUTPUT_DIR}/run.log

python3 evaluation.py --model-name-or-path ${CHECKPOINT_PATH} \
        --mlperf-accuracy-file ${OUTPUT_DIR}/mlperf_log_accuracy.json \
        --dataset-file ${DATASET_PATH} 2>&1 | tee -a ${OUTPUT_DIR}/accuracy.txt
