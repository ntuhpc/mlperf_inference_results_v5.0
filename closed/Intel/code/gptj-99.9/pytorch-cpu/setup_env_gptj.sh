# source /miniforge/bin/activate
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:$LD_PRELOAD
export TP_SIZE=1

export NUM_NUMA_NODES=$(lscpu | grep "NUMA node(s)" | awk '{print $NF}')
export NUM_CORES=$(($(lscpu | grep "Socket(s):" | awk '{print $2}') * $(lscpu | grep "Core(s) per socket:" | awk '{print $4}')))
export CORES_PER_NODE=$(($NUM_CORES / $NUM_NUMA_NODES))

export BATCH_SIZE_OFFLINE=32
export BATCH_SIZE_SERVER=64

MEM_AVAILABLE=$(free -g | awk '/Mem:/ {print $7}')
MEM_MODEL=12
export MEM_CACHE_OFFLINE=$((5 * ($BATCH_SIZE_OFFLINE / 4)))
export MEM_CACHE_SERVER=$((5 * ($BATCH_SIZE_SERVER / 4)))
MEM_OFFLINE=$(($MEM_MODEL + MEM_CACHE_OFFLINE))
MEM_PER_NODE=$(($MEM_AVAILABLE / $NUM_NUMA_NODES))
INSTS_PER_NODE_OFFLINE=$(($MEM_PER_NODE / $MEM_OFFLINE))
export CORES_PER_INST_OFFLINE=$(($CORES_PER_NODE / $INSTS_PER_NODE_OFFLINE))
export CORES_PER_INST_SERVER=$CORES_PER_NODE

#export CORES_PER_INST_OFFLINE=6
echo $CORES_PER_INST_OFFLINE
echo $CORES_PER_INST_SERVER

# export OMP_WAIT_POLICY=active # issue with tp
export IPEX_WOQ_GEMM_LOOP_SCHEME=ACB
export VLLM_CPU_VNNI_VALUE_CACHE_LAYOUT=1
unset VLLM_CPU_VNNI_VALUE_CACHE_LAYOUT
#export CHECKPOINT_PATH=/model/gpt-j-checkpoint-mlperf-autoround-w4g128-gpu
export CHECKPOINT_PATH=/model/checkpoint-final-autoround-w4g128-cpu
# export CHECKPOINT_PATH=/model/gpt-j/checkpoint-final
export DATASET_PATH=/data/cnn_dailymail_validation.json
export CALIBRATION_DATA_PATH=/data/cnn_dailymail_calibration.json
