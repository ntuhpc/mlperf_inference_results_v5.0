export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib

CHECKPOINT_PATH=/model/gpt-j-checkpoint/
CALIBRATION_DATA_PATH=/data/cnn_dailymail_calibration.json
NUM_GROUPS=128
NUM_SAMPLES=512
ITERS=200
NUM_CORES=$(($(lscpu | grep "Socket(s):" | awk '{print $2}') * $(lscpu | grep "Core(s) per socket:" | awk '{print $4}')))
END_CORE=$(($NUM_CORES - 1))

cd /workspace
numactl -C 0-$END_CORE python -u quantize_autoround.py \
	--model_name ${CHECKPOINT_PATH} \
	--dataset ${CALIBRATION_DATA_PATH} \
	--group_size ${NUM_GROUPS} \
	--bits 4 \
	--iters ${ITERS} \
	--nsamples ${NUM_SAMPLES} \
	--device cpu \
	--deployment_device "cpu" \
	--scale_dtype 'fp32' \
	--seqlen 1919 \
	--train_bs 1 \
	--lr 2.5e-3 \
	--disable_eval \
	--output_dir /model/ 2>&1 | tee autoround_log_${NUM_GROUPS}g_${NUM_SAMPLES}nsamples_${ITERS}iters.log
