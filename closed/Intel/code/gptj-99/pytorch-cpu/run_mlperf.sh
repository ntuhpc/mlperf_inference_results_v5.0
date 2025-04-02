#!/bin/bash

# Controls workload mode
export SCENARIO="${SCENARIO:-Offline}"
export MODE="${MODE:-Performance}"
export OFFLINE_QPS="${OFFLINE_QPS:-0}"
export SERVER_QPS="${SERVER_QPS:-0}"

# Setting environmental paths
export DATA_DIR=/data
export MODEL_DIR=/model
export LOG_DIR=/logs
export RESULTS_DIR=${LOG_DIR}/results
export COMPLIANCE_DIR=${LOG_DIR}/compliance
export COMPLIANCE_SUITE_DIR=/workspace/inference/compliance/nvidia

workload_specific_parameters () {
  export WORKLOAD="gptj-99.9"
  export MODEL="gpt-j"
  export IMPL="pytorch-cpu"
  export COMPLIANCE_TESTS=""
}

workload_specific_run () {
  export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:$LD_PRELOAD
  export TP_SIZE=1

  export NUM_NUMA_NODES=$(lscpu | grep "NUMA node(s)" | awk '{print $NF}')
  export NUM_CORES=$(($(lscpu | grep "Socket(s):" | awk '{print $2}') * $(lscpu | grep "Core(s) per socket:" | awk '{print $4}')))
  export CORES_PER_NODE=$(($NUM_CORES / $NUM_NUMA_NODES))
  export IPEX_WOQ_GEMM_LOOP_SCHEME=ACB
  export CHECKPOINT_PATH=/model/gpt-j-checkpoint-autoround-w4g128-cpu
  export DATASET_PATH=/data/cnn_dailymail_validation.json

  number_cores=${NUM_CORES}
  number_numa=${NUM_NUMA_NODES}
  if [ "${OFFLINE_QPS}" != "0" ] || [ "${SERVER_QPS}" != "0" ]; then
      echo "*.Offline.target_qps = ${OFFLINE_QPS}" > /workspace/user.conf
      echo "*.Server.target_qps = ${SERVER_QPS}" >> /workspace/user.conf
  elif [ "${number_cores}" == "256" ] && [ "${number_numa}" == "6" ]; then
      cp /workspace/systems/user.conf.GNR_128C /workspace/user.conf
      export SYSTEM="1-node-2S-GNR_128C"
  elif [ "${number_cores}" == "172" ] && [ "${number_numa}" == "4" ]; then
      cp /workspace/systems/user.conf.GNR_86C /workspace/user.conf
      export SYSTEM="1-node-2S-GNR_86C"
  else
      cp /workspace/systems/user.conf.EMR /workspace/user.conf
      export SYSTEM="1-node-2S-EMR"
  fi
  export USER_CONF=user.conf

  if [ "$SCENARIO" == "Server" ]; then
      export BATCH_SIZE_SERVER=64
      export MEM_CACHE_SERVER=$((5 * ($BATCH_SIZE_SERVER / 4)))
      export CORES_PER_INST_SERVER=$CORES_PER_NODE
      if [ "${MODE}" == "Accuracy" ]; then
          echo "Run ${MODEL} (${SCENARIO} Accuracy)."
          bash run_accuracy_server_gptj.sh
      else
          echo "Run ${MODEL} (${SCENARIO} Performance)."
	  bash run_server_gptj.sh
      fi
  else
      export BATCH_SIZE_OFFLINE=32
      export MEM_CACHE_OFFLINE=$((5 * ($BATCH_SIZE_OFFLINE / 4)))
      export MEM_AVAILABLE=$(free -g | awk '/Mem:/ {print $7}')
      export MEM_PER_NODE=$(($MEM_AVAILABLE / $NUM_NUMA_NODES))
      export MEM_MODEL=12
      export MEM_OFFLINE=$(($MEM_MODEL + MEM_CACHE_OFFLINE))
      export INSTS_PER_NODE_OFFLINE=$(($MEM_PER_NODE / $MEM_OFFLINE))
      export CORES_PER_INST_OFFLINE=$(($CORES_PER_NODE / $INSTS_PER_NODE_OFFLINE))
      if [ "${MODE}" == "Accuracy" ]; then
          echo "Run ${MODEL} (${SCENARIO} Accuracy)."
          bash run_accuracy_offline_gptj.sh
      else
          echo "Run ${MODEL} (${SCENARIO} Performance)."
          bash run_offline_gptj.sh
      fi
  fi

  if [ "${MODE}" == "Accuracy" ]; then
      cd /workspace/output_logs
      mv mlperf_log_accuracy.json mlperf_log_detail.txt mlperf_log_summary.txt accuracy.txt /workspace/
  else
      cd /workspace/output_logs
      mv mlperf_log_accuracy.json mlperf_log_detail.txt mlperf_log_summary.txt /workspace/
  fi

  cd /workspace
}

initialize () {
  if [ -f /workspace/audit.config ]; then
      rm /workspace/audit.config
  fi
  bash run_clean.sh
}

prepare_suplements () {
  # Ensure /logs/systems is populated or abort process.
  export SYSTEMS_DIR=${LOG_DIR}/systems
  mkdir -p ${SYSTEMS_DIR}
  cp /workspace/systems/${SYSTEM}.json ${SYSTEMS_DIR}/

  # Populate /logs/code directory
  export CODE_DIR=${LOG_DIR}/code/${WORKLOAD}/${IMPL}
  mkdir -p ${CODE_DIR}
  cp -r /workspace/README.md ${CODE_DIR}/

  # Populate /logs/measurements directory (No distibution between Offline and Server modes)
  export MEASUREMENTS_DIR=${LOG_DIR}/measurements/${SYSTEM}/${WORKLOAD}
  mkdir -p ${MEASUREMENTS_DIR}/${SCENARIO}
  cp /workspace/measurements.json ${MEASUREMENTS_DIR}/${SCENARIO}/${SYSTEM}.json
  cp /workspace/README.md ${MEASUREMENTS_DIR}/${SCENARIO}/
  cp /workspace/user.conf ${MEASUREMENTS_DIR}/${SCENARIO}/
  cp /workspace/scripts/run_calibration.sh ${MEASUREMENTS_DIR}/${SCENARIO}/
}

workload_specific_parameters

# Setting compliance test list (if applicable)
if [[ "${COMPLIANCE_TESTS}" == *"${MODE}"* ]]; then
    export COMPLIANCE_TESTS="${MODE}"
    export MODE="Compliance"
fi

if [ "${MODE}" == "Performance" ]; then
    initialize
    workload_specific_run
    OUTPUT_DIR=${RESULTS_DIR}/${SYSTEM}/${WORKLOAD}/${SCENARIO}/performance/run_1
    mkdir -p ${OUTPUT_DIR}
    mv mlperf_log_accuracy.json mlperf_log_detail.txt mlperf_log_summary.txt ${OUTPUT_DIR}
    prepare_suplements
elif [ "${MODE}" == "Accuracy" ]; then
    initialize
    workload_specific_run
    OUTPUT_DIR=${RESULTS_DIR}/${SYSTEM}/${WORKLOAD}/${SCENARIO}/accuracy
    mkdir -p ${OUTPUT_DIR}
    mv mlperf_log_accuracy.json mlperf_log_detail.txt mlperf_log_summary.txt accuracy.txt ${OUTPUT_DIR}
elif [ "${MODE}" == "Compliance" ]; then
    for TEST in ${COMPLIANCE_TESTS}; do
        initialize
        echo "Running compliance ${TEST} ..."

        if [ "$TEST" == "TEST01" ]; then
                cp ${COMPLIANCE_SUITE_DIR}/${TEST}/${MODEL}/audit.config .
        else
                cp ${COMPLIANCE_SUITE_DIR}/${TEST}/audit.config .
        fi

        workload_specific_run
        OUTPUT_DIR=${COMPLIANCE_DIR}/${SYSTEM}/${WORKLOAD}/${SCENARIO}/${TEST}/output
        mkdir -p ${OUTPUT_DIR}
        mv mlperf_log_accuracy.json mlperf_log_detail.txt mlperf_log_summary.txt ${OUTPUT_DIR}

        RESULTS=${RESULTS_DIR}/${SYSTEM}/${WORKLOAD}/${SCENARIO}
        if ! [ -d ${RESULTS} ]; then
            echo "[ERROR] Compliance run could not be verified due to unspecified or non-existant RESULTS dir: ${RESULTS}"
            exit
        else
            COMPLIANCE_VERIFIED=${COMPLIANCE_DIR}/${SYSTEM}/${WORKLOAD}/${SCENARIO}
            python ${COMPLIANCE_SUITE_DIR}/${TEST}/run_verification.py -r ${RESULTS} -c ${OUTPUT_DIR} -o ${COMPLIANCE_VERIFIED}
            rm -r ${OUTPUT_DIR}
        fi
    done
else
    echo "[ERROR] Missing value for MODE. Options: Performance, Accuracy, Compliance"
fi
