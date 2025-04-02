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
export COMPLIANCE_SUITE_DIR=/workspace/mlperf_inference/compliance/nvidia
export USER_CONF=user.conf
workload_specific_parameters () {
  export WORKLOAD="rgat"
  export MODEL="rgat"
  export IMPL="pytorch-cpu"
  export COMPLIANCE_TESTS="TEST01"
}

workload_specific_run () {

  export ENV_DEPS_DIR=/workspace/RGAT-env
  # Set HW specific qps settings: either manually-defined, GNR, or [default] EMR.
  export number_cores=`lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l`
  export number_numa=`lscpu | grep "NUMA node(s)" | rev | cut -d' ' -f1 | rev`
  if [ "${OFFLINE_QPS}" != "0" ] || [ "${SERVER_QPS}" != "0" ]; then
      echo "*.Offline.target_qps = ${OFFLINE_QPS}" > /workspace/${USER_CONF}
      echo "*.Server.target_qps = ${SERVER_QPS}" >> /workspace/${USER_CONF}
  elif [ "${number_cores}" == "256" ] && [ "${number_numa}" == "6" ]; then
      cp /workspace/systems/${USER_CONF}.GNR_128C /workspace/${USER_CONF}
      export SYSTEM="1-node-2S-GNR_128C"
      export NUM_PROC=2 
      export CPUS_PER_PROC=126
      export WORKERS_PER_PROC=3
      export CORE_OFFSET='[1,43,86,128,171,214]'
  elif [ "${number_cores}" == "172" ] && [ "${number_numa}" == "4" ]; then
      cp /workspace/systems/${USER_CONF}.GNR_86C /workspace/${USER_CONF}
      export SYSTEM="1-node-2S-GNR_86C"
      export NUM_PROC=1 
      export CPUS_PER_PROC=172
      export WORKERS_PER_PROC=4
      export CORE_OFFSET='[0,43,86,129]'
  else
      cp /workspace/systems/${USER_CONF}.EMR /workspace/${USER_CONF}
      export SYSTEM="1-node-2S-EMR"
      export NUM_PROC=1 
      export CPUS_PER_PROC=126
      export WORKERS_PER_PROC=4
      export CORE_OFFSET='[1,32,64,96]'
  fi
  export USER_CONF=user.conf
  
  export TMP_DIR=/workspace/output_logs
  if [ "${MODE}" == "Accuracy" ]; then
      echo "Run ${MODEL} (${SCENARIO} Accuracy)."
      bash run_accuracy.sh
      cd ${TMP_DIR}
      mv mlperf_log_accuracy.json mlperf_log_detail.txt mlperf_log_summary.txt accuracy.txt /workspace/
  else
      echo "Run ${MODEL} (${SCENARIO} Performance)."
      bash run_offline.sh
      cd ${TMP_DIR}
      echo "moving log files "
      mv mlperf_log_accuracy.json mlperf_log_detail.txt mlperf_log_summary.txt /workspace/
  fi

  rm -rf ${TMP_DIR}
  cd /workspace
}

initialize () {
  if [ -f /workspace/audit.config ]; then
      rm /workspace/audit.conf
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
                cp ${COMPLIANCE_SUITE_DIR}/${TEST}/${MODEL}/audit.conf .
        else
                cp ${COMPLIANCE_SUITE_DIR}/${TEST}/audit.conf .
        fi
        mv audit.conf audit.config
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
