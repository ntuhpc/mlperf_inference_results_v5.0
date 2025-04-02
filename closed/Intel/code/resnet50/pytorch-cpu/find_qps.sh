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
export USER_CONF=/workspace/user.conf


workload_specific_parameters () {
  source config/workload.conf
  export WORKLOAD="resnet50"
  export MODEL="resnet50"
  export IMPL="pytorch-cpu"
  export MAX_LATENCY=15000000
  export OFFLINE_DURATION=60000
  export SERVER_DURATION=120000
}

workload_specific_run () {
  export RN50_START=${MODEL_DIR}/resnet50-start-int8-model.pth
  export RN50_END=${MODEL_DIR}/resnet50-end-int8-model.pth
  export RN50_FULL=${MODEL_DIR}/resnet50-full.pth

  # Set HW specific qps settings: either manually-defined, GNR, or [default] EMR.
  export number_cores=`lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l`
  export number_numa=`lscpu | grep "NUMA node(s)" | rev | cut -d' ' -f1 | rev`
  if [ "${OFFLINE_QPS}" != "0" ] || [ "${SERVER_QPS}" != "0" ]; then
      echo "*.Offline.target_qps = ${OFFLINE_QPS}" > ${USER_CONF}
      echo "*.Server.target_qps = ${SERVER_QPS}" >> ${USER_CONF}
  elif [ "${number_cores}" == "256" ] && [ "${number_numa}" == "6" ]; then
      cp ${USER_CONF}.GNR128 ${USER_CONF}
      export number_cores=251
  elif [ "${number_cores}" == "172" ] && [ "${number_numa}" == "4" ]; then
      cp ${USER_CONF}.GNR86 ${USER_CONF}
      export number_cores=169
  else
      cp ${USER_CONF}.EMR ${USER_CONF}
  fi

  echo "*.Offline.min_duration = ${OFFLINE_DURATION}" >> ${USER_CONF}
  echo "*.Server.min_duration = ${SERVER_DURATION}" >> ${USER_CONF}

  # Set mode-specific settings and launch: Server/Offline
  if [ "$SCENARIO" == "Server" ]; then
      export CPUS_PER_INSTANCE=2
      export num_instances=$((number_cores/CPUS_PER_INSTANCE))
      echo "Run ${MODEL} (${SCENARIO} Performance)."
      bash run_server.sh
  else
      export CPUS_PER_INSTANCE=1
      export num_instances=$((number_cores/CPUS_PER_INSTANCE))
      echo "Run ${MODEL} (${SCENARIO} Performance)."
      bash run_offline.sh 256
  fi
}

initialize () {
  if [ -f /workspace/audit.config ]; then
      rm /workspace/audit.config
  fi
  bash run_clean.sh
}

run_offline () {
    workload_specific_run 2>&1 > /dev/null
    mv mlperf_log_summary.txt ${OUTPUT_DIR}
    THROUGHPUT=$(cat ${OUTPUT_DIR}/mlperf_log_summary.txt | grep "Samples per second" | rev | cut -d' ' -f1 | rev)
    echo "${THROUGHPUT}"
}

run_server () {
    workload_specific_run 2>&1 > /dev/null
    mv mlperf_log_summary.txt ${OUTPUT_DIR}
    THROUGHPUT=$(cat ${OUTPUT_DIR}/mlperf_log_summary.txt | grep "Completed samples per second" | rev | cut -d' ' -f1 | rev)
    LATENCY=$(cat ${OUTPUT_DIR}/mlperf_log_summary.txt | grep "99.00 percentile latency" | rev | cut -d' ' -f1 | rev)
    echo "${THROUGHPUT},${LATENCY}"
}

workload_specific_parameters
initialize

OUTPUT_DIR=${LOG_DIR}/tmp
mkdir -p ${OUTPUT_DIR}

if [ "$SCENARIO" == "Server" ]; then
    SERVER_RETURN=$(run_server)
    THROUGHPUT=$(echo ${SERVER_RETURN} | cut -d',' -f1)
    LATENCY=$(echo ${SERVER_RETURN} | cut -d',' -f2)
    SERVER_QPS=$(cat ${USER_CONF} | grep "*.Server.target_qps" | tail -1 | rev | cut -d' ' -f1 | rev)
    echo "SERVER_INITIAL: ${SERVER_QPS} ${THROUGHPUT} ${LATENCY}"
    while [ $(echo "${LATENCY} > ${MAX_LATENCY}" | bc -l) == "1" ]; do
        SERVER_QPS=$(echo "${SERVER_QPS} * 0.95" | bc -l)
        SERVER_RETURN=$(run_server)
        THROUGHPUT=$(echo ${SERVER_RETURN} | cut -d',' -f1)
        LATENCY=$(echo ${SERVER_RETURN} | cut -d',' -f2)
        SERVER_QPS=$(cat ${USER_CONF} | grep "*.Server.target_qps" | tail -1 | rev | cut -d' ' -f1 | rev)
        echo "SERVER_INTERMEDIATE: ${SERVER_QPS} ${THROUGHPUT} ${LATENCY}"
    done
    while [ $(echo "${LATENCY} < ${MAX_LATENCY}" | bc -l) == "1" ]; do
        PASSING_SERVER_QPS=${SERVER_QPS}
        PASSING_THROUGHPUT=${THROUGHPUT}
        PASSING_LATENCY=${LATENCY}
        SERVER_QPS=$(echo "${PASSING_SERVER_QPS} * 1.01" | bc -l)
        SERVER_RETURN=$(run_server)
        THROUGHPUT=$(echo ${SERVER_RETURN} | cut -d',' -f1)
        LATENCY=$(echo ${SERVER_RETURN} | cut -d',' -f2)
        echo "SERVER_INTERMEDIATE: ${SERVER_QPS} ${THROUGHPUT} ${LATENCY}"
    done
    echo "SERVER_FINAL: ${PASSING_SERVER_QPS} ${PASSING_THROUGHPUT} ${PASSING_LATENCY}"
else
    MAX_THROUGHPUT=0
    THROUGHPUT=$(run_offline)
    echo "OFFLINE_INITIAL: ${THROUGHPUT}"
    while [ $(echo "${THROUGHPUT} > 1.05 * ${MAX_THROUGHPUT}" | bc -l) == "1" ]; do
        MAX_THROUGHPUT=${THROUGHPUT}
        THROUGHPUT=$(run_offline)
        echo "OFFLINE_INTERMEDIATE: ${THROUGHPUT}"
    done
    echo "OFFLINE_FINAL: ${THROUGHPUT}"
fi
