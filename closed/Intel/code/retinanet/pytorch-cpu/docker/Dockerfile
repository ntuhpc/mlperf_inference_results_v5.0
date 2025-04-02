# syntax = docker/dockerfile:experimental
# based onhttps://github.com/pytorch/pytorch/blob/master/Dockerfile
# 
# NOTE: To build this you will need a docker version > 18.06 with
#       experimental enabled and DOCKER_BUILDKIT=1
#
#       If you do not use buildkit you are not going to have a good time
#
#       For reference: 
#           https://docs.docker.com/develop/develop-images/build_enhancements/

ARG BASE_IMAGE=intel/intel-optimized-pytorch:mlperf-inference-4.1-retinanet
FROM ${BASE_IMAGE} AS dev-base
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get autoclean && \
    apt-get autoremove -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    libnuma-dev \
    bc \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install cmake==3.27.0

RUN echo "alias ll='ls -l'" >> /root/.bashrc
ENV PATH /opt/conda/bin:$PATH
ENV CONDA_PREFIX "/opt/conda"

RUN cp -r /workspace/retinanet-env /tmp/ && rm -rf /workspace/*
COPY . /workspace
RUN mkdir -p /workspace/retinanet-env; cp -r /tmp/retinanet-env/* /workspace/retinanet-env/

WORKDIR /workspace
RUN cd retinanet-env && \
    rm -rf mlperf_inference && \
    git clone https://github.com/mlcommons/inference.git mlperf_inference && \
    cd mlperf_inference && \
    git checkout master && \
    cd loadgen && mkdir build && cd build && cmake .. && make && cd .. && \
    CFLAGS="-std=c++14" python setup.py install

RUN export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"} && \
    export CC=`which gcc` && export CXX=`which g++` && \
    cd retinanet-env && \
    export IPEX_PATH=${PWD}/ipex-cpu-dev/build/Release/packages/intel_extension_for_pytorch && \
    export LOADGEN_DIR=${PWD}/mlperf_inference/loadgen && \
    export OPENCV_DIR=${PWD}/opencv/build && \
    export RAPIDJSON_INCLUDE_DIR=${PWD}/rapidjson/include && \
    export GFLAGS_DIR=${PWD}/gflags/build && \
    export LIBRARY_PATH=${CONDA_PREFIX}/lib:${LIBRARY_PATH} && \
    export TORCH_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` && \
    cd .. && \
    cmake -DCMAKE_PREFIX_PATH=${TORCH_PATH} \
        -DLOADGEN_DIR=${LOADGEN_DIR} \
        -DOpenCV_DIR=${OPENCV_DIR} \
        -DRapidJSON_INCLUDE_DIR=${RAPIDJSON_INCLUDE_DIR} \
        -Dgflags_DIR=${GFLAGS_DIR} \
        -DINTEL_EXTENSION_FOR_PYTORCH_PATH=${IPEX_PATH} \
        -B${PWD}/build \
        -H${PWD}/src && \
    cmake --build ${PWD}/build --config Release

RUN for FILE in $(cat /workspace/redactions.txt); do rm -rf /workspace/${FILE}; rm -rf /workspace/workload_code/${FILE}; done

ENV MALLOC_CONF "oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"
ENV LD_PRELOAD "/opt/conda/lib/libjemalloc.so":"/opt/conda/lib/libiomp5.so":${LD_PRELOAD}
ENV CONDA_PREFIX "/opt/conda"
ENV MODEL_DIR=/model
ENV DATA_DIR=/data
