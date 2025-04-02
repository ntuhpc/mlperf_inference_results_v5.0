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

ARG BASE_IMAGE=intel/intel-optimized-pytorch:mlperf-inference-4.1-resnet50
FROM ${BASE_IMAGE} AS dev-base

RUN DEBIAN_FRONTEND=noninteractive dnf install -y bc 

RUN cp -r /workspace/rn50-mlperf /tmp/
RUN cp -r /workspace/src/ckernels/3rdparty /tmp/
RUN rm -rf /workspace/*
COPY . /workspace
RUN mkdir -p /workspace/rn50-mlperf; cp -r /tmp/rn50-mlperf/* /workspace/rn50-mlperf/
RUN mkdir -p /workspace/src/ckernels/3rdparty; cp -r /tmp/3rdparty/* /workspace/src/ckernels/3rdparty/

ENV CONDA_PREFIX "/opt/conda"
ARG IPEX_VERSION=v1.12.0
RUN source /opt/rh/gcc-toolset-11/enable && \
    cd /workspace/rn50-mlperf && \
    rm -rf mlperf_inference && \
    git clone https://github.com/mlcommons/inference.git mlperf_inference && \
    cd mlperf_inference && \
    git checkout master && \
    git log -1 && \
    cd loadgen && mkdir build && cd build && cmake .. && make -j$(nproc) && cd .. && \
    CFLAGS="-std=c++14" python setup.py install && cd ../.. && \
    cd /workspace/rn50-mlperf/ && \
    export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"} && \
    export IPEX_PATH=${PWD}/ipex-cpu-dev/build/Release/packages/intel_extension_for_pytorch && \
    export TORCH_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` && \
    export LOADGEN_DIR=${PWD}/mlperf_inference/loadgen && \
    export OPENCV_DIR=${PWD}/opencv/build && \
    export RAPIDJSON_INCLUDE_DIR=${PWD}/rapidjson/include && \
    export GFLAGS_DIR=${PWD}/gflags/build && \
    export ONEDNN_DIR=${PWD}/oneDNN && \
    export LIBRARY_PATH=${LIBRARY_PATH}:${CONDA_PREFIX}/lib && \
    cd ../ && \
    cmake -DCMAKE_PREFIX_PATH=${TORCH_PATH} \
        -DLOADGEN_DIR=${LOADGEN_DIR} \
        -DOpenCV_DIR=${OPENCV_DIR} \
        -DRapidJSON_INCLUDE_DIR=${RAPIDJSON_INCLUDE_DIR} \
        -Dgflags_DIR=${GFLAGS_DIR} \
        -DINTEL_EXTENSION_FOR_PYTORCH_PATH=${IPEX_PATH} \
        -DONEDNN_DIR=${ONEDNN_DIR} \
        -DCMAKE_BUILD_TYPE=Release \
        -B${PWD}/build \
        -H${PWD}/src && \
    cmake --build ${PWD}/build --config Release -j$(nproc)

RUN for FILE in $(cat /workspace/redactions.txt); do rm -rf /workspace/${FILE}; rm -rf /workspace/workload_code/${FILE}; done

WORKDIR /workspace
ENV CONDA_ENV_NAME "/workspace"
ENV CONDA_PREFIX "/opt/conda"
ENV MODEL_DIR=/model
ENV DATA_DIR=/data
