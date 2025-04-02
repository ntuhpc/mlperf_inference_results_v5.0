# based onhttps://github.com/pytorch/pytorch/blob/master/Dockerfile
# 
# NOTE: To build this you will need a docker version > 18.06 with
#       experimental enabled and DOCKER_BUILDKIT=1
#
#       If you do not use buildkit you are not going to have a good time
#
#       For reference: 
#           https://docs.docker.com/develop/develop-images/build_enhancements/

ARG BASE_IMAGE=intel/intel-optimized-pytorch:mlperf-inference-4.1-dlrmv2
FROM ${BASE_IMAGE} AS dev-base

RUN DEBIAN_FRONTEND=noninteractive dnf install -y bc

RUN rm -rf /workspace/*
COPY . /workspace
WORKDIR /workspace
RUN for FILE in $(cat /workspace/redactions.txt); do rm -rf /workspace/${FILE}; rm -rf /workspace/workload_code/${FILE}; done

ARG GCC_MAJOR_VERSION=12
RUN source /opt/rh/gcc-toolset-${GCC_MAJOR_VERSION}/enable && \
    rm -rf inference && \
    git clone --recurse-submodules https://github.com/mlcommons/inference.git inference && \
    cd inference && \
    git checkout master && \
    git submodule update --init --recursive && \
    cd loadgen && \
    CFLAGS="-std=c++14" python setup.py install

RUN pip install -e git+https://github.com/mlperf/logging#egg=mlperf-logging

ENV CONDA_PREFIX "/opt/conda"
ENV MODEL_DIR=/model
ENV DATA_DIR=/data

