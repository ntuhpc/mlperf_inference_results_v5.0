# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
#
# THIS IS A GENERATED DOCKERFILE.
#
# This file was assembled from multiple pieces, whose use is documented
# throughout. Please refer to the TensorFlow dockerfiles documentation
# for more information.

ARG BASE_IMAGE=ubuntu
ARG BASE_TAG=22.04
FROM ${BASE_IMAGE}:${BASE_TAG} as dev-base

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

# Copy code into docker
COPY ./ /workspace/

# Install ubuntu prerequisites and graphics UMD components
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get autoclean && \
    apt-get autoremove -y && \
    apt-get install -y --no-install-recommends \
                    cmake \
                    libblas-dev \
                    liblapack-dev \
                    autoconf \
                    unzip \
                    wget \
                    git \
                    vim \
                    ca-certificates \
                    pkg-config \
                    build-essential \
                    numactl \
                    libtcmalloc-minimal4 \
                    sudo \
                    python3.10 \
                    python3-pip \
                    python3.10-dev \
                    python-is-python3\
                    curl \
                    gcc-12 \
                    g++-12 \
                    bc && \
                    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12 && \
                    rm -rf /var/lib/apt/lists/*

# Install python packages
RUN echo "alias ll='ls -l'" >> ~/.bashrc
RUN echo "alias ll='ls -l'" >> /root/.bashrc

# pip packages
ARG PYTHON_VERSION=3.10
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install cmake==3.28.1 ray==2.40.0 transformers==4.41.2 tokenizers==0.19.1 accelerate==0.29.3 trl==0.8.6 py-libnuma==1.2 evaluate==0.4.3 protobuf==5.26.1 pandas==2.0.3 omegaconf==2.3.0 rouge-score==0.1.2 sentencepiece==0.1.99 py-cpuinfo==9.0.0 ninja==1.11.1 starlette==0.40.0 auto-round==0.3 setuptools-scm==8.1.0 aiohttp==3.10.11 && \
    pip install https://download.pytorch.org/whl/nightly/cpu/torch-2.6.0.dev20241010%2Bcpu-cp310-cp310-linux_x86_64.whl
 

# mlperf loadgen
RUN --mount=type=cache,target=/root/.cache/pip \
    mkdir -p /workspace/third_party && \
    cd /workspace/third_party && \
    git clone https://github.com/mlcommons/inference.git mlperf-inference && \
    cd mlperf-inference && \
    git checkout master && \
    cd loadgen && \
    python3 -m pip install . 

# vllm
RUN --mount=type=cache,target=/root/.cache/pip \
    cd /workspace/third_party && \
    git clone https://github.com/vllm-project/vllm.git vllm-xpu && cd vllm-xpu && git checkout 5116274 && git fetch origin pull/12141/head:wip_mlperf_rebased && git checkout wip_mlperf_rebased && \
    VLLM_TARGET_DEVICE=cpu python3 setup.py install && \
    rm -rf /workspace/third_party/vllm-xpu

# AutoAWQ
RUN --mount=type=cache,target=/root/.cache/pip \
    cd /workspace/third_party && \
    git clone https://github.com/casper-hansen/AutoAWQ.git && \
    cd AutoAWQ && \
    git checkout f0321ee && \
    git apply /workspace/AWQ.patch && \
    python3 -m pip install . && \
    rm -rf /workspace/third_party/AutoAWQ

# Intel extension for PyTorch
RUN --mount=type=cache,target=/root/.cache/pip \
    set -a && . /workspace/setup_env_gptj.sh && set +a && \
    cd /workspace/third_party && \
    git clone https://github.com/intel/intel-extension-for-pytorch.git ipex-cpu && cd ipex-cpu && git checkout 9b22d84 && git submodule update --init --recursive && \
    python3 setup.py install && \
    rm -rf /workspace/third_party/ipex-cpu

# Needed to prevent a version incompatibility from previous pulled dependency
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install triton==3.1.0
