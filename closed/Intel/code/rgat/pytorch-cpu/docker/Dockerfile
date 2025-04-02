# based onhttps://github.com/pytorch/pytorch/blob/master/Dockerfile
# 
# NOTE: To build this you will need a docker version > 18.06 with
#       experimental enabled and DOCKER_BUILDKIT=1
#
#       If you do not use buildkit you are not going to have a good time
#
#       For reference: 
#           https://docs.docker.com/develop/develop-images/build_enhancements/

ARG BENCHMARK=rgat-99
ARG IMPL=pytorch-cpu-int8
ARG BASE_IMAGE=rockylinux:8.6
ARG PYTHON_VERSION=3.9
ARG GCC_MAJOR_VERSION=12
ARG DGL_BRANCH=88f109f17338d7905d6f5618f0b2b3afc689fd54
ARG TPP_BRANCH=mlperf_gnn_infer

FROM ${BASE_IMAGE} AS dev-base
ARG GCC_MAJOR_VERSION

RUN yum -y update && \
    yum -y upgrade
RUN --mount=type=cache,id=yum-dev,target=/var/cache/yum \
    DEBIAN_FRONTEND=noninteractive dnf install -y \
    ca-certificates \
    git \
    vim \
    numactl \
    cmake \
    sudo \
    wget \
    gcc-c++ \
    gcc-toolset-${GCC_MAJOR_VERSION}-gcc \
    gcc-toolset-${GCC_MAJOR_VERSION}-gcc-c++ \
    && rm -rf /var/lib/yum/lists
RUN echo "source /opt/rh/gcc-toolset-${GCC_MAJOR_VERSION}/enable" >> /root/.bashrc
ENV PATH /opt/conda/bin:$PATH

FROM dev-base as conda
ARG PYTHON_VERSION
RUN wget -O ~/miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" && \
    chmod +x ~/miniforge3.sh && \
    ~/miniforge3.sh -b -p /opt/conda && \
    rm ~/miniforge3.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} && \
    /opt/conda/bin/conda install -y -c conda-forge ninja setuptools tqdm future cmake numpy pyyaml scikit-learn pydot && \
    /opt/conda/bin/conda install -y -c conda-forge gperftools && \
    /opt/conda/bin/conda install -y -c conda-forge llvm-openmp==17.0.6 && \
    /opt/conda/bin/conda install -y pytorch==2.3.1 torchvision torchaudio cpuonly -c pytorch && \
    /opt/conda/bin/conda install -y pybind11 black==22.6.0 clang-format==5.0.1 -c sarcasm && \
    /opt/conda/bin/conda clean -ya

RUN pip install -e git+https://github.com/mlperf/logging@3.0.0-rc2#egg=mlperf-logging
RUN pip install intel-openmp mkl==2024.0
RUN pip install requests

FROM dev-base AS tpp
ARG GCC_MAJOR_VERSION
ARG TPP_BRANCH
ARG BENCHMARK
ARG IMPL
COPY --from=conda /opt/conda /opt/conda
RUN --mount=type=cache,target=/opt/ccache \
    source /opt/rh/gcc-toolset-${GCC_MAJOR_VERSION}/enable && \
    export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"} && \
    git clone https://github.com/libxsmm/tpp-pytorch-extension && \
    cd tpp-pytorch-extension && \
    git checkout ${TPP_BRANCH} && \
    git submodule sync && git submodule update --init --recursive && \
    python setup.py install

FROM dev-base AS dgl
ARG GCC_MAJOR_VERSION
ARG DGL_BRANCH
ARG BENCHMARK
ARG IMPL
COPY --from=tpp /opt/conda /opt/conda
COPY ./dgl.patch dgl.patch
RUN --mount=type=cache,target=/opt/ccache \
    source /opt/rh/gcc-toolset-${GCC_MAJOR_VERSION}/enable && \
    export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"} && \
    git clone --recursive https://github.com/dmlc/dgl.git dgl && \
    cd dgl && \
    git checkout ${DGL_BRANCH} && \
    cd third_party/libxsmm && \
    git checkout main && \
    cd ../../ && \
    git apply --reject --whitespace=fix ../dgl.patch && \
    mkdir build && cd build && \
    cmake -DBUILD_TYPE='release' -DUSE_LIBXSMM=ON -DUSE_OPENMP=ON -DBUILD_SPARSE=OFF -DUSE_EPOLL=OFF .. && \
    export CC=gcc && export CXX=g++ && make -j && \
    cd ../python && python setup.py install

FROM dev-base as mp
ARG GCC_MAJOR_VERSION
ARG BENCHMARK
ARG IMPL
COPY --from=dgl /opt/conda /opt/conda
WORKDIR /workspace
COPY . /workspace
RUN source /opt/rh/gcc-toolset-${GCC_MAJOR_VERSION}/enable && \
    export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"} && \
    git clone --recurse-submodules https://github.com/mlcommons/inference.git mlperf_inference  && \
    cd mlperf_inference  && \
    git submodule update --init --recursive && cd loadgen && \
    CFLAGS="-std=c++14" python setup.py install && \
    cp ../graph/R-GAT/tools/split_seeds.py /workspace/ && \
    cp ../graph/R-GAT/tools/accuracy_igbh.py /workspace/ && \
    cd /workspace && \
    cd utils && python -m pip install .
    
RUN for FILE in $(cat /workspace/redactions.txt); do rm -rf /workspace/${FILE}; rm -rf /workspace/workload_code/${FILE}; done
ENV LD_PRELOAD "/opt/conda/lib/libiomp5.so"
ENV CONDA_PREFIX "/opt/conda"
