FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      locales \
      cmake \
      wget \
      subversion \
      git \
      curl \
      vim \
      unzip \
      sudo \
      ca-certificates \
      libjpeg-dev \
      libpng-dev \
      libfreetype6-dev \
      libopenblas-dev \
      python3-dev \
      python3-pip \
      python3-setuptools \
      pandoc \
      libgl1-mesa-glx \
      libxft-dev \
      gcc \
      libtinfo-dev \
      zlib1g-dev \
      build-essential \
      lsb-release \
      software-properties-common \
      libedit-dev \
      libxml2-dev && \
  rm -rf /var/lib/apt/lists/*

# Install TVM
RUN bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"
RUN git clone --recursive https://github.com/apache/tvm tvm
WORKDIR tvm
# Ping to a commit
RUN git checkout 2f109a7 && git submodule update
RUN mkdir build && cp cmake/config.cmake build
RUN cd build && echo set\(USE_CUDA ON\) >> config.cmake && echo set\(USE_LLVM ON\) >> config.cmake && cmake .. && make -j4
# Python Binding
ENV PYTHONPATH=/tvm/python:/tvm/vta/python:${PYTHONPATH}
RUN pip3 install --upgrade pip
RUN pip3 install --ignore-installed --no-cache --upgrade \
    numpy==1.19.5 \
    decorator==5.0.5 \
    attrs==20.3.0
    
# Prepare gluoncv
WORKDIR /
RUN pip3 install --ignore-installed --no-cache --upgrade \
    wheel==0.36.2 \
    cmake==3.18.4.post1 \
    awscli==1.19.45 \
    pypandoc==1.5 \
    PyYAML==5.4.1 \
    nose==1.3.7 \
    nose-timer==1.0.1 \
    torch==1.8.1 \
    torchvision==0.9.1
RUN git clone https://github.com/dmlc/gluon-cv
WORKDIR gluon-cv
ADD gluon_cv_job.sh .
RUN chmod +x gluon_cv_job.sh
