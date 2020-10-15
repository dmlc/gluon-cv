FROM ubuntu:18.04

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
      libxft-dev &&\
  rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip
RUN pip3 install  --no-cache --upgrade \
    wheel \
    cmake \
    awscli \
    pypandoc
RUN git clone https://github.com/dmlc/gluon-cv
WORKDIR gluon-cv
ADD gluon_cv_job.sh .
RUN chmod +x gluon_cv_job.sh
