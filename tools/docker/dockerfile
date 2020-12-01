FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

LABEL maintainer="GluonCV Team"

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib" \
    PYTHONIOENCODING=UTF-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

ENV WORKDIR=/workspace
ENV SHELL=/bin/bash

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    ca-certificates \
    curl \
    emacs \
    subversion \
    locales \
    cmake \
    git \
    libopencv-dev \
    htop \
    vim \
    wget \
    unzip \
    libopenblas-dev \
    ninja-build \
    openssh-client \
    openssh-server \
    python3-dev \
    python3-pip \
    python3-setuptools \
    libxft-dev \
    zlib1g-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip --no-cache-dir install --upgrade \
    pip \
    setuptools

###########################################################################
# Horovod dependencies
###########################################################################

# Install Open MPI
RUN mkdir /tmp/openmpi \
 && cd /tmp/openmpi \
 && curl -fSsL -O https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.1.tar.gz \
 && tar zxf openmpi-4.0.1.tar.gz \
 && cd openmpi-4.0.1 \
 && ./configure --enable-orterun-prefix-by-default \
 && make -j $(nproc) all \
 && make install \
 && ldconfig \
 && rm -rf /tmp/openmpi

# Create a wrapper for OpenMPI to allow running as root by default
RUN mv /usr/local/bin/mpirun /usr/local/bin/mpirun.real \
 && echo '#!/bin/bash' > /usr/local/bin/mpirun \
 && echo 'mpirun.real --allow-run-as-root "$@"' >> /usr/local/bin/mpirun \
 && chmod a+x /usr/local/bin/mpirun

RUN echo "hwloc_base_binding_policy = none" >> /usr/local/etc/openmpi-mca-params.conf \
 && echo "rmaps_base_mapping_policy = slot" >> /usr/local/etc/openmpi-mca-params.conf

ENV LD_LIBRARY_PATH=/usr/local/openmpi/lib:$LD_LIBRARY_PATH
ENV PATH=/usr/local/openmpi/bin/:/usr/local/bin:/root/.local/bin:$PATH

RUN ln -s $(which python3) /usr/local/bin/python

RUN mkdir -p ${WORKDIR}

# install PyYAML==5.1.2 to avoid conflict with latest awscli
# python-dateutil==2.8.0 to satisfy botocore associated with latest awscli
RUN pip3 install --no-cache --upgrade \
    wheel \
    numpy==1.19.1 \
    pandas==0.25.1 \
    pytest \
    Pillow \
    requests==2.22.0 \
    scikit-learn==0.20.4 \
    scipy==1.2.2 \
    urllib3==1.25.8 \
    python-dateutil==2.8.0 \
    sagemaker-experiments==0.* \
    PyYAML==5.3.1 \
    mpi4py==3.0.2 \
    jupyterlab==2.2.4 \
    cmake \
    awscli

# Install MXNet
RUN pip3 install --no-cache --upgrade mxnet-cu102==1.7.0

# Install PyTorch
RUN pip3 install torch==1.6.0 torchvision==0.7.0

RUN mkdir -p ${WORKDIR}/notebook
RUN mkdir -p ${WORKDIR}/data
RUN mkdir -p /.init
RUN cd ${WORKDIR} \
   && git clone https://github.com/dmlc/gluon-cv \
   && cd gluon-cv \
   && git checkout master \
   && python3 -m pip install -U -e ."[extras]" --user

COPY start_jupyter.sh /start_jupyter.sh
COPY devel_entrypoint.sh /devel_entrypoint.sh
RUN chmod +x /devel_entrypoint.sh

EXPOSE 8888
EXPOSE 8787
EXPOSE 8786

WORKDIR ${WORKDIR}

# Debug horovod by default
RUN echo NCCL_DEBUG=INFO >> /etc/nccl.conf

# Install NodeJS + Tensorboard + TensorboardX
RUN curl -sL https://deb.nodesource.com/setup_14.x | bash - \
    && apt-get install -y nodejs

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    libsndfile1-dev

RUN pip3 install --no-cache --upgrade \
    soundfile==0.10.2 \
    ipywidgets==7.5.1 \
    jupyter_tensorboard==0.2.0 \
    widgetsnbextension==3.5.1 \
    tensorboard==2.1.1 \
    tensorboardX==2.1
RUN jupyter labextension install jupyterlab_tensorboard \
   && jupyter nbextension enable --py widgetsnbextension \
   && jupyter labextension install @jupyter-widgets/jupyterlab-manager

# Revise default shell to /bin/bash
RUN jupyter notebook --generate-config \
  && echo "c.NotebookApp.terminado_settings = { 'shell_command': ['/bin/bash'] }" >> /root/.jupyter/jupyter_notebook_config.py

# Add Tini
ARG TINI_VERSION=v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini
ENTRYPOINT [ "/tini", "--", "/devel_entrypoint.sh" ]
CMD ["/bin/bash"]
