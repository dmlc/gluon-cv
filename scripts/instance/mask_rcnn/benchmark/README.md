# Benchmarking instruction to run Mask R-CNN

## Operating System
We recommend to use the latest AWS Deep Learning AMI as it provides all the infrastructure and tools to 
accelerate deep learning in the cloud, at any scale. 
Please find more information about DLAMI [here](https://docs.aws.amazon.com/dlami/latest/devguide/options.html)


## Installation

#### 1. MXNet with CUDA-10.0
```bash
pip install https://repo.mxnet.io/dist/python/cu100mkl/mxnet_cu100mkl-1.6.0b20191230-py2.py3-none-manylinux1_x86_64.whl
```

#### 2. Horovod
```bash
pip uninstall -y horovod
HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITH_MXNET=1 HOROVOD_WITHOUT_TENSORFLOW=1 HOROVOD_WITHOUT_PYTORCH=1 pip install --no-cache-dir horovod==0.19.0
```

#### 3. NVIDIA cocoapi
```bash
pip uninstall -y pycocotools
pip install pybind11==2.4.3
git clone --recursive https://github.com/NVIDIA/cocoapi.git
cd $HOME/cocoapi/PythonAPI/
python setup.py build_ext install
cd $HOME
```

#### 4. GluonCV
```bash
pip uninstall -y gluoncv
pip install Cython==0.29.6
pip install mpi4py==3.0.3
git clone --recursive https://github.com/dmlc/gluon-cv.git $HOME/gluoncv
cd $HOME/gluoncv
rm -rf build/ dist/
python setup.py install --with-cython
cd $HOME
```

#### 5. numactl
This package only requires if using `ompi_bind_DGX1.sh`
```bash
# Ubuntu
sudo apt-get update -y
sudo apt-get install -y numactl

# RHEL
sudo yum check-update
sudo yum install numactl
```


## Download MSCOCO-2017 dataset
Download the dataset on all nodes locally using below command:
```bash
cd $HOME/gluoncv/scripts/datasets
python mscoco.py
```

## Library path
If you are training the model using AWS P3dn.24xlarge, make sure your `LD_LIBRARY_PATH` have this below library path.
Since the AWS P3dn.24xlarge supports [EFA](https://aws.amazon.com/hpc/efa/), we just want to make sure `EFA` and `libfabric` paths are included in `LD_LIBRARY_PATH`
.
```bash
/opt/amazon/efa/lib64
$HOME/aws-ofi-nccl/install/lib
```

## Export Flags
The below flags value are selected based on extensive experiments and it offers best training throughput 
on AWS EC2 [P3](https://aws.amazon.com/ec2/instance-types/p3/) Instances. Please export this flags on master or root node
```bash
export HOROVOD_FUSION_THRESHOLD=134217728
export HOROVOD_NUM_STREAMS=2
export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD=999
export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD=25
export OMP_NUM_THREADS=2
export NCCL_DEBUG=VERSION

export HOROVOD_CYCLE_TIME=0.1
export HOROVOD_HIERARCHICAL_ALLREDUCE=0
export HOROVOD_CACHE_CAPACITY=0

export NCCL_MIN_NRINGS=1
export NCCL_TREE_THRESHOLD=4294967296
export NCCL_NSOCKS_PERTHREAD=8
export NCCL_SOCKET_NTHREADS=2
export NCCL_BUFFSIZE=16777216
export HOROVOD_NUM_NCCL_STREAMS=2

export NCCL_NET_GDR_READ=1 
export HOROVOD_TWO_STAGE_LOOP=1
export HOROVOD_ALLREDUCE_MODE=1
export HOROVOD_FIXED_PAYLOAD=161
export HOROVOD_MPI_THREADS_DISABLE=1
export MXNET_USE_FUSION=0
```

## Command to run

#### 1. Single Node: The below command is an example on how to run the horovod training on single AWS P3dn.24xlarge cluster node
```bash
mpirun --allow-run-as-root \
-np 8 -H localhost:8 \
-bind-to none -map-by slot -mca pml ob1 -mca btl ^openib \
-mca btl_tcp_if_include ens5 -x NCCL_SOCKET_IFNAME=ens5 \
-x FI_PROVIDER="efa" -x FI_EFA_TX_MIN_CREDITS=64 \
-x SSH_AGENT_PID -x LESS_TERMCAP_mb -x HOSTNAME -x LESS_TERMCAP_md \
-x LESS_TERMCAP_me -x TERM -x SHELL -x HISTSIZE -x EC2_AMITOOL_HOME \
-x SSH_CLIENT -x CONDA_SHLVL -x HOROVOD_CYCLE_TIME \
-x CONDA_PROMPT_MODIFIER \
-x PYTHON_INSTALL_LAYOUT -x MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD \
-x LESS_TERMCAP_ue -x SSH_TTY -x USER -x LD_LIBRARY_PATH -x LS_COLORS \
-x CONDA_EXE -x EC2_HOME -x SSH_AUTH_SOCK -x _CE_CONDA \
-x LESS_TERMCAP_us -x MAIL -x PATH \
-x CONDA_PREFIX -x PWD -x JAVA_HOME -x AWS_CLOUDWATCH_HOME -x LANG \
-x MODULEPATH -x LOADEDMODULES \
-x HOROVOD_NUM_STREAMS -x _CE_M -x HISTCONTROL -x SHLVL -x HOME \
-x AWS_PATH -x HOROVOD_FUSION_THRESHOLD -x AWS_AUTO_SCALING_HOME \
-x CONDA_PYTHON_EXE -x LOGNAME -x CVS_RSH -x AWS_ELB_HOME \
-x SSH_CONNECTION -x MODULESHOME -x OMP_NUM_THREADS -x CONDA_DEFAULT_ENV \
-x LESSOPEN -x MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD \
-x LESS_TERMCAP_se -x MXNET_USE_FUSION -x _ \
--tag-output $HOME/ompi_bind_DGX1.sh \
python -u $HOME/gluoncv/scripts/instance/mask_rcnn/train_mask_rcnn.py \
-j4 --horovod --amp --lr-decay-epoch 8,10 --epochs 12 --log-interval 100 \
--val-interval 12 --batch-size 16 --use-fpn --lr 0.02 \
--lr-warmup-factor 0.03 --lr-warmup 1000 --static-alloc \
--clip-gradient 1.5 --use-ext
```

#### 2. Multi Node: The below command is an example on how to run the horovod training on 24 P3dn.24xlarge node.
Note: The `hostfile` should looks like list of `<ip address> slots=<num_of_gpus>`
```bash
mpirun --allow-run-as-root \
-np 192 --hostfile $HOME/hosts_24 \
-bind-to none -map-by slot -mca pml ob1 -mca btl ^openib \
-mca btl_tcp_if_include ens5 -x NCCL_SOCKET_IFNAME=ens5 \
-x FI_PROVIDER="efa" -x FI_EFA_TX_MIN_CREDITS=64 \
-x SSH_AGENT_PID -x LESS_TERMCAP_mb -x HOSTNAME -x LESS_TERMCAP_md \
-x LESS_TERMCAP_me -x TERM -x SHELL -x NCCL_MIN_NRINGS -x HISTSIZE \
-x EC2_AMITOOL_HOME -x SSH_CLIENT -x CONDA_SHLVL -x HOROVOD_CYCLE_TIME \
-x CONDA_PROMPT_MODIFIER \
-x PYTHON_INSTALL_LAYOUT -x MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD \
-x LESS_TERMCAP_ue -x SSH_TTY -x USER -x LD_LIBRARY_PATH -x LS_COLORS \
-x CONDA_EXE -x EC2_HOME -x SSH_AUTH_SOCK -x _CE_CONDA -x LESS_TERMCAP_us \
-x MAIL -x PATH -x CONDA_PREFIX -x PWD \
-x JAVA_HOME -x AWS_CLOUDWATCH_HOME -x LANG -x MODULEPATH \
-x LOADEDMODULES -x NCCL_TREE_THRESHOLD \
-x HOROVOD_NUM_STREAMS -x _CE_M -x HISTCONTROL -x SHLVL -x HOME \
-x AWS_PATH -x HOROVOD_FUSION_THRESHOLD -x HOROVOD_HIERARCHICAL_ALLREDUCE \
-x AWS_AUTO_SCALING_HOME -x CONDA_PYTHON_EXE -x LOGNAME -x CVS_RSH \
-x AWS_ELB_HOME -x SSH_CONNECTION -x MODULESHOME -x OMP_NUM_THREADS \
-x CONDA_DEFAULT_ENV -x LESSOPEN -x MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD \
-x NCCL_DEBUG -x LESS_TERMCAP_se -x _ -x HOROVOD_STALL_CHECK_TIME_SECONDS \
-x HOROVOD_STALL_SHUTDOWN_TIME_SECONDS -x HOROVOD_NUM_NCCL_STREAMS \
-x HOROVOD_MLSL_BGT_AFFINITY -x HOROVOD_CACHE_CAPACITY \
-x HOROVOD_NUM_NCCL_STREAMS -x NCCL_NSOCKS_PERTHREAD \
-x NCCL_SOCKET_NTHREADS -x NCCL_BUFFSIZE -x NCCL_NET_GDR_READ \
-x HOROVOD_TWO_STAGE_LOOP -x HOROVOD_ALLREDUCE_MODE \
-x HOROVOD_FIXED_PAYLOAD -x HOROVOD_MPI_THREADS_DISABLE \
-x MXNET_USE_FUSION \
--tag-output $HOME/ompi_bind_DGX1.sh \
python -u $HOME/gluoncv/scripts/instance/mask_rcnn/train_mask_rcnn.py \
--num-workers 8 --horovod --amp --lr-decay-epoch 10,14 --epochs 12 \
--log-interval 100 --val-interval 1 --batch-size 192 --use-fpn \
--lr 0.16 --lr-warmup-factor 0.001 --lr-warmup 1600 \
--static-alloc --clip-gradient 1.5 --use-ext
```

#### Note:
- Please ignore the warning message as stated below if you see that in a log file:
```bash
Warning: could not find environment variable <name of env. variable>
```
- The `ompi_bind_DGX1.sh ` file provided here is specific to AWS [p3dn.24xlarge](https://aws.amazon.com/ec2/instance-types/p3/) Instance.

Please find the sample [logfile](https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/instance/mask_rcnn_fpn_resnet50_v1b_coco_train_horovod_24_p3dn24xlarge.log) that ran on AWS 24 P3dn.24xlarge Instances using Horovod(mpirun command).