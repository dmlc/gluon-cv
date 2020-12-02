"""5. DistributedDataParallel (DDP) Framework
=======================================================

Training deep neural networks on videos is very time consuming.
For example, training a state-of-the-art SlowFast network on Kinetics400 dataset (with 240K 10-seconds short videos)
using a server with 8 V100 GPUs takes more than 10 days.
Slow training causes long research cycles and is not friendly for new comers and students to work on video related problems.
Using distributed training is a natural choice.
Spreading the huge computation over multiple machines can speed up training a lot.
However, only a few open sourced Github repositories on video understanding support distributed training,
and they often lack documentation for this feature.
Besides, there is not much information/tutorial online on how to perform distributed training for deep video models.

Hence, we provide a simple tutorial here to demonstrate how to use our DistributedDataParallel (DDP) framework to perform
efficient distributed training. Note that, even in a single instance with multiple GPUs,
DDP should be used and is much more efficient that vanilla dataparallel.


"""

########################################################################
# Distributed training
# --------------------
#
# There are two ways in which we can distribute the workload of training a neural network across multiple devices,
# data parallelism and model parallelism. Data parallelism refers to the case where each device stores a complete copy of the model.
# Each device works with a different part of the dataset, and the devices collectively update a shared model.
# When models are so large that they don't fit into device memory, then model parallelism is useful.
# Here, different devices are assigned the task of learning different parts of the model.
# In this tutorial, we describe how to train a model with devices distributed across machines in a data parallel way.
# To be specific, we adopt DistributedDataParallel (DDP), which implements data parallelism at the module level that can be applied
# across multiple machines. DDP spawn multiple processes and create a single GPU instance per process.
# It can spread the computation more evenly and particular be useful for deep video model training.
#
# In order to keep this tutorial concise, I wouldn't go into details of what is DDP.
# Readers can refer to Pytorch `Official Tutorials <https://pytorch.org/tutorials/intermediate/ddp_tutorial.html>`_ for more information.


########################################################################
# How to use our DDP framework?
# ----------------------------------------------------------------------
#
# In order to perform distributed training, you need to (1) prepare the cluster; (2) prepare environment; and (3) prepare your code
# and data.

################################################################
# We need a cluster that each node can communicate with each other.
# The first step is to generate ssh keys for each machine.
# For better illustration, let's assume we have 2 machines, node1 and node2.
#
# First, ssh into node1 and type
# ::
#
#     ssh-keygen -t rsa
#
# Just follow the default, you will have a file named ``id_rsa`` and a file named ``id_rsa.pub``, both under the ``~/.ssh/`` folder.
# ``id_rsa`` is the private RSA key, and ``id_rsa.pub`` is its public key.

################################################################
# Second, copy both files (``id_rsa`` and ``id_rsa.pub``) of node1 to all other machines.
# For each machine, you will find an ``authorized_keys`` file under ``~/.ssh/`` folder as well.
# Append ``authorized_keys`` with the content of ``id_rsa.pub``
# This step will make sure all the machines in the cluster is able to communicate with each other.

################################################################
# Before moving on to next step, it is better to perform some sanity checks to make sure the communication is good.
# For example, if you can successfully ssh into other machines, it means they can communicate with each other now. You are good to go.
# If there is any error during ssh, you can use option ``-vvv`` to get verbose information for debugging.

################################################################
# Once you get the cluster ready, it is time to prepare the enviroment.
# Please check `GluonCV installation guide <https://gluon-cv.mxnet.io/install/install-more.html>`_ for more information.
# Every machine should have the same enviroment, such as CUDA, PyTorch and GluonCV, so that the code is runnable.

################################################################
# Now it is time to prepare your code and data in each node.
# In terms of code change, you only need to modify the `DDP_CONFIG` part in the yaml configuration file.
# For example, in our case we have 2 nodes, then change `WORLD_SIZE` to 2.
# `WOLRD_URLS` contains all machines' IP used for training (only support LAN IP for now), you can put their IP addresses in the list.
# If `AUTO_RANK_MATCH` is True, the launcher will automatically assign a world rank number to each machine in `WOLRD_URLS`,
# and consider the first machine as the root. Please make sure to use root's IP for `DIST_URL`.
# If `AUTO_RANK_MATCH` is False, you need to manually set a ranking number to each instance.
# The instance assigned with `rank=0` will be considered as the root machine.
# We suggest always enable `AUTO_RANK_MATCH`.
# An example configuration look like below,
# ::
#
#     DDP_CONFIG:
#       AUTO_RANK_MATCH: True
#       WORLD_SIZE: 2 # Total Number of machines
#       WORLD_RANK: 0 # Rank of this machine
#       DIST_URL: 'tcp://172.31.72.195:23456'
#       WOLRD_URLS: ['172.31.72.195', '172.31.72.196']
#
#       GPU_WORLD_SIZE: 8 # Total Number of GPUs, will be assigned automatically
#       GPU_WORLD_RANK: 0 # Rank of GPUs, will be assigned automatically
#       DIST_BACKEND: 'nccl'
#       GPU: 0 # Rank of GPUs in the machine, will be assigned automatically
#       DISTRIBUTED: True

################################################################
# Once it is done, you can kickstart the training on each machine.
# Simply run `train_ddp_pytorch.py/test_ddp_pytorch.py` with the desire configuration file on each instance, e.g.,
# ::
#
#     python train_ddp_pytorch.py --config-file XXX.yaml
#
# If you are using multiple instances for training, we suggest you start running on the root instance firstly
# and then start the code on other instances.
# The log will only be shown on the root instance by default.

################################################################
# In the end, we want to point out that we have integrated dataloader and training/testing loop in our DDP framework.
# If you simply want to try out our model zoo on your dataset/usecase, please see previous tutorial on how to finetune.
# If you have your new video model, you can add it to the model zoo (e.g., a single .py file) and enjoy the speed up brought by our DDP framework.
# You don't need to handle the multiprocess dataloading and the underlying distributed training setup.
