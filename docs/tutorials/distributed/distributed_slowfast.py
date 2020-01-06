"""1. Distributed training of deep video models
================================================

Training deep neural networks on videos is very time consuming. For example, training a state-of-the-art SlowFast network [Feichtenhofer18]_
on Kinetics400 dataset using a server with 8 V100 GPUs takes more than 10 days. Slow training causes long research cycles
and is not friendly for new comers and students to work on video related problems.

Using distributed training is a natural choice. Spreading the huge computation over multiple machines can speed up training
a lot. However, only a few open sourced Github repositories on video understanding support distributed training,
and they often lack documentation for this feature.
Besides, there is not much information/tutorial online on how to perform distributed training for deep video models.

Hence, we provide a simple tutorial here to demonstrate how to use our code to perform distributed training of SlowFast models
on Kinetics400 dataset.

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
#
# There are some key concepts in distributed training, such as server, worker, scheduler, kvstore, etc.
# Server is a node to store model's parameters and communicate with workers.
# Worker is a node actually performing training on a batch of training samples.
# Before processing each batch, the workers pull weights from servers.
# The workers also send gradients to the servers after each batch.
# Scheduler is to set up the cluster for each node to communicate, and there is only one scheduler in the entire cluster
# Kvstore, which is key-value store, is a critical component used for multi-device training.
# It stores model parameters and optimizers, to receive gradient and update model.
# In order to keep this tutorial concise, I wouldn't go into details.
# Readers can refer to MXNet `official documentation <https://mxnet.apache.org/api/faq/distributed_training>`_ for more information.


########################################################################
# How to use our code to train a SlowFast model in a distributed manner?
# ----------------------------------------------------------------------
#
# In order to perform distributed training, you need to install MXNet and prepare the cluster ready.
# MXNet provides a script ``tools/launch.py`` to make it easy to launch distributed training on a cluster with ssh, mpi, sge or yarn.

################################################################
# First, let's install MXNet.
# ::
#
#     pip install mxnet-cu100
#
# For more installation options (i.e., different versions of MXNet or CUDA),
# please check `GluonCV installation guide <https://gluon-cv.mxnet.io/install/install-more.html>`_ for more information.


################################################################
# We also need the script to launch the job, let's clone the repo as well.
# ::
#
#     git clone https://github.com/apache/incubator-mxnet.git
#
# The script we need is under folder ``tools``, named ``launch.py``.
# Note that, we only need to put MXNet on the server node, not on the worker nodes.

################################################################
# Now it is time to prepare the cluster. We need a cluster that the worker nodes can communicate with the server node.
# The first step is to generate ssh keys for each machine.
# For better illustration, let's assume we have four machines, node1, node2, node3 and node4.
# We use node1 as server, and node1, node2, node3 and node4 as workers.
# Note that, a node can either be a server or worker. We can also have multiple servers.
#
# First, ssh into each node and type
# ::
#
#     ssh-keygen -t rsa
#
# Just follow the default, you will have a file named ``id_rsa.pub`` under ``~/.ssh/`` folder.
# The file's content usually looks like
# ::
#
#     ssh-rsa XXXXXXXXXXXXXXXX node1@ip-123-123-1-123
#
# The content in the middle is the actuall ssh key, and the IP address in the end is the node's internal IP address.

################################################################
# Second, copy the content in ``id_rsa.pub`` of the server and paste it into each workers' ``authorized_keys`` file.
# The ``authorized_keys`` file is under ``~/.ssh/`` folder as well. This step will make all the workers accessible to the server.
# Similarly, copy the content in ``id_rsa.pub`` of each worker and paste it into the server's ``authorized_keys`` file.
# This step will make the server accessible to all the workers.
# Note that, there is no need for the workers to connect with each other because they don't communicate.
# Before kickstarting the actual distributed training, it is better to perform some sanity checks to make sure the communication is good.
# For example, if you are inside worker node2 and want to test the connection to server node1,
#
# ::
#
#     ssh node1@123.123.1.123
#
# If you can ssh into the server node1, it means they can communicate with each other now. You are good to go.


################################################################
# Once you get the cluster and MXNet script ready, the next thing is to prepare your code and data in each node.
# The code needs to be in the same directoty on every machine so that a single command can work on multiple machines.
# Let's clone the GluonCV repo and install it,
# ::
#
#     git clone https://github.com/dmlc/gluon-cv.git
#     cd gluon-cv
#     pip install -e .
#
# Similarly, the data needs to be in the same path on every machine as well so that the dataloader knows where to find the data.
#


################################################################
# Ok, now it is time to type the command and start the training.
#
# ::
#
#     ../incubator-mxnet/tools/launch.py -n 4 -H host.txt --launcher ssh \
#     python ./scripts/action-recognition/train_recognizer.py --dataset kinetics400 \
#     --data-dir ~/.mxnet/kinetics400/train --val-data-dir ~/.mxnet/kinetics400/val \
#     --train-list ~/.mxnet/kinetics400/train.txt --val-list ~/.mxnet/kinetics400/val.txt \
#     --dtype float32 --mode hybrid --prefetch-ratio 1.0 --kvstore dist_sync_device \
#     --model slowfast_4x16_resnet50_kinetics400 --slowfast --slow-temporal-stride 16 --fast-temporal-stride 2 \
#     --video-loader --use-decord --num-classes 400 --batch-size 8 --num-gpus 8 --num-data-workers 32 \
#     --input-size 224 --new-height 256 --new-width 340 --new-length 64 --new-step 1 \
#     --lr-mode cosine --lr 0.4 --momentum 0.9 --wd 0.0001 --num-epochs 196 --warmup-epochs 34 --warmup-lr 0.01 \
#     --scale-ratios 1.0,0.8 --save-frequency 10 --log-interval 50 --logging-file slowfast_4x16.log --save-dir ./checkpoints
#
# Here, the ``host.txt`` file contains the IP addresses of all machines, e.g.,
# ::
#     123.123.1.123
#     123.123.2.123
#     123.123.3.123
#     123.123.4.123
#
# ``--kvstore dist_sync_device`` is when there are multiple GPUs being used on each node, this mode aggregates gradients and updates weights on GPU.
# There are other modes, like dist_sync, dist_async etc.
# To find out more details, check out `this tutorial <https://mxnet.apache.org/api/faq/distributed_training>`_.
#
# Another thing to notice is the learning rate. For single node training, we set lr to 0.1. However, for multi-node training, we increase it to 0.4
# because we have four machines. It's a good practice, called linear scaling rule, introduced in [Goyal17]_.
# When the minibatch size is multiplied by k, multiply the learning rate by k. All other hyper-parameters (weight decay, etc.) are kept unchanged.
# The linear scaling rule can help us to not only match the accuracy between using small and large minibatches, but equally importantly, to largely
# match their training curves, which enables rapid debugging and comparison of experiments prior to convergence.
#
# If everything is setup well, you will see the model is training now. All printed information will be captured and sent to the worker running launch.py
# (which is the server node). Checkpoints will be saved locally on each machine.

################################################################
# Speed
# -----
#
# Usually, the training will be faster when you use more machines, but not linear upscaling due to communication cost.
# The actual speed up depends on the network bandwidth, server CPU capibility, dataloader efficiency, etc.
# For example, if you use our code on four P3.16xlarge machines on AWS in the same placement group, you will get 3x speed boost.
# Similar speed up ratio (0.75) can be observed when you use 8 machines or more.
# In our case, we use eight P3.16xlarge machines to train a ``slowfast_4x16_resnet50_kinetics400`` model. The training can be completed in 1.5 days.


################################################################
# References
# ----------
#
# .. [Goyal17] Priya Goyal, Piotr Dollár, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, Kaiming He. \
#     "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour." \
#     arXiv preprint arXiv:1706.02677 (2017).
#
# .. [Feichtenhofer18] Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, Kaiming He. \
#     "SlowFast Networks for Video Recognition." \
#     arXiv preprint arXiv:1812.03982 (2018).
