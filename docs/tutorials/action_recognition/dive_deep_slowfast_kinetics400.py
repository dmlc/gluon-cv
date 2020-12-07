"""6. Dive Deep into Training SlowFast mdoels on Kinetcis400
============================================================

This is a video action recognition tutorial using Gluon CV toolkit, a step-by-step example.
The readers should have basic knowledge of deep learning and should be familiar with Gluon API.
New users may first go through `A 60-minute Gluon Crash Course <http://gluon-crash-course.mxnet.io/>`_.
You can `Start Training Now`_ or `Dive into Deep`_.

Start Training Now
~~~~~~~~~~~~~~~~~~

.. note::

    Feel free to skip the tutorial because the training script is self-complete and ready to launch.

    :download:`Download Full Python Script: train_recognizer.py<../../../scripts/action-recognition/train_recognizer.py>`

    For more training command options, please run ``python train_recognizer.py -h``
    Please checkout the `model_zoo <../model_zoo/index.html#action_recognition>`_ for training commands of reproducing the pretrained model.


Network Structure
-----------------

First, let's import the necessary libraries into python.

"""
from __future__ import division

import argparse, time, logging, os, sys, math

import numpy as np
import mxnet as mx
import gluoncv as gcv
from mxnet import gluon, nd, init, context
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

from gluoncv.data.transforms import video
from gluoncv.data import Kinetics400
from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, LRSequential, LRScheduler, split_and_load, TrainingHistory


################################################################
#
# Here we pick a widely adopted model, ``SlowFast``, for the tutorial.
# `SlowFast <https://arxiv.org/abs/1812.03982>`_ is a new 3D video
# classification model, aiming for best trade-off between accuracy and efficiency.
# It proposes two branches, fast branch and slow branch, to handle different aspects in a video.
# Fast branch is to capture motion dynamics by using many but small video frames.
# Slow branch is to capture fine apperance details by using few but large video frames.
# Features from two branches are combined using lateral connections.

# number of GPUs to use
num_gpus = 1
ctx = [mx.gpu(i) for i in range(num_gpus)]

# Get the model slowfast_4x16_resnet50_kinetics400 with 400 output classes, without pre-trained weights
net = get_model(name='slowfast_4x16_resnet50_kinetics400', nclass=400)
net.collect_params().reset_ctx(ctx)
print(net)

################################################################
# Data Augmentation and Data Loader
# ---------------------------------
#
# Data augmentation for video is different from image. For example, if you
# want to randomly crop a video sequence, you need to make sure all the video
# frames in this sequence undergo the same cropping process. We provide a
# new set of transformation functions, working with multiple images.
# Please checkout the `video.py <../../../gluoncv/data/transforms/video.py>`_ for more details.
# Most video data augmentation strategies used here are introduced in [Wang15]_.

transform_train = transforms.Compose([
    # Fix the input video frames size as 256×340 and randomly sample the cropping width and height from
    # {256,224,192,168}. After that, resize the cropped regions to 224 × 224.
    video.VideoMultiScaleCrop(size=(224, 224), scale_ratios=[1.0, 0.875, 0.75, 0.66]),
    # Randomly flip the video frames horizontally
    video.VideoRandomHorizontalFlip(),
    # Transpose the video frames from height*width*num_channels to num_channels*height*width
    # and map values from [0, 255] to [0,1]
    video.VideoToTensor(),
    # Normalize the video frames with mean and standard deviation calculated across all images
    video.VideoNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

##################################################################
# With the transform functions, we can define data loaders for our
# training datasets.

# Batch Size for Each GPU
per_device_batch_size = 5
# Number of data loader workers
num_workers = 0
# Calculate effective total batch size
batch_size = per_device_batch_size * num_gpus

# Set train=True for training the model.
# ``new_length`` indicates the number of frames we will cover.
# For SlowFast network, we evenly sample 32 frames for the fast branch and 4 frames for the slow branch.
# This leads to the actual input length of 36 video frames.
train_dataset = Kinetics400(train=True, new_length=64, slowfast=True, transform=transform_train)
print('Load %d training samples.' % len(train_dataset))
train_data = gluon.data.DataLoader(train_dataset, batch_size=batch_size,
                                   shuffle=True, num_workers=num_workers)

################################################################
# Optimizer, Loss and Metric
# --------------------------

lr_decay = 0.1
warmup_epoch = 34
total_epoch = 196
num_batches = len(train_data)
lr_scheduler = LRSequential([
    LRScheduler('linear', base_lr=0.01, target_lr=0.1,
                nepochs=warmup_epoch, iters_per_epoch=num_batches),
    LRScheduler('cosine', base_lr=0.1, target_lr=0,
                nepochs=total_epoch - warmup_epoch,
                iters_per_epoch=num_batches,
                step_factor=lr_decay, power=2)
])

# Stochastic gradient descent
optimizer = 'sgd'
# Set parameters
optimizer_params = {'learning_rate': 0.01, 'wd': 0.0001, 'momentum': 0.9}
optimizer_params['lr_scheduler'] = lr_scheduler

# Define our trainer for net
trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)

################################################################
# In order to optimize our model, we need a loss function.
# For classification tasks, we usually use softmax cross entropy as the
# loss function.

loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()

################################################################
# For simplicity, we use accuracy as the metric to monitor our training
# process. Besides, we record metric values, and will print them at the
# end of training.

train_metric = mx.metric.Accuracy()
train_history = TrainingHistory(['training-acc'])

################################################################
# Training
# --------
#
# After all the preparations, we can finally start training!
# Following is the script.
#
# .. note::
#   In order to finish the tutorial quickly, we only train for 0 epoch on a tiny subset of Kinetics400,
#   and 100 iterations per epoch. In your experiments, we recommend setting ``epochs=100`` for the full Kinetics400 dataset.

epochs = 0

for epoch in range(epochs):
    tic = time.time()
    train_metric.reset()
    train_loss = 0

    # Loop through each batch of training data
    for i, batch in enumerate(train_data):
        # Extract data and label
        data = split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = split_and_load(batch[1], ctx_list=ctx, batch_axis=0)

        # AutoGrad
        with ag.record():
            output = []
            for _, X in enumerate(data):
                X = X.reshape((-1,) + X.shape[2:])
                pred = net(X)
                output.append(pred)
            loss = [loss_fn(yhat, y) for yhat, y in zip(output, label)]

        # Backpropagation
        for l in loss:
            l.backward()

        # Optimize
        trainer.step(batch_size)

        # Update metrics
        train_loss += sum([l.mean().asscalar() for l in loss])
        train_metric.update(label, output)

        if i == 100:
            break

    name, acc = train_metric.get()

    # Update history and print metrics
    train_history.update([acc])
    print('[Epoch %d] train=%f loss=%f time: %f' %
        (epoch, acc, train_loss / (i+1), time.time()-tic))

# We can plot the metric scores with:
train_history.plot()

##############################################################################
# Due to the tiny subset, the accuracy number is quite low.
# You can `Start Training Now`_ on the full Kinetics400 dataset.
#
# References
# ----------
#
# .. [Wang15] Limin Wang, Yuanjun Xiong, Zhe Wang, and Yu Qiao. \
#     "Towards Good Practices for Very Deep Two-Stream ConvNets." \
#     arXiv preprint arXiv:1507.02159 (2015).
