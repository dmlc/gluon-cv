"""7. Fine-tuning SOTA video models on your own dataset
=======================================================

This is a video action recognition tutorial using Gluon CV toolkit, a step-by-step example.
The readers should have basic knowledge of deep learning and should be familiar with Gluon API.
New users may first go through `A 60-minute Gluon Crash Course <http://gluon-crash-course.mxnet.io/>`_.
You can `Start Training Now`_ or `Dive into Deep`_.

Fine-tuning is an important way to obtain good video models on your own data when you don't have large annotated dataset or don't have the
computing resources to train a model from scratch for your use case.
In this tutorial, we provide a simple unified solution.
The only thing you need to prepare is a text file containing the information of your videos (e.g., the path to your videos),
we will take care of the rest.
You can start fine-tuning from many popular pre-trained models (e.g., I3D, I3D-nonlocal, SlowFast) using a single command line.

Start Training Now
~~~~~~~~~~~~~~~~~~

.. note::

    Feel free to skip the tutorial because the training script is self-complete and ready to launch.

    :download:`Download Full Python Script: train_recognizer.py<../../../scripts/action-recognition/train_recognizer.py>`

    For more training command options, please run ``python train_recognizer.py -h``
    Please checkout the `model_zoo <../model_zoo/index.html#action_recognition>`_ for training commands of reproducing the pretrained model.


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
from gluoncv.data import VideoClsCustom
from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, LRSequential, LRScheduler, split_and_load, TrainingHistory


######################################################################
# Custom DataLoader
# ------------------
#
# We provide a general dataloader for you to use on your own dataset. Your data can be stored in any hierarchy,
# can be stored in either video format or already decoded to frames. The only thing you need
# to prepare is a text file, ``train.txt``.
#
# If your data is stored in image format (already decoded to frames). Your ``train.txt`` should look like:
#
# ::
#
#     video_001 200 0
#     video_001 200 0
#     video_002 300 0
#     video_003 100 1
#     video_004 400 2
#     ......
#     video_100 200 10
#
# There are three items in each line, separated by spaces.
# The first item is the path to your training videos, e.g., video_001.
# It should be a folder containing the frames of video_001.mp4.
# The second item is the number of frames in each video, e.g., 200.
# The third item is the label of the videos, e.g., 0.
#
# If your data is stored in video format. Your ``train.txt`` should look like:
#
# ::
#
#     video_001.mp4 200 0
#     video_001.mp4 200 0
#     video_002.mp4 300 0
#     video_003.mp4 100 1
#     video_004.mp4 400 2
#     ......
#     video_100.mp4 200 10
#
# Similarly, there are three items in each line, separated by spaces.
# The first item is the path to your training videos, e.g., video_001.mp4.
# The second item is the number of frames in each video. But you can put any number here
# because our video loader will compute the number of frames again automatically during training.
# The third item is the label of that video, e.g., 0.
#
#
# Once you prepare the ``train.txt``, you are good to go.
# Just use our general dataloader `VideoClsCustom <https://github.com/dmlc/gluon-cv/blob/master/gluoncv/data/kinetics400/classification.py>`_ to load your data.
#
# In this tutorial, we will use UCF101 dataset as an example.
# For your own dataset, you can just replace the value of ``root`` and ``setting`` to your data directory and your prepared text file.
# Let's first define some basics.

num_gpus = 1
ctx = [mx.gpu(i) for i in range(num_gpus)]
transform_train = video.VideoGroupTrainTransform(size=(224, 224), scale_ratios=[1.0, 0.8], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
per_device_batch_size = 5
num_workers = 0
batch_size = per_device_batch_size * num_gpus

train_dataset = VideoClsCustom(root=os.path.expanduser('~/.mxnet/datasets/ucf101/rawframes'),
                               setting=os.path.expanduser('~/.mxnet/datasets/ucf101/ucfTrainTestlist/ucf101_train_split_1_rawframes.txt'),
                               train=True,
                               new_length=32,
                               transform=transform_train)
print('Load %d training samples.' % len(train_dataset))
train_data = gluon.data.DataLoader(train_dataset, batch_size=batch_size,
                                   shuffle=True, num_workers=num_workers)


################################################################
# Custom Network
# --------------
#
# You can always define your own network architecture. Here, we want to show how to fine-tune on a pre-trained model.
# Since I3D model is a very popular network, we will use I3D with ResNet50 backbone trained on Kinetics400 dataset (i.e., ``i3d_resnet50_v1_kinetics400``) as an example.
#
# For simple fine-tuning, people usually just replace the last classification (dense) layer to the number of classes in your dataset
# without changing other things. In GluonCV, you can get your customized model with one line of code.
net = get_model(name='i3d_resnet50_v1_custom', nclass=101)
net.collect_params().reset_ctx(ctx)
print(net)

################################################################
# We also provide other customized network architectures for you to use on your own dataset. You can simply change the ``dataset`` part in
# any pretrained model name to ``custom``, e.g., ``slowfast_4x16_resnet50_kinetics400`` to ``slowfast_4x16_resnet50_custom``.
#
# Once you have the dataloader and network for your own dataset, the rest is the same as in previous tutorials.
# Just define the optimizer, loss and metric, and kickstart the training.


################################################################
# Optimizer, Loss and Metric
# --------------------------

# Learning rate decay factor
lr_decay = 0.1
# Epochs where learning rate decays
lr_decay_epoch = [40, 80, 100]

# Stochastic gradient descent
optimizer = 'sgd'
# Set parameters
optimizer_params = {'learning_rate': 0.001, 'wd': 0.0001, 'momentum': 0.9}

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
#   In order to finish the tutorial quickly, we only fine tune for 3 epochs, and 100 iterations per epoch for UCF101.
#   In your experiments, you can set the hyper-parameters depending on your dataset.

epochs = 0
lr_decay_count = 0

for epoch in range(epochs):
    tic = time.time()
    train_metric.reset()
    train_loss = 0

    # Learning rate decay
    if epoch == lr_decay_epoch[lr_decay_count]:
        trainer.set_learning_rate(trainer.learning_rate*lr_decay)
        lr_decay_count += 1

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

######################################################################
# We can see that the training accuracy increase quickly.
# Actually, if you look back tutorial 4 (Dive Deep into Training I3D mdoels on Kinetcis400) and compare the training curve,
# you will see fine-tuning can achieve much better result using much less time.
# Try fine-tuning other SOTA video models on your own dataset and see how it goes.
