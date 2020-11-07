#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""4. Transfer Learning with Your Own Image Dataset
=======================================================

Dataset size is a big factor in the performance of deep learning models.
``ImageNet`` has over one million labeled images, but
we often don't have so much labeled data in other domains.
Training a deep learning models on small datasets may lead to severe overfitting.

Transfer learning is a technique that addresses this problem.
The idea is simple: we can start training with a pre-trained model,
instead of starting from scratch.
As Isaac Newton said, "If I have seen further it is by standing on the
shoulders of Giants".

In this tutorial, we will explain the basics of transfer
learning, and apply it to the ``MINC-2500`` dataset.

Data Preparation
----------------

`MINC <http://opensurfaces.cs.cornell.edu/publications/minc/>`__ is
short for Materials in Context Database, provided by Cornell.
``MINC-2500`` is a resized subset of ``MINC`` with 23 classes, and 2500
images in each class. It is well labeled and has a moderate size thus is
perfect to be our example.

|image-minc|

To start, we first download ``MINC-2500`` from
`here <http://opensurfaces.cs.cornell.edu/publications/minc/>`__.
Suppose we have the data downloaded to ``~/data/`` and
extracted to ``~/data/minc-2500``.

After extraction, it occupies around 2.6GB disk space with the following
structure:

::

    minc-2500
    ├── README.txt
    ├── categories.txt
    ├── images
    └── labels

The ``images`` folder has 23 sub-folders for 23 classes, and ``labels``
folder contains five different splits for training, validation, and test.

We have written a script to prepare the data for you:

:download:`Download prepare_minc.py<../../../scripts/classification/finetune/prepare_minc.py>`

Run it with

::

    python prepare_minc.py --data ~/data/minc-2500 --split 1

Now we have the following structure:

::

    minc-2500
    ├── categories.txt
    ├── images
    ├── labels
    ├── README.txt
    ├── test
    ├── train
    └── val

In order to go through this tutorial within a reasonable amount of time,
we have prepared a small subset of the ``MINC-2500`` dataset,
but you should substitute it with the original dataset for your experiments.
We can download and extract it with:
"""

import zipfile, os
from gluoncv.utils import download

file_url = 'https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/classification/minc-2500-tiny.zip'
zip_file = download(file_url, path='./')
with zipfile.ZipFile(zip_file, 'r') as zin:
    zin.extractall(os.path.expanduser('./'))

################################################################################
# Hyperparameters
# ----------
#
# First, let's import all other necessary libraries.

import mxnet as mx
import numpy as np
import os, time, shutil

from mxnet import gluon, image, init, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from gluoncv.utils import makedirs
from gluoncv.model_zoo import get_model

################################################################################
# We set the hyperparameters as following:

classes = 23

epochs = 5
lr = 0.001
per_device_batch_size = 1
momentum = 0.9
wd = 0.0001

lr_factor = 0.75
lr_steps = [10, 20, 30, np.inf]

num_gpus = 1
num_workers = 8
ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
batch_size = per_device_batch_size * max(num_gpus, 1)

################################################################################
# Things to keep in mind:
#
# 1. ``epochs = 5`` is just for this tutorial with the tiny dataset. please change it to a larger number in your experiments, for instance 40.
# 2. ``per_device_batch_size`` is also set to a small number. In your experiments you can try larger number like 64.
# 3. remember to tune ``num_gpus`` and ``num_workers`` according to your machine.
# 4. A pre-trained model is already in a pretty good status. So we can start with a small ``lr``.
#
# Data Augmentation
# -----------------
#
# In transfer learning, data augmentation can also help.
# We use the following augmentation in training:
#
# 2. Randomly crop the image and resize it to 224x224
# 3. Randomly flip the image horizontally
# 4. Randomly jitter color and add noise
# 5. Transpose the data from height*width*num_channels to num_channels*height*width, and map values from [0, 255] to [0, 1]
# 6. Normalize with the mean and standard deviation from the ImageNet dataset.
#
jitter_param = 0.4
lighting_param = 0.1

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomFlipLeftRight(),
    transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param,
                                 saturation=jitter_param),
    transforms.RandomLighting(lighting_param),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

################################################################################
# With the data augmentation functions, we can define our data loaders:

path = './minc-2500-tiny'
train_path = os.path.join(path, 'train')
val_path = os.path.join(path, 'val')
test_path = os.path.join(path, 'test')

train_data = gluon.data.DataLoader(
    gluon.data.vision.ImageFolderDataset(train_path).transform_first(transform_train),
    batch_size=batch_size, shuffle=True, num_workers=num_workers)

val_data = gluon.data.DataLoader(
    gluon.data.vision.ImageFolderDataset(val_path).transform_first(transform_test),
    batch_size=batch_size, shuffle=False, num_workers = num_workers)

test_data = gluon.data.DataLoader(
    gluon.data.vision.ImageFolderDataset(test_path).transform_first(transform_test),
    batch_size=batch_size, shuffle=False, num_workers = num_workers)

################################################################################
#
# Note that only ``train_data`` uses ``transform_train``, while
# ``val_data`` and ``test_data`` use ``transform_test`` to produce deterministic
# results for evaluation.
#
# Model and Trainer
# -----------------
#
# We use a pre-trained ``ResNet50_v2`` model, which has balanced accuracy and
# computation cost.

model_name = 'ResNet50_v2'
finetune_net = get_model(model_name, pretrained=True)
with finetune_net.name_scope():
    finetune_net.output = nn.Dense(classes)
finetune_net.output.initialize(init.Xavier(), ctx = ctx)
finetune_net.collect_params().reset_ctx(ctx)
finetune_net.hybridize()

trainer = gluon.Trainer(finetune_net.collect_params(), 'sgd', {
                        'learning_rate': lr, 'momentum': momentum, 'wd': wd})
metric = mx.metric.Accuracy()
L = gluon.loss.SoftmaxCrossEntropyLoss()

################################################################################
# Here's an illustration of the pre-trained model
# and our newly defined model:
#
# |image-model|
#
# Specifically, we define the new model by::
#
# 1. load the pre-trained model
# 2. re-define the output layer for the new task
# 3. train the network
#
# This is called "fine-tuning", i.e. we have a model trained on another task,
# and we would like to tune it for the dataset we have in hand.
#
# We define a evaluation function for validation and testing.

def test(net, val_data, ctx):
    metric = mx.metric.Accuracy()
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        outputs = [net(X) for X in data]
        metric.update(label, outputs)

    return metric.get()

################################################################################
# Training Loop
# -------------
#
# Following is the main training loop. It is the same as the loop in
# `CIFAR10 <dive_deep_cifar10.html>`__
# and ImageNet.
#
# .. note::
#
#     Once again, in order to go through the tutorial faster, we are training on a small
#     subset of the original ``MINC-2500`` dataset, and for only 5 epochs. By training on the
#     full dataset with 40 epochs, it is expected to get accuracy around 80% on test data.

lr_counter = 0
num_batch = len(train_data)

for epoch in range(epochs):
    if epoch == lr_steps[lr_counter]:
        trainer.set_learning_rate(trainer.learning_rate*lr_factor)
        lr_counter += 1

    tic = time.time()
    train_loss = 0
    metric.reset()

    for i, batch in enumerate(train_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        with ag.record():
            outputs = [finetune_net(X) for X in data]
            loss = [L(yhat, y) for yhat, y in zip(outputs, label)]
        for l in loss:
            l.backward()

        trainer.step(batch_size)
        train_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)

        metric.update(label, outputs)

    _, train_acc = metric.get()
    train_loss /= num_batch

    _, val_acc = test(finetune_net, val_data, ctx)

    print('[Epoch %d] Train-acc: %.3f, loss: %.3f | Val-acc: %.3f | time: %.1f' %
             (epoch, train_acc, train_loss, val_acc, time.time() - tic))

_, test_acc = test(finetune_net, test_data, ctx)
print('[Finished] Test-acc: %.3f' % (test_acc))

################################################################################
#
# Next
# ----
#
# Now that you have learned to muster the power of transfer
# learning, to learn more about training a model on
# ImageNet, please read `this tutorial <dive_deep_imagenet.html>`__.
#
# The idea of transfer learning is the basis of
# `object detection <../examples_detection/index.html>`_ and
# `semantic segmentation <../examples_segmentation/index.html>`_,
# the next two chapters of our tutorial.
#
# .. |image-minc| image:: https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/datasets/MINC-2500.png
# .. |image-model| image:: https://zh.gluon.ai/_images/fine-tuning.svg
