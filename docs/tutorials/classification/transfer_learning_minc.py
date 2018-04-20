#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""Transfer Learning with Your Own Image Dataset
===============================================


ImageNet is a huge and comprehensive dataset. However in practice, we
may not have the dataset of the same size. Training a deep learning
model on a small dataset may suffer from underfitting for not enough
data.

Transfer learning is a technique enabling us to train a decent model on
a relatively small training dataset. The idea is simple: we can start
training with our own dataset from a pre-trained model. As Isaac Newton
said, "If I have seen further it is by standing on the shoulders of
Giants".

In this tutorial, we will walk you through the basic idea of transfer
learning, and apply to the ``MINC-2500`` dataset.

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
Suppose we have the data downloaded at ``~/data/`` and
extracted at ``~/data/minc-2500``.

After extraction, it occupies around 2.6GB disk space with the following
structure:

::

    minc-2500
    ├── README.txt
    ├── categories.txt
    ├── images
    └── labels

The ``images`` folder has 23 sub-folders for 23 classes, and ``labels``
folder contains five different splits for train, validation, and test.
We can prepare the dataset according to one of its split.

We have written a script to prepare the data for you:

:download:`Download Python Script prepare_minc.py<../../../scripts/classification/finetune/prepare_minc.py>`

Execute it by

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

In order to compile the tutorial with a reasonable cost, we have prepared a small subset of the
``MINC-2500`` data. We can download and extract it with:
"""

import zipfile, os
from gluonvision.utils import download

file_url = 'https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/classification/minc-2500-tiny.zip'
zip_file = download(file_url, path='./')
with zipfile.ZipFile(zip_file, 'r') as zin:
    zin.extractall(os.path.expanduser('./'))

################################################################################
# Hyperparameters
# ----------
#
# First, let's load all other necessary libraries.

import mxnet as mx
import numpy as np
import os, time, shutil

from mxnet import gluon, image, init, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision as models
from mxnet.gluon.data.vision import transforms
from gluonvision.utils import makedirs

################################################################################
# We set the hyperparameters as follows:

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
# 1. ``epochs = 5`` is just for this tutorial with the tiny dataset. please
# change it to a larger number in a full training, for instance 40.
# 2. ``per_device_batch_size`` is also set to a small number. In a full training
# you can try larger number like 64.
# 3. remember to tune ``num_gpus`` and ``num_workers`` according to your machine.
# 4. A pre-trained model is already in a pretty good status.
# So we can start with a small ``lr``.
#
# Data Augmentation
# -----------------
#
# In transfer learning, data augmentation can also help.
# We use the following augmentation in training:
#
# 1. Resize the short edge of the image to 480px
# 2. Randomly crop the image and resize it to 224x224
# 3. Randomly flip the image horizontally
# 4. Randomly disturb the color and add noise
# 5. Transpose the data from Height*Width*Channel to Channel*Height*Width,
# and map values from [0, 255] to [0, 1]
# 6. Normalize with the mean and standard deviation from the ImageNet dataset.
#
jitter_param = 0.4
lighting_param = 0.1
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

transform_train = transforms.Compose([
    transforms.Resize(480),
    transforms.RandomResizedCrop(224),
    transforms.RandomFlipLeftRight(),
    transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param,
                                 saturation=jitter_param),
    transforms.RandomLighting(lighting_param),
    transforms.ToTensor(),
    normalize
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
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
# Note that only the ``train_data`` calls ``transform_train``.
# ``val_data`` and ``test_data`` call ``transform_test`` to have a deterministic
# result as a performance metric.
#
# Model and Trainer
# -----------------
#
# We use a pre-trained ``ResNet50_v2``, for a balance of performance and
# computational cost.

model_name = 'ResNet50_v2'
finetune_net = gluon.model_zoo.vision.get_model(model_name, pretrained=True)
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
# We can define a performance evaluation function for validation data and test data.

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
# Once again, in order to build the tutorial faster, we are training on a small
# subset of the original ``MINC-2500``, and with only 5 epochs. By training on the
# full dataset with 40 epochs, it is expected to get accuracy around 80% on test data.
#
# Next
# ----
#
# Now that you have learned how powerful a model could be by transfer
# learning. If you would like to know more about how to train a model on
# ImageNet, please read this tutorial.
#
# The idea of transfer learning is the basis of object detection and
# semantic segmentation, the next two chapters in our tutorial.
#
# .. |image-minc| image:: https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/datasets/MINC-2500.png
