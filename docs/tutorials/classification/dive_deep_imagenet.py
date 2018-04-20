"""Train Your Own Model on ImageNet
======================

``ImageNet`` is the most well-known dataset for image classification task.
Since it's been published, there has been plenty of works pushing the performance
of classification accuracy.

Although there are a lot of available models, it is still a non-trivial job to
train a well-performing model on ``ImageNet`` from scratch. In this tutorial, we will
try to pull necessary code together to walk you through the process of
training a model on ``ImageNet``.

.. note::

    :download:`Download Python Script train_imagenet.py<../../../scripts/classification/imagenet/train_imagenet.py>`

    Before dive into the details of the training, one can take a look at the complete
    training script. The commands to reproduce papers are given in our model zoo.

    Since the training is extremely resource consuming, we don't actually
    execute code blocks in this tutorial.

Prerequisites
-------------

**Knowledge**

We assume readers have a basic understanding of ``Gluon``. If you would
like to know more about it, we suggest the `Gluon Crash
Course <http://gluon-crash-course.mxnet.io/index.html>`__ as a good
place to start.

Also, it is suggested that readers have read our tutorials on
`CIFAR10 Training <dive_deep_cifar10.html>`_
and `ImageNet Demo <demo_imagenet.html>`_
to gain some experience of model training.

**Data Preparation**

Unlike ``CIFAR10``, we need to prepare the data manually.
If you haven't done so, please go through our tutorial on
`Prepare ImageNet Data <../examples_datasets/imagenet.html>`_ and
prepare the data.

**Hardware**

Training a deep learning model on a dataset of over 100GB is resource demanding.
Two main bottlenecks are training computation and data IO.

For training computation, it is required to have a GPU, preferably a high-end
GPU. Training on CPU is almost mission impossible, and training with a low-end GPU
may still largely limit your performance. It is the best to have multiple
strong GPUs to work together.

For data IO, we recommend a strong CPU and a SSD disk. Data loading can largely benefit
from multiple CPU threads, and a fast SSD disk. Note that in total the compressed
and extracted ``ImageNet`` data could occupy around 300GB disk space, thus a SSD with
at least 300GB is necessary.

Network structure
-----------------

If everything's prepared, Let's get started!

First, load the necessary libraries into python.

.. code-block:: python

    import argparse, time,

    import numpy as np
    import mxnet as mx

    from mxnet import gluon, nd
    from mxnet import autograd as ag
    from mxnet.gluon import nn
    from mxnet.gluon.data.vision import transforms

    from gluonvision.model_zoo import get_model
    from gluonvision.utils import makedirs, TrainingHistory

``ResNet50_v2`` is a network with balanced prediction accuracy and computational cost.
Here we use this model to demonstrate the training process.

.. code-block:: python

    # number of GPUs to use
    num_gpus = 4
    ctx = [mx.gpu(i) for i in range(num_gpus)]

    # Get the model ResNet50_v2, with 10 output classes
    net = get_model('ResNet50_v2', classes=1000)
    net.initialize(mx.init.Xavier(magnitude=2), ctx = ctx)


Note that the ResNet model we use here for ``ImageNet`` is different in structure from
the one we used to train ``CIFAR10``. Please refer to the original paper or
our implementations for details.

Data Augmentation and Data Loader
---------------------------------

Data augmentation is essential for a good result. It is similar to what we have
in the ``CIFAR10`` training tutorial, just different in the parameters.

We compose our transform functions as following:

.. code-block:: python

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    jitter_param = 0.4
    lighting_param = 0.1

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


Since ``ImageNet`` contains images with much higher resolution and quality than
``CIFAR10``, thus we can crop a larger image (224x224) as the input to the model.

For prediction, we still need a deterministic result. The transform function is:

.. code-block:: python

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])


Notice that it is important to keep the normalization consistent, since the
model only works well on input with the same distribution.

With the transform functions, we can define data loaders for our
training and validation datasets.

.. code-block:: python

    # Batch Size for Each GPU
    per_device_batch_size = 64
    # Number of data loader workers
    num_workers = 32
    # Calculate effective total batch size
    batch_size = per_device_batch_size * num_gpus

    data_path = '~/.mxnet/datasets/imagenet'

    # Set train=True for training data
    # Set shuffle=True to shuffle the training data
    train_data = gluon.data.DataLoader(
        imagenet.classification.ImageNet(data_path, train=True).transform_first(transform_train),
        batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=num_workers)

    # Set train=False for validation data
    val_data = gluon.data.DataLoader(
        imagenet.classification.ImageNet(data_path, train=False).transform_first(transform_test),
        batch_size=batch_size, shuffle=False, num_workers=num_workers)

Note that we set ``per_device_batch_size=64``, which may not suit GPUs with
Memory smaller than 12GB. Please tune the value according to your specific configuration.

The path ``'~/.mxnet/datasets/imagenet'`` is the default path if you
prepare the data with our script.

Optimizer, Loss and Metric
--------------------------

Optimizer is what improves the model during training. We use the popular
Nesterov accelerated gradient descent algorithm.

.. code-block:: python

    # Learning rate decay factor
    lr_decay = 0.1
    # Epochs where learning rate decays
    lr_decay_epoch = [30, 60, 90, np.inf]

    # Nesterov accelerated gradient descent
    optimizer = 'nag'
    # Set parameters
    optimizer_params = {'learning_rate': 0.1, 'wd': 0.0001, 'momentum': 0.9}

    # Define our trainer for net
    trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)


For classification tasks, we usually use softmax cross entropy as the
loss function.

.. code-block:: python

    loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()


With 1000 classes the model may not always rate the correct answer with the highest
rank. Besides the top-1 accuracy, we also consider top-5 accuracy as a measurement
of how well a model can predict.

At the end of every epoch, we record and print the metric scores.

.. code-block:: python

    acc_top1 = mx.metric.Accuracy()
    acc_top5 = mx.metric.TopKAccuracy(5)
    train_history = TrainingHistory(['training-top1-err', 'training-top5-err',
                                     'validation-top1-err', 'validation-top5-err'])


Validation
----------

At the end of every training epoch, we evaluate it on the validation data set,
and report the top-1 and top-5 error rate.

.. code-block:: python

    def test(ctx, val_data):
        acc_top1_val = mx.metric.Accuracy()
        acc_top5_val = mx.metric.TopKAccuracy(5)
        for i, batch in enumerate(val_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            outputs = [net(X) for X in data]
            acc_top1_val.update(label, outputs)
            acc_top5_val.update(label, outputs)

        _, top1 = acc_top1_val.get()
        _, top5 = acc_top5_val.get()
        return (1 - top1, 1 - top5)

Training
--------

After all these preparation, we can finally start our training process!
Following is the main training loop

.. code-block:: python

    epochs = 120
    lr_decay_count = 0
    log_interval = 50
    num_batch = len(train_data)

    for epoch in range(epochs):
        tic = time.time()
        btic = time.time()
        acc_top1.reset()
        acc_top5.reset()
        train_loss = 0

        if lr_decay_period == 0 and epoch == lr_decay_epoch[lr_decay_count]:
            trainer.set_learning_rate(trainer.learning_rate*lr_decay)
            lr_decay_count += 1

        for i, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            with ag.record():
                outputs = [net(X) for X in data]
                loss = [L(yhat, y) for yhat, y in zip(outputs, label)]
            for l in loss:
                l.backward()
            trainer.step(batch_size)
            acc_top1.update(label, outputs)
            acc_top5.update(label, outputs)
            train_loss += sum([l.sum().asscalar() for l in loss])
            if log_interval and not log_interval:
                _, top1 = acc_top1.get()
                _, top5 = acc_top5.get()
                err_top1, err_top5 = (1-top1, 1-top5)
                print('Epoch[%d] Batch [%d]\tSpeed: %f samples/sec\ttop1-err=%f\ttop5-err=%f'%(
                             epoch, i, batch_size*opt.log_interval/(time.time()-btic), err_top1, err_top5))
                btic = time.time()

        _, top1 = acc_top1.get()
        _, top5 = acc_top5.get()
        err_top1, err_top5 = (1-top1, 1-top5)
        train_loss /= num_batch * batch_size

        err_top1_val, err_top5_val = test(ctx, val_data)
        train_history.update([err_top1, err_top5, err_top1_val, err_top5_val])

        print('[Epoch %d] training: err-top1=%f err-top5=%f loss=%f'%(epoch, err_top1, err_top5, train_loss))
        print('[Epoch %d] time cost: %f'%(epoch, time.time()-tic))
        print('[Epoch %d] validation: err-top1=%f err-top5=%f'%(epoch, err_top1_val, err_top5_val))


We can plot the top-1 error rates with:

.. code-block:: python

    train_history.plot(['training-top1-err', 'validation-top1-err'])

If you train the model with ``epochs=120``, the plot may look like:

|image-imagenet-curve|

Next Step
---------

This is our script to help you to train a model on ``ImageNet``.

If you want like to know what can be done with the model you just
trained, please read the tutorial about `Transfer learning <transfer_learning_minc.html>`__.

Besides classification, deep learning models nowadays can do other exciting jobs
like `object detection <../examples_detection/index.html>`_,
`semantic segmentation <../examples_segmentation/index.html>`_. Check out their tutorials!

.. |image-imagenet-curve| image:: https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/classification/imagenet_resnet50_v2.png
"""
