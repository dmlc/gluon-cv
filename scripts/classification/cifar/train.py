"""Dive Deep Into CIFAR10
======================

Hope you enjoyed playing with our demo script. One question, as may
naturally arise in your mind: how exactly did we train the model?

In this tutorial, we will focus on answering this question.
Specifically, the following discusses

-  Model Structure
-  Data Augmentation and Data Loader
-  Optimizer, Loss and Metric
-  Validation
-  Training
-  Model save and load

Prerequisites
-------------

We assume readers have a basic understanding of ``Gluon``. If you would
like to know more about it, we suggest `Crash
Course <http://gluon-crash-course.mxnet.io/index.html>`__ as a good
place to start.

As we all know, deep learning training is way faster on GPU than on CPU.
In our demo, we only used CPU by default since that script only performs
minimum conputation. However, since we are about to train a model, it is
strongly recommended to have a platform with GPU(s).

Training usually takes several hours. While you are reading the
tutorial, it is a good idea to start the training script with

::

    python train.py --num-epochs 240 --mode hybrid --num-gpus 2 -j 32 --batch-size 64\
        --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 80,160 --model cifar_resnet20_v1

and remember to replace ``--num-gpus`` to the number of GPUs you have,
and ``-j`` to a number not larger than your CPU threads.

Let's load the necessary libraries first.

.. code:: ipython2

    from __future__ import division

    import matplotlib
    matplotlib.use('Agg')

    import argparse, time, logging, random, math

    import numpy as np
    import mxnet as mx

    from mxnet import gluon, nd
    from mxnet import autograd as ag
    from mxnet.gluon import nn
    from mxnet.gluon.data.vision import transforms

    from gluonvision.model_zoo import get_model
    from gluonvision.utils import makedirs, TrainingHistory

Network Structure
-----------------

There are numerous structures for convolutional neural networks. The
structure mainly affects

-  The upperbound of the accuracy.
-  The cost of resources, in terms of training time and memory.

Here we pick a simple yet good structure, ``cifar_resnet20_v1``, for the
tutorial.

.. code:: ipython2

    # GPUs to use
    ctx = [mx.gpu(0), mx.gpu(1)]

    # Get the model CIFAR_ResNet20_v1, with 10 output classes
    net = get_model('cifar_resnet20_v1', classes=10)
    net.initialize(mx.init.Xavier(), ctx = ctx)

Data Augmentation and Data Loader
---------------------------------

Data augmentation is a common technique used in model training. It is
proposed base on this assumption: given an object, photos with different
composition, lighting condition, or different color may still be
classified as the same object.

Here are photos taken by different people, at different time. We can all
tell that they are the photo for the same thing.

.. figure::
   :alt:

We would like to teach the model to learn about it, by playing "tricks"
on the input picture. The trick is to transform the picture with
resizing, cropping and flipping before sending to the model.

In ``Gluon``, we can compose our transform function as following:

.. code:: ipython2

    transform_train = transforms.Compose([
        # Resize the short edge of the input to 32 pixels
        transforms.Resize(32),
        # Randomly crop an area, and then resize it to be 32x32
        transforms.RandomResizedCrop(32),
        # Randomly flip the picture horizontally
        transforms.RandomFlipLeftRight(),
        # Randomly manipulate the brightness, contrast and saturation of the picture
        transforms.RandomColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        # Randomly adding noise to the picture
        transforms.RandomLighting(0.1),
        # Transpose the data from Height*Width*Channel to Channel*Height*Width
        # and map values from [0, 255] to [0,1]
        transforms.ToTensor(),
        # Normalize the image
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

You may notice that most of the operations are at random. This largely
increase the number of pictures the model can see during the training
process. The more data we have, the better our model can generalize on
unseen pictures.

On the other hand, when making prediction, we would like to remove all
random operations because we want a deterministic result. The transform
function for prediction is:

.. code:: ipython2

    transform_test = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

Notice that it is important to keep the normalization step, since the
model only works well on input with the same distribution.

With the transform functions, we can define data loaders for our
training and validation datasets.

.. code:: ipython2

    # Batch Size for Each GPU
    per_device_batch_size = 64
    # Number of data loader workers
    num_workers = 32
    # Calculate effective total batch size
    batch_size = per_device_batch_size * 2

    # Set train=True for training data
    # Set shuffle=True to shuffle the training data
    train_data = gluon.data.DataLoader(
        gluon.data.vision.CIFAR10(train=True).transform_first(transform_train),
        batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=num_workers)

    # Set train=False for validation data
    val_data = gluon.data.DataLoader(
        gluon.data.vision.CIFAR10(train=False).transform_first(transform_test),
        batch_size=batch_size, shuffle=False, num_workers=num_workers)


Now the data is ready. Let's move on to the optimizer.

Optimizer, Loss and Metric
--------------------------

Optimizer is what improves the model during training. We use the popular
Nesterov accelerated gradient descent algorithm.

.. code:: ipython2

    # Learning rate decay factor
    lr_decay = 0.1
    # Epochs where learning rate decays
    lr_decay_epoch = [80, 160, np.inf]

    # Nesterov accelerated gradient descent
    optimizer = 'nag'
    # Set parameters
    optimizer_params = {'learning_rate': 0.1, 'wd': 0.0001, 'momentum': 0.9}

    # Define our trainer for net
    trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)

In the above code, ``lr_decay`` and ``lr_decay_epoch`` are not directly
used in ``trainer``. One important idea in model training is to decrease
the learning rate in a later stage. This means the model takes larger
steps at the beginning, and smaller steps after a while.

Our plan is to have the learning rate as 0.1 at the beginning, then
divide it by 10 at the 80-th epoch, then again at the 160-th epoch.
Later we'll show how to implement it.

In order to let the optimizer work, we need a loss function. In plain
words, the loss function measures how good our model performs, and pass
the "difference" to the model. For the Nesterov algorithm we are using,
the difference is the gradient of the loss function. With the
difference, the optimizer knows towards which direction to improve the
model parameters.

For classification tasks, we usually use softmax cross entropy as the
loss function.

.. code:: ipython2

    loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()

Metric is somehow similar to loss function, but they are essentially
different.

-  Metric is how we evaluate the model performance. It is related to the
   specific task, but independent from the model training process.
-  For classification, we usually only use one loss function to train
   our model, but we can have multiplt metrics to evaluate the
   performance.
-  Loss function can be used as a metric, but sometimes it is hard to
   interpretate its value. For instance, the concept "accuracy" is
   easier to understand than "softmax cross entropy"

For simplicity, we use accuracy as the metric to monitor our training
process. Besides, we record the metric values, and will print it in the
end of the training.

.. code:: ipython2

    train_metric = mx.metric.Accuracy()
    train_history = TrainingHistory(['training-error', 'validation-error'])

Validation
----------

The existance of the validation dataset provides us a way to monitor the
training process. We have the labels on validation data, but just don't
use it to train. Therefore we can predict on the validation with the
model, and evaluate the performance at anytime.

.. code:: ipython2

    def test(ctx, val_data):
        metric = mx.metric.Accuracy()
        for i, batch in enumerate(val_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            outputs = [net(X) for X in data]
            metric.update(label, outputs)
        return metric.get()

In order to evaluate the performance, we need a metric. Then we loop
through the validation data and predict with our model. We'll plug it
into the end of each training epoch to show the improvement.

Training
--------

After all these preparation, we can finally start our training process!
Following is the script.

Notice: in order to speed up the training process, we only train the
model for 5 epochs.

.. code:: ipython2

    epochs = 5
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
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)

            # AutoGrad
            with ag.record():
                output = [net(X) for X in data]
                loss = [loss_fn(yhat, y) for yhat, y in zip(output, label)]

            # Backpropagation
            for l in loss:
                l.backward()

            # Optimize
            trainer.step(batch_size)

            # Update metrics
            train_loss += sum([l.sum().asscalar() for l in loss])
            train_metric.update(label, output)

        name, acc = train_metric.get()
        # Evaluate on Validation data
        name, val_acc = test(ctx, val_data)

        # Update history and print metrics
        train_history.update({'training-error': 1-acc, 'validation-error': 1-val_acc})
        print('[Epoch %d] train=%f val=%f loss=%f time: %f' %
            (epoch, acc, val_acc, train_loss, time.time()-tic))


.. parsed-literal::

    [Epoch 0] train=0.362941 val=0.426600 loss=86895.413788 time: 25.992320
    [Epoch 1] train=0.496014 val=0.584200 loss=69834.481152 time: 25.585870
    [Epoch 2] train=0.562099 val=0.646200 loss=61467.327183 time: 25.127841
    [Epoch 3] train=0.599479 val=0.658300 loss=56717.487179 time: 25.180641
    [Epoch 4] train=0.624119 val=0.670500 loss=52825.070946 time: 25.824175


We can plot the metric scores with:

.. code:: ipython2

    train_history.plot(items=['training-error', 'validation-error'])

This is just a plot for 5 epochs. Instead, if you change to
``epochs=240``, the plot may look like:

Model Save and Load
-------------------

Since the model is here, we may want to save it for later use, for
example, to predict the class from an arbitrary picture.

It's simple! We can do it by:

.. code:: ipython2

    net.save_params('dive_deep_cifar10_resnet20_v2.params')

Next time if you need to use it, just

.. code:: ipython2

    net.load_params('dive_deep_cifar10_resnet20_v2.params', ctx=ctx)

Next Step
---------

This is the end of our adventure with ``CIFAR10``, but there are still a
lot more we can do!

Following is a script extending our tutorial with command line arguments.
Please train a model with your own configurations.

If you would like to know how to train a model on a much larger dataset
than ``CIFAR10``, e.g. ImageNet, please read `xxx <>`__.

Or, if you want like to know what can be done with the model you just
trained, please read `finetune <>`__.

"""
from __future__ import division

import matplotlib
matplotlib.use('Agg')

import argparse, time, logging, random, math

import numpy as np
import mxnet as mx

from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

from gluonvision.model_zoo import get_model
from gluonvision.utils import makedirs, TrainingHistory

# CLI
parser = argparse.ArgumentParser(description='Train a model for image classification.')
parser.add_argument('--batch-size', type=int, default=32,
                    help='training batch size per device (CPU/GPU).')
parser.add_argument('--num-gpus', type=int, default=0,
                    help='number of gpus to use.')
parser.add_argument('--model', type=str, default='resnet',
                    help='model to use. options are resnet and wrn. default is resnet.')
parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=4, type=int,
                    help='number of preprocessing workers')
parser.add_argument('--num-epochs', type=int, default=3,
                    help='number of training epochs.')
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate. default is 0.1.')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum value for optimizer, default is 0.9.')
parser.add_argument('--wd', type=float, default=0.0001,
                    help='weight decay rate. default is 0.0001.')
parser.add_argument('--lr-decay', type=float, default=0.1,
                    help='decay rate of learning rate. default is 0.1.')
parser.add_argument('--lr-decay-period', type=int, default=0,
                    help='period in epoch for learning rate decays. default is 0 (has no effect).')
parser.add_argument('--lr-decay-epoch', type=str, default='40,60',
                    help='epoches at which learning rate decays. default is 40,60.')
parser.add_argument('--width-factor', type=int, default=1,
                    help='width factor for wide resnet. default is 1.')
parser.add_argument('--drop-rate', type=float, default=0.0,
                    help='dropout rate for wide resnet. default is 0.')
parser.add_argument('--mode', type=str,
                    help='mode in which to train the model. options are imperative, hybrid')
parser.add_argument('--save-period', type=int, default=10,
                    help='period in epoch of model saving.')
parser.add_argument('--save-dir', type=str, default='params',
                    help='directory of saved models')
parser.add_argument('--logging-dir', type=str, default='logs',
                    help='directory of training logs')
parser.add_argument('--resume-from', type=str,
                    help='resume training from the model')
parser.add_argument('--save-plot-dir', type=str, default='.',
                    help='the path to save the history plot')
opt = parser.parse_args()

batch_size = opt.batch_size
classes = 10

num_gpus = opt.num_gpus
batch_size *= max(1, num_gpus)
context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
num_workers = opt.num_workers

lr_decay = opt.lr_decay
lr_decay_epoch = [int(i) for i in opt.lr_decay_epoch.split(',')] + [np.inf]

model_name = opt.model
if model_name.startswith('cifar_wideresnet'):
    kwargs = {'classes': classes,
              'drop_rate': opt.drop_rate, 'width_factor': opt.width_factor}
else:
    kwargs = {'classes': classes}
net = get_model(model_name, **kwargs)
if opt.resume_from:
    net.load_params(opt.resume_from, ctx = context)
optimizer = 'nag'

save_period = opt.save_period
if opt.save_dir and save_period:
    save_dir = opt.save_dir
    makedirs(save_dir)
else:
    save_dir = ''
    save_period = 0

plot_path = opt.save_plot_dir

logging_handlers = [logging.StreamHandler()]
if opt.logging_dir:
    logging_dir = opt.logging_dir
    makedirs(logging_dir)
    logging_handlers.append(logging.FileHandler('%s/train_cifar10_%s.log'%(logging_dir, model_name)))

logging.basicConfig(level=logging.INFO, handlers = logging_handlers)
logging.info(opt)

transform_train = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomResizedCrop(32),
    transforms.RandomFlipLeftRight(),
    transforms.RandomColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.RandomLighting(0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

transform_test = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

def test(ctx, val_data):
    metric = mx.metric.Accuracy()
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        outputs = [net(X) for X in data]
        metric.update(label, outputs)
    return metric.get()

def train(epochs, ctx):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    net.initialize(mx.init.Xavier(), ctx=ctx)

    train_data = gluon.data.DataLoader(
        gluon.data.vision.CIFAR10(train=True).transform_first(transform_train),
        batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=num_workers)

    val_data = gluon.data.DataLoader(
        gluon.data.vision.CIFAR10(train=False).transform_first(transform_test),
        batch_size=batch_size, shuffle=False, num_workers=num_workers)

    trainer = gluon.Trainer(net.collect_params(), optimizer,
                            {'learning_rate': opt.lr, 'wd': opt.wd, 'momentum': opt.momentum})
    metric = mx.metric.Accuracy()
    train_metric = mx.metric.Accuracy()
    loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
    train_history = TrainingHistory(['training-error', 'validation-error'])

    iteration = 0
    lr_decay_count = 0

    best_val_score = 1

    for epoch in range(epochs):
        tic = time.time()
        train_metric.reset()
        metric.reset()
        train_loss = 0
        num_batch = len(train_data)
        alpha = 1

        if epoch == lr_decay_epoch[lr_decay_count]:
            trainer.set_learning_rate(trainer.learning_rate*lr_decay)
            lr_decay_count += 1

        for i, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)

            with ag.record():
                output = [net(X) for X in data]
                loss = [loss_fn(yhat, y) for yhat, y in zip(output, label)]
            for l in loss:
                l.backward()
            trainer.step(batch_size)
            train_loss += sum([l.sum().asscalar() for l in loss])

            train_metric.update(label, output)
            name, acc = train_metric.get()
            iteration += 1

        train_loss /= batch_size * num_batch
        name, acc = train_metric.get()
        name, val_acc = test(ctx, val_data)
        train_history.update([1-acc, 1-val_acc])
        train_history.plot(save_path='%s/%s_history.png'%(plot_path, model_name))

        if val_acc > best_val_score and epoch > 50:
            best_val_score = val_acc
            net.save_params('%s/%.4f-imagenet-%s-%d-best.params'%(save_dir, best_val_score, model_name, epoch))

        name, val_acc = test(ctx, val_data)
        logging.info('[Epoch %d] train=%f val=%f loss=%f time: %f' %
            (epoch, acc, val_acc, train_loss, time.time()-tic))

        if save_period and save_dir and (epoch + 1) % save_period == 0:
            net.save_params('%s/cifar10-%s-%d.params'%(save_dir, model_name, epoch))

    if save_period and save_dir:
        net.save_params('%s/cifar10-%s-%d.params'%(save_dir, model_name, epochs-1))

def main():
    if opt.mode == 'hybrid':
        net.hybridize()
    train(opt.num_epochs, context)

if __name__ == '__main__':
    main()
