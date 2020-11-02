"""02. Train SiamRPN on COCO、VID、DET、Youtube_bb
==================================================

This is a Single Obejct Tracking tutorial using Gluon CV toolkit, a step-by-step example.
The readers should have basic knowledge of deep learning and should be familiar with Gluon API.
New users may first go through `A 60-minute Gluon Crash Course <http://gluon-crash-course.mxnet.io/>`_.
You can `Start Training Now`_ or `Dive into Deep`_.

Start Training Now
~~~~~~~~~~~~~~~~~~

.. note::

    Feel free to skip the tutorial because the training script is self-complete and ready to launch.

    :download:`Download Full Python Script: train.py<../../../scripts/tracking/train.py>`
    :download:`Download Full Python Script: test.py<../../../scripts/tracking/test.py>`

    Example training command::

        python train.py   --ngpus 8 --epochs 50 --base-lr 0.005

    Example test command::

        python test.py   --model-path  --results-path

    Please checkout the `model_zoo <../model_zoo/index.html#single_object_tracking>`_ for training and test commands of reproducing the pretrained model.


Network Structure
-----------------

First, let's import the necessary libraries into python.

"""
import mxnet as mx
import time
import numpy as np
from mxnet import gluon, nd, autograd
from mxnet.contrib import amp

import gluoncv
from gluoncv.utils import LRScheduler, LRSequential, split_and_load
from gluoncv.data.tracking_data.track import TrkDataset
from gluoncv.model_zoo import get_model
from gluoncv.loss import SiamRPNLoss

################################################################
#
# `SiamRPN <http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_High_Performance_Visual_CVPR_2018_paper.pdf>`_ is a widely adopted Single Object Tracking method.
# Send the template frame and detection frame to the siamese network,
# and get the score map and coordinate regression of the anchor through the RPN network and cross correlation layers.

# number of GPUs to use
num_gpus = 1
ctx = [mx.cpu(0)]
batch_size = 32  # adjust to 128 if memory is sufficient
epochs = 1
# Get the model siamrpn_alexnet with SiamRPN backbone
net = get_model('siamrpn_alexnet_v2_otb15', bz=batch_size, is_train=True, ctx=ctx)
net.collect_params().reset_ctx(ctx)
print(net)

# We provide Single Obejct datasets in :class:`gluoncv.data`.
# For example, we can easily get the vid,det,coco dataset:
'''``python scripts/datasets/ilsvrc_det.py``
   ``python scripts/datasets/ilsvrc_vid.py``
   ``python scripts/datasets/coco_tracking.py``'''
# If you want to download youtube_bb dataset,you can You can follow it from the following `link <https://github.com/STVIR/pysot/tree/master/training_dataset/yt_bb>`:

# prepare dataset and dataloader
train_dataset = TrkDataset(train_epoch=epochs)
print('Training images:', len(train_dataset))
workers = 0
train_loader = gluon.data.DataLoader(train_dataset,
                                     batch_size=batch_size,
                                     last_batch='discard',
                                     num_workers=workers)

def train_batch_fn(data, ctx):
    """split and load data in GPU"""
    template = split_and_load(data[0], ctx_list=ctx, batch_axis=0)
    search = split_and_load(data[1], ctx_list=ctx, batch_axis=0)
    label_cls = split_and_load(data[2], ctx_list=ctx, batch_axis=0)
    label_loc = split_and_load(data[3], ctx_list=ctx, batch_axis=0)
    label_loc_weight = split_and_load(data[4], ctx_list=ctx, batch_axis=0)
    return template, search, label_cls, label_loc, label_loc_weight

#############################################################################
# Training Details
# ----------------
#
# - Training Losses:
#
#     We apply Softmax Cross Entropy Loss and L2 loss to train SiamRPN.
#

criterion = SiamRPNLoss(batch_size)

##############################################################################
# - Learning Rate and Scheduling:
lr_scheduler = LRScheduler(mode='step', base_lr=0.005, step_epoch=[0],
                           nepochs=epochs, iters_per_epoch=len(train_loader), power=0.9)

##############################################################################
# - Dataparallel for multi-gpu training, using cpu for demo only
# Stochastic gradient descent
optimizer = 'sgd'
# Set parameters
optimizer_params = {'lr_scheduler': lr_scheduler,
                     'wd': 1e-4,
                     'momentum': 0.9,
                     'learning_rate': 0.005}
trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)
cls_weight = 1.0
loc_weight = 1.2

################################################################
# Training
# --------
#
# After all the preparations, we can finally start training!
# Following is the script.
#
# .. note::
#   In your experiments, we recommend setting ``epochs=50`` for the dataset.
#   We will skip the training in this tutorial
epochs = 0

for epoch in range(epochs):
    loss_total_val = 0
    loss_loc_val = 0
    loss_cls_val = 0
    batch_time = time.time()
    for i, data in enumerate(train_loader):
        template, search, label_cls, label_loc, label_loc_weight = train_batch_fn(data, ctx)
        cls_losses = []
        loc_losses = []
        total_losses = []
        with autograd.record():
            for j in range(len(ctx)):
                cls, loc = net(template[j], search[j])
                label_cls_temp = label_cls[j].reshape(-1).asnumpy()
                pos_index = np.argwhere(label_cls_temp == 1).reshape(-1)
                neg_index = np.argwhere(label_cls_temp == 0).reshape(-1)
                if len(pos_index):
                    pos_index = nd.array(pos_index, ctx=ctx[j])
                else:
                    pos_index = nd.array(np.array([]), ctx=ctx[j])
                if len(neg_index):
                    neg_index = nd.array(neg_index, ctx=ctx[j])
                else:
                    neg_index = nd.array(np.array([]), ctx=ctx[j])
                cls_loss, loc_loss = criterion(cls, loc, label_cls[j], pos_index, neg_index,
                                                label_loc[j], label_loc_weight[j])
                total_loss = cls_weight*cls_loss+loc_weight*loc_loss
                cls_losses.append(cls_loss)
                loc_losses.append(loc_loss)
                total_losses.append(total_loss)
            autograd.backward(total_losses)
        trainer.step(batch_size)
        loss_total_val += sum([l.mean().asscalar() for l in total_losses]) / len(total_losses)
        loss_loc_val += sum([l.mean().asscalar() for l in loc_losses]) / len(loc_losses)
        loss_cls_val += sum([l.mean().asscalar() for l in cls_losses]) / len(cls_losses)
        print('Epoch %d iteration %04d/%04d: loc loss %.3f, cls loss %.3f, \
               training loss %.3f, batch time %.3f'% \
               (epoch, i, len(train_loader), loss_loc_val/(i+1), loss_cls_val/(i+1),
                loss_total_val/(i+1), time.time()-batch_time))
        batch_time = time.time()
        mx.nd.waitall()

##############################################################################
# You can `Start Training Now`_.
#
# References
# ----------
#
# ..  Bo Li, Junjie Yan, Wei Wu, Zheng Zhu, Xiaolin Hu. \
#     "High Performance Visual Tracking With Siamese Region Proposal Network。" \
#     Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.
#
