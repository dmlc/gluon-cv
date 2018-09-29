"""4. Train PSPNet on ADE20K Dataset
=================================

This is a tutorial of training PSPNet on ADE20K dataset using Gluon Vison.
The readers should have basic knowledge of deep learning and should be familiar with Gluon API.
New users may first go through `A 60-minute Gluon Crash Course <http://gluon-crash-course.mxnet.io/>`_.
You can `Start Training Now`_ or `Dive into Deep`_.

Start Training Now
~~~~~~~~~~~~~~~~~~

.. hint::

    Feel free to skip the tutorial because the training script is self-complete and ready to launch.

    :download:`Download Full Python Script: train.py<../../../scripts/segmentation/train.py>`

    Example training command::

        CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset ade20k --model psp --backbone resnet50 --syncbn --epochs 120 --lr 0.01 --checkname mycheckpoint

    For more training command options, please run ``python train.py -h``
    Please checkout the `model_zoo <../model_zoo/index.html#semantic-segmentation>`_ for training commands of reproducing the pretrained model.

Dive into Deep
~~~~~~~~~~~~~~
"""
import numpy as np
import mxnet as mx
from mxnet import gluon, autograd
import gluoncv

##############################################################################
# Pyramid Scene Parsing Network
# -----------------------------
#
# .. image:: https://hszhao.github.io/projects/pspnet/figures/pspnet.png
#     :width: 80%
#     :align: center
#
# (figure credit to `Zhao et al. <https://arxiv.org/pdf/1612.01105.pdf>`_ )
#
# Pyramid Scene Parsing Network (PSPNet) [Zhao17]_  exploit the
# capability of global context information by different-regionbased
# context aggregation through the pyramid pooling module.
#


##############################################################################
# PSPNet Model
# ------------
#
# A Pyramid Pooling Module is built on top of FCN, which combines multiple scale
# features with different receptive field sizes. It pools the featuremaps
# into different sizes and then concatinating together after upsampling.
#
# The Pyramid Pooling Module is defined as::
#
#     class _PyramidPooling(HybridBlock):
#         def __init__(self, in_channels, **kwargs):
#             super(_PyramidPooling, self).__init__()
#             out_channels = int(in_channels/4)
#             with self.name_scope():
#                 self.conv1 = _PSP1x1Conv(in_channels, out_channels, **kwargs)
#                 self.conv2 = _PSP1x1Conv(in_channels, out_channels, **kwargs)
#                 self.conv3 = _PSP1x1Conv(in_channels, out_channels, **kwargs)
#                 self.conv4 = _PSP1x1Conv(in_channels, out_channels, **kwargs)
#     
#         def pool(self, F, x, size):
#             return F.contrib.AdaptiveAvgPooling2D(x, output_size=size)
#     
#         def upsample(self, F, x, h, w):
#             return F.contrib.BilinearResize2D(x, height=h, width=w)
#     
#         def hybrid_forward(self, F, x):
#             _, _, h, w = x.shape
#             feat1 = self.upsample(F, self.conv1(self.pool(F, x, 1)), h, w)
#             feat2 = self.upsample(F, self.conv2(self.pool(F, x, 2)), h, w)
#             feat3 = self.upsample(F, self.conv3(self.pool(F, x, 3)), h, w)
#             feat4 = self.upsample(F, self.conv4(self.pool(F, x, 4)), h, w)
#             return F.concat(x, feat1, feat2, feat3, feat4, dim=1)
#
# PSPNet model is provided in :class:`gluoncv.model_zoo.PSPNet`. To get
# PSP model using ResNet50 base network for ADE20K dataset:
model = gluoncv.model_zoo.get_psp(dataset='ade20k', backbone='resnet50', pretrained=False)
print(model)

##############################################################################
# Dataset and Data Augmentation
# -----------------------------
#
# image transform for color normalization
from mxnet.gluon.data.vision import transforms
input_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
])

##############################################################################
# We provide semantic segmentation datasets in :class:`gluoncv.data`.
# For example, we can easily get the ADE20K dataset:
trainset = gluoncv.data.ADE20KSegmentation(split='train', transform=input_transform)
print('Training images:', len(trainset))
# set batch_size = 2 for toy example
batch_size = 2
# Create Training Loader
train_data = gluon.data.DataLoader(
    trainset, batch_size, shuffle=True, last_batch='rollover',
    num_workers=batch_size)

##############################################################################
# For data augmentation,
# we follow the standard data augmentation routine to transform the input image
# and the ground truth label map synchronously. (*Note that "nearest"
# mode upsample are applied to the label maps to avoid messing up the boundaries.*)
# We first randomly scale the input image from 0.5 to 2.0 times, then rotate
# the image from -10 to 10 degrees, and crop the image with padding if needed.
# Finally a random Gaussian blurring is applied.
#
# Random pick one example for visualization:
import random
from datetime import datetime
random.seed(datetime.now())
idx = random.randint(0, len(trainset))
img, mask = trainset[idx]
from gluoncv.utils.viz import get_color_pallete, DeNormalize
# get color pallete for visualize mask
mask = get_color_pallete(mask.asnumpy(), dataset='ade20k')
mask.save('mask.png')
# denormalize the image
img = DeNormalize([.485, .456, .406], [.229, .224, .225])(img)
img = np.transpose((img.asnumpy()*255).astype(np.uint8), (1, 2, 0))

##############################################################################
# Plot the image and mask
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
# subplot 1 for img
fig = plt.figure()
fig.add_subplot(1,2,1)

plt.imshow(img)
# subplot 2 for the mask
mmask = mpimg.imread('mask.png')
fig.add_subplot(1,2,2)
plt.imshow(mmask)
# display
plt.show()

##############################################################################
# Training Details
# ----------------
#
# - Training Losses:
#
#     We apply a standard per-pixel Softmax Cross Entropy Loss to train PSPNet. 
#     Additionally, an Auxiliary Loss as in PSPNet [Zhao17]_ at Stage 3 can be enabled when
#     training with command ``--aux``. This will create an additional FCN "head" after Stage 3.
#
from gluoncv.loss import MixSoftmaxCrossEntropyLoss
criterion = MixSoftmaxCrossEntropyLoss(aux=True)

##############################################################################
# - Learning Rate and Scheduling:
#
#     We use different learning rate for PSP "head" and the base network. For the PSP "head",
#     we use :math:`10\times` base learning rate, because those layers are learned from scratch.
#     We use a poly-like learning rate scheduler for FCN training, provided in :class:`gluoncv.utils.LRScheduler`.
#     The learning rate is given by :math:`lr = baselr \times (1-iter)^{power}`
# 
lr_scheduler = gluoncv.utils.LRScheduler(mode='poly', baselr=0.001, niters=len(train_data), 
                                          nepochs=50)

##############################################################################
# - Dataparallel for multi-gpu training, using cpu for demo only
from gluoncv.utils.parallel import *
ctx_list = [mx.cpu(0)]
model = DataParallelModel(model, ctx_list)
criterion = DataParallelCriterion(criterion, ctx_list)

##############################################################################
# - Create SGD solver
kv = mx.kv.create('local')
optimizer = gluon.Trainer(model.module.collect_params(), 'sgd',
                          {'lr_scheduler': lr_scheduler,
                           'wd':0.0001,
                           'momentum': 0.9,
                           'multi_precision': True},
                          kvstore = kv)

##############################################################################
# The training loop
# -----------------
#
train_loss = 0.0
epoch = 0
for i, (data, target) in enumerate(train_data):
    lr_scheduler.update(i, epoch)
    with autograd.record(True):
        outputs = model(data)
        losses = criterion(outputs, target)
        mx.nd.waitall()
        autograd.backward(losses)
    optimizer.step(batch_size)
    for loss in losses:
        train_loss += loss.asnumpy()[0] / len(losses)
    print('Epoch %d, batch %d, training loss %.3f'%(epoch, i, train_loss/(i+1)))
    # just demo for 2 iters
    if i > 1:
        print('Terminated for this demo...')
        break


##############################################################################
# You can `Start Training Now`_.
#
# References
# ----------
#
# .. [Long15] Long, Jonathan, Evan Shelhamer, and Trevor Darrell. \
#     "Fully convolutional networks for semantic segmentation." \
#     Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.
#
# .. [Zhao17] Zhao, Hengshuang, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, and Jiaya Jia. \
#     "Pyramid scene parsing network." IEEE Conf. on Computer Vision and Pattern Recognition (CVPR). 2017.
#

