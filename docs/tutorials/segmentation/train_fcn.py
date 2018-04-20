"""Train FCN on Pascal VOC Dataset
===============================

This is a semantic segmentation tutorial using Gluon Vison, a step-by-step example.
The readers should have basic knowledge of deep learning and should be familiar with Gluon API.
New users may first go through `A 60-minute Gluon Crash Course <http://gluon-crash-course.mxnet.io/>`_.
You can `Start Training Now`_ or `Dive into Deep`_.

Start Training Now
~~~~~~~~~~~~~~~~~~

- Please follow the `installation guide <../../index.html>`_ to install MXNet and GluonVision if not yet.
  Use the quick script to `Prepare Pascal VOC Dataset <../examples_datasets/pascal_voc.html>`_.

- Clone the code::

    git clone https://github.com/dmlc/gluon-vision
    cd scipts/segmentation/

- Example training command::

    # First training on augmented set
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset pascal_aug --model fcn --backbone resnet50 --lr 0.001 --checkname mycheckpoint
    # Finetuning on original set
    CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset pascal_voc --model fcn --backbone resnet50 --lr 0.0001 --checkname mycheckpoint --resume runs/pascal_aug/fcn/mycheckpoint/checkpoint.params

  For more training command options, please run ``python train.py -h``


- Please checkout the `model_zoo <../model_zoo/index.html#semantic-segmentation>`_ for training commands of reproducing the experiments.

Dive into Deep
~~~~~~~~~~~~~~
"""
import numpy as np
import mxnet as mx
import gluonvision

##############################################################################
# Fully Convolutional Network
# ---------------------------
# 
# .. image:: https://cdn-images-1.medium.com/max/800/1*wRkj6lsQ5ckExB5BoYkrZg.png
#     :width: 70%
#     :align: center
#
# (figure redit to `Long et al. <https://arxiv.org/pdf/1411.4038.pdf>`_ )
# 
# State-of-the-art approaches of semantic segmentation are typically based on
# Fully Convolutional Network (FCN) [Long15]_.
# The key idea of a fully convolutional network is that it is "fully convolutional",
# which means it does have any fully connected layers. Therefore, the network can
# accept arbitrary input size and make dense per-pixel predictions.
# Base/Encoder network is typically pre-trained on ImageNet, because the features
# learned from diverse set of images contain rich contextual information, which
# can be beneficial for semantic segmentation.
# 
# 


##############################################################################
# Model Dilation
# --------------
# 
# The adaption of base network pre-trained on ImageNet leads to loss spatial resolution,
# because these networks are originally designed for classification task.
# Following standard implementation in recent works of semantic segmentation,
# we apply dilation strategy to the
# stage 3 and stage 4 of the pre-trained networks, which produces stride of 8
# featuremaps (models are provided in
# :class:`gluonvision.model_zoo.dilatedresnetv0.DilatedResNetV0`).
# Visualization of dilated/atrous convoution
# (figure credit to `conv_arithmetic <https://github.com/vdumoulin/conv_arithmetic>`_ ):
# 
# .. image:: https://raw.githubusercontent.com/vdumoulin/conv_arithmetic/master/gif/dilation.gif
#     :width: 40%
#     :align: center
# 
# Loading a dilated ResNet50 is simply:
# 
pretrained_net = gluonvision.model_zoo.dilatedresnetv0.dilated_resnet50(pretrained=True)

##############################################################################
# For convenience, we provide a base model for semantic segmentation, which automatically
# load the pre-trained dilated ResNet :class:`gluonvision.model_zoo.SegBaseModel`
# with a convenient method ``base_forward(input)`` to get stage 3 & 4 featuremaps:
# 
basemodel = gluonvision.model_zoo.SegBaseModel(nclass=10, aux=False)
x = mx.nd.random.uniform(shape=(1, 3, 224, 224))
c3, c4 = basemodel.base_forward(x)
print('Shapes of c3 & c4 featuremaps are ', c3.shape, c4.shape)

##############################################################################
# FCN Model
# ---------
# 
# We build a fully convolutional "head" on top of the base network,
# the FCNHead is defined as::
#
# class _FCNHead(HybridBlock):
#     # pylint: disable=redefined-outer-name
#     def __init__(self, in_channels, channels, norm_layer, **kwargs):
#         super(_FCNHead, self).__init__()
#         with self.name_scope():
#             self.block = nn.HybridSequential()
#             inter_channels = in_channels // 4
#             with self.block.name_scope():
#                 self.block.add(nn.Conv2D(in_channels=in_channels, channels=inter_channels,
#                                          kernel_size=3, padding=1))
#                 self.block.add(norm_layer(in_channels=inter_channels))
#                 self.block.add(nn.Activation('relu'))
#                 self.block.add(nn.Dropout(0.1))
#                 self.block.add(nn.Conv2D(in_channels=inter_channels, channels=channels,
#                                          kernel_size=1))
# 
#     # pylint: disable=arguments-differ
#     def hybrid_forward(self, F, x):
#         return self.block(x)
# 
# FCN model is provided in :class:`gluonvision.model_zoo.FCN`. To get
# FCN model using ResNet50 base network for Pascal VOC dataset:
model = gluonvision.model_zoo.get_fcn(dataset='pascal_voc', backbone='resnet50', pretrained=False)
print(model)

##############################################################################
# Dataset and Data Augmentation
# -----------------------------
# 
# We provide semantic segmentation datasets in :class:`gluonvision.data`.
# For example, we can easily get the Pascal VOC 2012 dataset:
train_dataset = gluonvision.data.VOCSegmentation(split='train')
print('Training images:', len(train_dataset))

##############################################################################
# We follow the standard data augmentation routine to transform the input image
# and the ground truth label map synchronously. (*Note that "nearest"
# mode upsample are applied to the label maps to avoid messing up the boundaries.*)
# We first randomly scale the input image from 0.5 to 2.0 times, then rotate
# the image from -10 to 10 degrees, and crop the image with padding if needed.
# Finally a random Gaussian blurring is applied.
#
# Random pick one example data:
from random import randint
idx = randint(0, len(train_dataset))
img, mask = train_dataset[idx]
from gluonvision.utils.viz import get_color_pallete
mask = get_color_pallete(np.array(mask), dataset='pascal_voc')
mask.save('mask.png')

##############################################################################
# Visualize the data
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
# subplot 1 for img
fig = plt.figure()
fig.add_subplot(1,2,1)
plt.imshow(np.array(img))
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
#     We apply a standard per-pixel Softmax Cross Entropy Loss to train FCN. For Pascal
#     VOC dataset, we ignore the loss from boundary class (number 22).
#     Additionally, an Auxiliary Loss as in PSPNet [Zhao17]_ at Stage 3 can be enabled when
#     training with command ``--aux``. This will create an additional FCN "head" after Stage 3.
# 
# - Learning Rate and Scheduling:
# 
#     We use different learning rate for FCN "head" and the base network. For the FCN "head",
#     we use :math:`10\times` base learning rate, because those layers are learned from scratch.
#     We use a poly-like learning rate scheduler for FCN training, provided in :class:`gluonvision.utils.PolyLRScheduler`.
#     The learning rate is given by :math:`lr = baselr \times (1-iter)^{power}`
# 
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
