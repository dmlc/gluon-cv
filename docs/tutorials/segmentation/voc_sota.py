"""6. Reproducing SoTA on Pascal VOC Dataset
=========================================

This is a semantic segmentation tutorial for reproducing state-of-the-art results
on Pascal VOC dataset using Gluon CV toolkit.

Start Training Now
~~~~~~~~~~~~~~~~~~

.. hint::

    Feel free to skip the tutorial because the training script is self-complete and ready to launch.

    :download:`Download Full Python Script: train.py<../../../scripts/segmentation/train.py>`

    Example training command for training DeepLabV3::

        # First finetuning COCO dataset pretrained model on the augmented set
        # If you would like to train from scratch on COCO, please see deeplab_resnet101_coco.sh
        CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset pascal_aug --model-zoo deeplab_resnet101_coco --aux --lr 0.001 --syncbn --ngpus 4 --checkname res101
        # Finetuning on original set
        CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset pascal_voc --model deeplab --aux --backbone resnet101 --lr 0.0001 --syncbn --ngpus 4 --checkname res101 --resume runs/pascal_aug/deeplab/res101/checkpoint.params

    For more training command options, please run ``python train.py -h``
    Please checkout the `model_zoo <../model_zoo/index.html#semantic-segmentation>`_ for training commands of reproducing the pretrained model.

"""

import numpy as np
import mxnet as mx
from mxnet import gluon, autograd
import gluoncv

##############################################################################
# Evils in the Training Details
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# State-of-the-art results [Chen17]_ [Zhao17]_ on Pascal VOC dataset are typically
# difficult to reproduce due to the sophisticated training details.
# In this tutorial we walk through our state-of-the-art implementation step-by-step.
#
# DeepLabV3 Implementation
# ------------------------
#
# We implemented state-of-the-art semantic segmentation model of DeepLabV3 in Gluon-CV.
# Atrous Spatial Pyramid Pooling (ASPP) is the key part of DeepLabV3 model, which is
# built on top of FCN. It combines multiple scale features with different receptive
# field sizes, by using different atrous rate of dilated convolution and incorporating
# a global pooling branch with a global receptive field.
#
# The ASPP module is defined as::
#
#     class _ASPP(nn.HybridBlock):
#         def __init__(self, in_channels, atrous_rates, norm_layer, norm_kwargs):
#             super(_ASPP, self).__init__()
#             out_channels = 256
#             b0 = nn.HybridSequential()
#             with b0.name_scope():
#                 b0.add(nn.Conv2D(in_channels=in_channels, channels=out_channels,
#                                  kernel_size=1, use_bias=False))
#                 b0.add(norm_layer(in_channels=out_channels, **norm_kwargs))
#                 b0.add(nn.Activation("relu"))
#
#             rate1, rate2, rate3 = tuple(atrous_rates)
#             b1 = _ASPPConv(in_channels, out_channels, rate1, norm_layer, norm_kwargs)
#             b2 = _ASPPConv(in_channels, out_channels, rate2, norm_layer, norm_kwargs)
#             b3 = _ASPPConv(in_channels, out_channels, rate3, norm_layer, norm_kwargs)
#             b4 = _AsppPooling(in_channels, out_channels, norm_layer=norm_layer,
#                               norm_kwargs=norm_kwargs)
#
#             self.concurent = gluon.contrib.nn.HybridConcurrent(axis=1)
#             with self.concurent.name_scope():
#                 self.concurent.add(b0)
#                 self.concurent.add(b1)
#                 self.concurent.add(b2)
#                 self.concurent.add(b3)
#                 self.concurent.add(b4)
#
#             self.project = nn.HybridSequential()
#             with self.project.name_scope():
#                 self.project.add(nn.Conv2D(in_channels=5*out_channels, channels=out_channels,
#                                            kernel_size=1, use_bias=False))
#                 self.project.add(norm_layer(in_channels=out_channels, **norm_kwargs))
#                 self.project.add(nn.Activation("relu"))
#                 self.project.add(nn.Dropout(0.5))
#
#         def hybrid_forward(self, F, x):
#             return self.project(self.concurent(x))
#
# DeepLabV3 model is provided in :class:`gluoncv.model_zoo.DeepLabV3`. To get
# DeepLabV3 model using ResNet50 base network for VOC dataset:
#
model = gluoncv.model_zoo.get_deeplab (dataset='pascal_voc', backbone='resnet50', pretrained=False)
print(model)

##############################################################################
# COCO Pretraining
# ----------------
#
# COCO dataset is an large instance segmentation dataset with 80 categories, which has 127K
# training images. From the training set of MS-COCO dataset, we select with
# images containing the 20 classes shared with PASCAL dataset with more than 1,000 labeled pixels,
# resulting 92.5K images. All the other classes are marked as background. You can simply get this
# dataset using the following command:
#
#

# image transform for color normalization
from mxnet.gluon.data.vision import transforms
input_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
])

# get the dataset
trainset = gluoncv.data.COCOSegmentation(split='train', transform=input_transform)
print('Training images:', len(trainset))

# set batch_size = 2 for toy example
batch_size = 2
# Create Training Loader
train_data = gluon.data.DataLoader(
    trainset, batch_size, shuffle=True, last_batch='rollover',
    num_workers=0)


##############################################################################
# Plot an Example of generated images:
#

# Random pick one example for visualization:
import random
from datetime import datetime
random.seed(datetime.now())
idx = random.randint(0, len(trainset))
img, mask = trainset[idx]
from gluoncv.utils.viz import get_color_pallete, DeNormalize
# get color pallete for visualize mask
mask = get_color_pallete(mask.asnumpy(), dataset='coco')
mask.save('mask.png')
# denormalize the image
img = DeNormalize([.485, .456, .406], [.229, .224, .225])(img)
img = np.transpose((img.asnumpy()*255).astype(np.uint8), (1, 2, 0))

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
# Direct launch command of the COCO pretraining::
#
#   CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset coco --model deeplab --aux --backbone resnet101 --lr 0.01 --syncbn --ngpus 4 --checkname res101 --epochs 30
#
# You can also skip the COCO pretraining by getting the pretrained model::
#
#   from gluoncv import model_zoo
#   model_zoo.get_model('deeplab_resnet101_coco', pretrained=True)
#
# Pascal VOC and the Augmented Set
# --------------------------------
#
# Pascal VOC dataset [Everingham10]_ has 2,913 images in training and validation sets.
# The augmented set [Hariharan15]_ has 10,582 and 1449 training and validation images.
# We first fine-tune the COCO pretrained model on Pascal Augmentation dataset, then
# fine-tune again on Pascal VOC dataset to get the best performance.
#
# Learning Rates
# --------------
#
# We use different learning rates for pretrained base network and the DeepLab head without
# pretrained weights.
# We enlarge the learning rate of the head by 10 times. A poly-like cosine learning rate
# scheduling strategy is used.
# The learning rate is given by :math:`lr = base\_lr \times (1-iters/niters)^{power}`. Please
# check https://gluon-cv.mxnet.io/api/utils.html#gluoncv.utils.LRScheduler for more details.
lr_scheduler = gluoncv.utils.LRScheduler(mode='poly', base_lr=0.01,
                                         nepochs=30, iters_per_epoch=len(train_data), power=0.9)

##############################################################################
# We first use the base learning rate of 0.01 to pretrain on MS-COCO dataset,
# then we divide the base learning rate by 10 times and 100 times respectively when
# fine-tuning on Pascal Augmented dataset and Pascal VOC original dataset.
#

##############################################################################
# You can `Start Training Now`_.
#
# References
# ----------
#
# .. [Chen17] Chen, Liang-Chieh, et al. "Rethinking atrous convolution for semantic image segmentation." \
#     arXiv preprint arXiv:1706.05587 (2017).
#
# .. [Zhao17] Zhao, Hengshuang, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, and Jiaya Jia. \
#     "Pyramid scene parsing network." IEEE Conf. on Computer Vision and Pattern Recognition (CVPR). 2017.
#
# .. [Everingham10] Everingham, Mark, Luc Van Gool, Christopher KI Williams, John Winn, \
#     and Andrew Zisserman. "The pascal visual object classes (voc) challenge." \
#     International journal of computer vision 88, no. 2 (2010): 303-338.
#
# .. [Hariharan15] Hariharan, Bharath, Pablo Arbeláez, Ross Girshick, and Jitendra Malik. \
#     "Hypercolumns for object segmentation and fine-grained localization." In Proceedings of \
#     the IEEE conference on computer vision and pattern recognition, pp. 447-456. 2015.
