"""2. Train SSD on Pascal VOC dataset
======================================

This tutorial goes through the basic building blocks of object detection
provided by GluonCV.
Specifically, we show how to build a state-of-the-art Single Shot Multibox
Detection [Liu16]_ model by stacking GluonCV components.
This is also a good starting point for your own object detection project.

.. hint::

    You can skip the rest of this tutorial and start training your SSD model
    right away by downloading this script:

    :download:`Download train_ssd.py<../../../scripts/detection/ssd/train_ssd.py>`

    Example usage:

    Train a default vgg16_atrous 300x300 model with Pascal VOC on GPU 0:

    .. code-block:: bash

        python train_ssd.py

    Train a resnet50_v1 512x512 model on GPU 0,1,2,3:

    .. code-block:: bash

        python train_ssd.py --gpus 0,1,2,3 --network resnet50_v1 --data-shape 512

    Check the supported arguments:

    .. code-block:: bash

        python train_ssd.py --help

"""

##########################################################
# Dataset
# -------
#
# Please first go through this :ref:`sphx_glr_build_examples_datasets_pascal_voc.py` tutorial to setup Pascal
# VOC dataset on your disk.
# Then, we are ready to load training and validation images.

from gluoncv.data import VOCDetection
# typically we use 2007+2012 trainval splits for training data
train_dataset = VOCDetection(splits=[(2007, 'trainval'), (2012, 'trainval')])
# and use 2007 test as validation data
val_dataset = VOCDetection(splits=[(2007, 'test')])

print('Training images:', len(train_dataset))
print('Validation images:', len(val_dataset))

##########################################################
# Data transform
# ------------------
# We can read an image-label pair from the training dataset:
train_image, train_label = train_dataset[0]
bboxes = train_label[:, :4]
cids = train_label[:, 4:5]
print('image:', train_image.shape)
print('bboxes:', bboxes.shape, 'class ids:', cids.shape)

##############################################################################
# Plot the image, together with the bounding box labels:
from matplotlib import pyplot as plt
from gluoncv.utils import viz

ax = viz.plot_bbox(train_image.asnumpy(), bboxes, labels=cids, class_names=train_dataset.classes)
plt.show()

##############################################################################
# Validation images are quite similar to training because they were
# basically split randomly to different sets
val_image, val_label = val_dataset[0]
bboxes = val_label[:, :4]
cids = val_label[:, 4:5]
ax = viz.plot_bbox(val_image.asnumpy(), bboxes, labels=cids, class_names=train_dataset.classes)
plt.show()

##############################################################################
# For SSD networks, it is critical to apply data augmentation (see explanations in paper [Liu16]_).
# We provide tons of image and bounding box transform functions to do that.
# They are very convenient to use as well.
from gluoncv.data.transforms import presets
from gluoncv import utils
from mxnet import nd

##############################################################################
width, height = 512, 512  # suppose we use 512 as base training size
train_transform = presets.ssd.SSDDefaultTrainTransform(width, height)
val_transform = presets.ssd.SSDDefaultValTransform(width, height)

##############################################################################
utils.random.seed(233)  # fix seed in this tutorial

##############################################################################
# apply transforms to train image
train_image2, train_label2 = train_transform(train_image, train_label)
print('tensor shape:', train_image2.shape)

##############################################################################
# Images in tensor are distorted because they no longer sit in (0, 255) range.
# Let's convert them back so we can see them clearly.
train_image2 = train_image2.transpose((1, 2, 0)) * nd.array((0.229, 0.224, 0.225)) + nd.array((0.485, 0.456, 0.406))
train_image2 = (train_image2 * 255).clip(0, 255)
ax = viz.plot_bbox(train_image2.asnumpy(), train_label2[:, :4],
                   labels=train_label2[:, 4:5],
                   class_names=train_dataset.classes)
plt.show()

##############################################################################
# apply transforms to validation image
val_image2, val_label2 = val_transform(val_image, val_label)
val_image2 = val_image2.transpose((1, 2, 0)) * nd.array((0.229, 0.224, 0.225)) + nd.array((0.485, 0.456, 0.406))
val_image2 = (val_image2 * 255).clip(0, 255)
ax = viz.plot_bbox(val_image2.clip(0, 255).asnumpy(), val_label2[:, :4],
                   labels=val_label2[:, 4:5],
                   class_names=train_dataset.classes)
plt.show()

##############################################################################
# Transforms used in training include random expanding, random cropping, color distortion, random flipping, etc.
# In comparison, validation transforms are simpler and only resizing and color normalization is used.

##########################################################
# Data Loader
# ------------------
# We will iterate through the entire dataset many times during training.
# Keep in mind that raw images have to be transformed to tensors
# (mxnet uses BCHW format) before they are fed into neural networks.
# In addition, to be able to run in mini-batches,
# images must be resized to the same shape.
#
# A handy DataLoader would be very convenient for us to apply different transforms and aggregate data into mini-batches.
#
# Because the number of objects varys a lot across images, we also have
# varying label sizes. As a result, we need to pad those labels to the same size.
# To deal with this problem, GluonCV provides DetectionDataLoader,
# which handles padding automatically.

from gluoncv.data import DetectionDataLoader

batch_size = 4  # for tutorial, we use smaller batch-size
num_workers = 0  # you can make it larger(if your CPU has more cores) to accelerate data loading

train_loader = DetectionDataLoader(train_dataset.transform(train_transform), batch_size, shuffle=True,
                                   last_batch='rollover', num_workers=num_workers)
val_loader = DetectionDataLoader(val_dataset.transform(val_transform), batch_size, shuffle=False,
                                 last_batch='keep', num_workers=num_workers)

for ib, batch in enumerate(train_loader):
    if ib > 5:
        break
    print('data:', batch[0].shape, 'label:', batch[1].shape)

##########################################################
# SSD Network
# ------------------
# GluonCV's SSD implementation is a composite Gluon HybridBlock
# (which means it can be exported
# to symbol to run in C++, Scala and other language bindings.
# We will cover this usage in future tutorials).
# In terms of structure, SSD networks are composed of base feature extraction
# network, anchor generators, class predictors and bounding box offset predictors.
#
# For more details on how SSD detector works, please refer to our introductory
# [tutorial](http://gluon.mxnet.io/chapter08_computer-vision/object-detection.html)
# You can also refer to the original paper to learn more about the intuitions
# behind SSD.
#
# `Gluon Model Zoo <../../model_zoo/index.html>`__ has a lot of built-in SSD networks.
# You can load your favorate one with one simple line of code:
from gluoncv import model_zoo
net = model_zoo.get_model('ssd_300_vgg16_atrous_voc', pretrained_base=False)
print(net)

##############################################################################
# SSD network is a HybridBlock as mentioned before. You can call it with an input as:
import mxnet as mx
x = mx.nd.zeros(shape=(1, 3, 300, 300))
net.initialize()
cids, scores, bboxes = net(x)

##############################################################################
# SSD returns three values, where ``cids`` are the class labels,
# ``scores`` are confidence scores of each prediction,
# and ``bboxes`` are absolute coordinates of corresponding bounding boxes.

##########################################################
# Training targets
# ------------------
# Unlike a single ``SoftmaxCrossEntropyLoss`` used in image classification,
# the loss used in SSD is more complicated.
# Don't worry though, because we have these modules available out of the box.

##############################################################################
# Checkout the ``target_generator`` in SSD networks.
print(net.target_generator)

##############################################################################
# You can observe that there are:
#
# - A bounding box encoder which transfers raw coordinates to bbox prediction targets.
#
# - A class encoder which generates class labels for each anchor box.
#
# - Matcher and samplers used to apply various advanced strategies described in paper.
#
# .. hint::
#
#   Please checkout the full :download:`training script <../../../scripts/detection/ssd/train_ssd.py>` for complete implementation.

##########################################################
# References
# ----------
#
# .. [Liu16] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg. SSD: Single Shot MultiBox Detector. ECCV 2016.
