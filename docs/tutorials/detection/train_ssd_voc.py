"""Train SSD on Pascal VOC dataset
==================================

This article walk you through the components GluonVision provided to you
that are very useful to start an object detection project. By going
through this tutorial, we show how stacking the existing modules can
produce a SOTA Single Shot Multibox Detection [Liu16]_ model.

A Python training script to reproduce SOTA models
-------------------------------------------------
Feel free to skip this tutorial because the training script is
self-complete and only requires a single command line to launch.

:download:`Download Full Python Script train_ssd.py<../../../scripts/detection/ssd/train_ssd.py>`

Example usage:

.. code-block:: bash

    # train a default vgg16_atrous 300x300 model with Pascal VOC on GPU 0
    python train_ssd.py
    # train a resnet50_v1 512x512 model on GPU 0,1,2,3
    python train_ssd.py --gpus 0,1,2,3 --network resnet50_v1 --data-shape 512
    # check the supported arguments
    python train_ssd.py --help

"""

##########################################################
# Dataset
# -------
#
# We hope you already read this :ref:`pascal_voc` so Pascal VOC dataset is well sitting on your disk.
# If so we are ready to load some training and validation images.
from gluonvision.data import VOCDetection
# typically we use 2007+2012 trainval splits as training data
train_dataset = VOCDetection(splits=[(2007, 'trainval'), (2012, 'trainval')])
# use 2007 test as validation
val_dataset = VOCDetection(splits=[(2007, 'test')])

print('Training images:', len(train_dataset))
print('Validation images:', len(val_dataset))

##########################################################
# Data transform
# ------------------
# We can read a image and label pair from training dataset:
train_image, train_label = train_dataset[0]
bboxes = train_label[:, :4]
cids = train_label[:, 4:5]
print('image:', train_image.shape)
print('bboxes:', bboxes.shape, 'class ids:', cids.shape)

##############################################################################
# We could illustrate the image, together with the bounding box labels:
from matplotlib import pyplot as plt
from gluonvision.utils import viz

ax = viz.plot_bbox(train_image.asnumpy(), bboxes, labels=cids, class_names=train_dataset.classes)
plt.show()

##############################################################################
# At this point, validation images are quite similar to training because they were
# basically splited randomly to different sets
val_image, val_label = val_dataset[0]
bboxes = val_label[:, :4]
cids = val_label[:, 4:5]
ax = viz.plot_bbox(val_image.asnumpy(), bboxes, labels=cids, class_names=train_dataset.classes)
plt.show()

##############################################################################
# For SSD networks, it is critical to apply data augmentation (see explanations in paper [Liu16]_).
# We provide tons of image and bounding box transform functions to supply that.
# It is very convenient to use as well.
from gluonvision.data.transforms import presets
from gluonvision import utils
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
# Images directly from tensor is distorted because they no longer sit in (0, 255) range.
# Let's convert it back so we can see it clearly.
train_image2 = train_image2.transpose((1, 2, 0)) * nd.array((0.229, 0.224, 0.225)) + nd.array((0.485, 0.456, 0.406))
train_image2 = (train_image2 * 255).clip(0, 255)
ax = viz.plot_bbox(train_image2.asnumpy(), train_label2[:, :4],
                   labels=train_label2[:, 4:5], class_names=train_dataset.classes)
plt.show()

##############################################################################
# apply transforms to validation image
val_image2, val_label2 = val_transform(val_image, val_label)
val_image2 = val_image2.transpose((1, 2, 0)) * nd.array((0.229, 0.224, 0.225)) + nd.array((0.485, 0.456, 0.406))
val_image2 = (val_image2 * 255).clip(0, 255)
ax = viz.plot_bbox(val_image2.clip(0, 255).asnumpy(), val_label2[:, :4],
                   labels=val_label2[:, 4:5], class_names=train_dataset.classes)
plt.show()

##############################################################################
# Transforms used in training include random expanding, random cropping, color distortion, random flipping, etc.
# In comparison, validation transforms are conservative, where only resizing and color normalization is used.

##########################################################
# Data Loader
# ------------------
# We want iterate through the entire dataset many times during training.
# Keep in mind that raw images have to be transformed into tensors(mxnet use BCHW format) before they are fed into neural networks.
# Besides, to be able to run in mini-batches, images must be resized to same shape.

# A handy DataLoader would be very convenient for us to apply different transforms and aggregate data into mini-batches.

# Because number of objects varys a lot in different images, we have fluctuating label sizes. As a result, we need to pad those labels to the same size.
# In response, we have DetectionDataLoader ready for you which handles it automatically.
from gluonvision.data import DetectionDataLoader

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
# SSD network is a composite Gluon HybridBlock(which means it can be exported to symbol to run in C++, Scala and other language bindings, but we will cover it future tutorials).
# In terms of structure, SSD networks are composed of feature extraction base network, anchor generators, class predictors and bounding box offsets predictors.

# If you have read our introductory [tutorial](http://gluon.mxnet.io/chapter08_computer-vision/object-detection.html) of SSD, you may have better idea how it works.
# You can also refer to original paper and entry level tutorials for idea that support SSD.

# GluonVision has a model zoo which has a lot of built-in SSD networks.
# Therefore you can simply load them from model_zoo module like this:
from gluonvision import model_zoo
net = model_zoo.get_model('ssd_300_vgg16_atrous_voc', pretrained_base=False)
print(net)

##############################################################################
# SSD network is a HybridBlock as mentioned before. So you can call it with an input as simple as:
import mxnet as mx
x = mx.nd.zeros(shape=(1, 3, 300, 300))
net.initialize()
cids, scores, bboxes = net(x)

##############################################################################
# where ``cids`` is the class labels, ``scores`` are confidences of each predictions, ``bboxes`` are corresponding bounding boxes' absolute coordinates.

##########################################################
# Training targets
# ------------------
# Unlike a single ``SoftmaxCrossEntropyLoss`` used in image classification, the losses used in SSD is more complicated.
# Don't worry though, because we have these modules available out of box.

##############################################################################
# Checkout the ``target_generator`` in SSD networks.
print(net.target_generator)

##############################################################################
# You can see there are: a bounding boxes encoder which transfers raw coordinates to bbox prediction targets, a class encoder which generates class labels for each anchor box.
# Matcher and samplers included are used to apply various advanced strategies described in paper.

##########################################################
# References
# ----------
#
# .. [Liu16] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg. SSD: Single Shot MultiBox Detector. ECCV 2016.
