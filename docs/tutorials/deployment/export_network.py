"""1. Export trained GluonCV network to JSON
============================================

It is awesome if you are enjoy using GluonCV in Python for training and testing.
At some point, you might ask: "Is it possible to deploy the existing models to somewhere out of Python environments?"

The answer is "Absolutely!", and it's super easy actually.

This article will show you how to export networks/models to be used somewhere other than Python.

"""
import gluoncv as gcv
from gluoncv.utils import export_block

################################################################################
# First of all, we need a network to play with, a pre-trained one is perfect
net = gcv.model_zoo.get_model('resnet18_v1', pretrained=True)
export_block('resnet18_v1', net, preprocess=True, layout='HWC')
print('Done.')

################################################################################
# .. hint::
#
#       Use ``preprocess=True`` will add a default preprocess layer above the network,
#       which will subtract mean [123.675, 116.28, 103.53], divide
#       std [58.395, 57.12, 57.375], and convert original image (B, H, W, C and range [0, 255]) to
#       tensor (B, C, H, W) as network input. This is the default preprocess behavior of all GluonCV
#       pre-trained models. With this preprocess head, you can use raw RGB image in C++ without
#       explicitly applying these operations.

################################################################################
# The above code generates two files: xxxx.json and xxxx.params
import glob
print(glob.glob('*.json') + glob.glob('*.params'))

################################################################################
# JSON file includes computational graph and params file includes pre-trained weights.

################################################################################
# The exportable networks are not limited to image classification models.
# We can export detection and segmentation networks as well:

# YOLO
net = gcv.model_zoo.get_model('yolo3_darknet53_coco', pretrained=True)
export_block('yolo3_darknet53_coco', net)

# FCN
net = gcv.model_zoo.get_model('fcn_resnet50_ade', pretrained=True)
export_block('fcn_resnet50_ade', net)

# MaskRCNN
net = gcv.model_zoo.get_model('mask_rcnn_resnet50_v1b_coco', pretrained=True)
export_block('mask_rcnn_resnet50_v1b_coco', net)

################################################################################
#
# We are all set here. Please checkout the other tutorials of how to use the JSON and params files.
