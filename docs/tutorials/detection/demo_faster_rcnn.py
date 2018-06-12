"""1. Predict with pre-trained Faster RCNN models
==============================================

This article shows how to play with pre-trained Faster RCNN model.

First let's import some necessary libraries:
"""

from matplotlib import pyplot as plt
from mxnet import image
import gluoncv
from gluoncv import model_zoo, data, utils

######################################################################
# Load a pretrained model
# -------------------------
#
# Let's get an Faster RCNN model trained on MS-COCO
# dataset with ResNet-50 backbone. By specifying
# ``pretrained=True``, it will automatically download the model from the model
# zoo if necessary. For more pretrained models, please refer to
# :doc:`../../model_zoo/index`.

net = gluoncv.model_zoo.get_model('faster_rcnn_resnet50_coco', pretrained = True)

######################################################################
# Pre-process an image
# --------------------
#
# Next we download an image, and pre-process with preset data transforms. We
# resize the short edge of the image to 800 px and subtract the ImageNet mean.
#

im_fname = utils.download('https://github.com/dmlc/web-data/blob/master/' +
                          'gluoncv/detection/biking.jpg?raw=true',
                          path='biking.jpg')
x, scale, imw, imh = gluoncv.data.transforms.presets.faster_rcnn.load_image(im_fname)

######################################################################
# Inference and display
# ---------------------
#
# The Faster RCNN model 
#
# We can use :py:func:`gluoncv.utils.viz.plot_bbox` to visualize the
# results. We slice the results for the first image and feed them into `plot_bbox`:

rcnn_cls, rcnn_bbox_pred, mxrois, base_feat = net(x, scale)
scores, boxes, box_ids, cls_boxes = net.rcnn_prediction(
    mxrois, scale, imh, imw, rcnn_cls, rcnn_bbox_pred)

orimg = image.imread(im_fname)
ax = utils.viz.plot_bbox(orimg, boxes, scores, box_ids, class_names=data.COCODetection.CLASSES)

from matplotlib import pyplot as plt
plt.show()
