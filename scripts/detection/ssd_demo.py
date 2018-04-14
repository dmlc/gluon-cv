# -*- coding: utf-8 -*-
"""
Getting started with SSD pre-trained models
===========================================

This article is an introductory tutorial to play with pre-trained
models with several lines of code.

For us to begin with, mxnet and gluonvvision modules are required to be installed.

A quick solution is

```
pip install --user --pre mxnet gluonvision
```

or please refer to offical installation guide:

http://gluon-vision.mxnet.io.s3-website-us-west-2.amazonaws.com/index.html#installation

"""
import mxnet as mx
import gluonvision as gv
from matplotlib import pyplot as plt


######################################################################
# Download test image and pretrained model
# ---------------------------------------------
# In this section, we grab a pretrained model and test image to play with

# try grab a 300x300 model trained on Pascal voc dataset.
# it will automatically download from s3 servers if not exists
net = gv.model_zoo.get_model('ssd_300_vgg16_atrous_voc', pretrained=True)

# a demo image, feel free to use your own image
gv.utils.download("https://cloud.githubusercontent.com/assets/3307514/" +
    "20012568/cbc2d6f6-a27d-11e6-94c3-d35a9cb47609.jpg", 'street.jpg')
image_list = ['street.jpg']

######################################################################
# Preprocess image so it is normalized and converted to tensor
# ---------------------------------------------
# Images are scaled to (0, 1), mean pixel values are (0.485, 0.456, 0.406)
# std values are (0.229, 0.224, 0.225), pixel order is RGB.
def process_img(im_name, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Pre-process image to tensor required by network."""
    img = mx.image.imread(im_name)
    img = mx.image.resize_short(img, 480)
    orig_img = img.asnumpy().astype('uint8')
    img = mx.nd.image.to_tensor(img)
    img = mx.nd.image.normalize(img, mean=mean, std=std)
    return img, orig_img

######################################################################
# Inference and visualize detection results
# ---------------------------------------------
# It's as simple as few lines
for image in image_list:
    x, orig_img = process_img(image)
    ids, scores, bboxes = [xx[0].asnumpy() for xx in net(x)]
    ax = gv.utils.viz.plot_bbox(orig_img, bboxes, scores, ids,
                                class_names=net.classes, ax=ax)
    plt.show()
