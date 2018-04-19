"""Predict with pre-trained SSD models
===================================

This article goes through how to play with pre-trained SSD models with several
lines of code.

"""

from gluonvision import model_zoo, data, utils
from matplotlib import pyplot as plt

######################################################################
# Obtain a pretrained model
# -------------------------
#
# Let's get a SSD model that is trained with 512x512 images on the Pascal VOC
# dataset with ResNet-50 V1 as the base model. By specifying
# ``pretrained=True``, it will automatically download the model from the model
# zoo if necessary. For more pretrained models, refer to
# :doc:`../../model_zoo/index`.

net = model_zoo.get_model('ssd_512_resnet50_v1_voc', pretrained=True)

######################################################################
# Pre-process an image
# --------------------
#
# Next we download an image, and pre-process with preset data transforms. Here we
# specify that we resize the short edge of the image into 512 px. But you can
# feed an arbitrary size image.

im_fname = utils.download('https://github.com/dmlc/web-data/blob/master/' +
                          'gluonvision/detection/street_small.jpg?raw=true')
x, img = data.transforms.presets.ssd.load_test(im_fname, short=512)
plt.imshow(img)
plt.show()
print('Shape of pre-processed image:', x.shape)

######################################################################
# Inference and display
# ---------------------
#
#

class_IDs, scores, bounding_boxs = net(x)

ax = utils.viz.plot_bbox(img, bounding_boxs.asnumpy(), scores.asnumpy(),
                         class_IDs.asnumpy(), class_names=net.classes, ax=None)
plt.show()
