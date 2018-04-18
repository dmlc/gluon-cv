"""Getting started with SSD pre-trained models
==============================================

This article is an introductory tutorial to play with pre-trained models
with several lines of code.

.. image:: https://github.com/dmlc/web-data/blob/master/gluonvision/detection/street_small.jpg?raw=true
"""

import mxnet as mx
import gluonvision as gv
from matplotlib import pyplot as plt

######################################################################
# Obtain a pretrained model
# -------------------------
# Try grab a 300x300 model trained on Pascal voc dataset.
# it will automatically download from s3 servers if not exists
# net = gv.model_zoo.get_model('ssd_512_resnet50_v1_voc', pretrained=True)

######################################################################
# Pre-process image
# -----------------
#
# A raw image must be converted to tensor before inference.
from gluonvision.data.transforms import presets

# a demo image, feel free to use your own image
gv.utils.download('https://github.com/dmlc/web-data/blob/master/' +
                  'gluonvision/detection/street_small.jpg?raw=true', 'street.jpg')
image_name = 'street.jpg'

# gluonvision support SSD models to accept arbitrary data shape
# so this size is not that restrictive
x, img = presets.ssd.load_test(image_name, short=512)
print('shape of x:', x.shape)

######################################################################
# Inference and display
# ---------------------

#ids, scores, bboxes = [xx[0].asnumpy() for xx in net(x)]
#from gluonvision.utils import viz
#ax = viz.plot_bbox(img, bboxes, scores, ids, class_names=net.classes, ax=None)
#plt.show()

######################################################################
# Play with complete python script for demo
# -----------------------------------------
#
# :download:`Download Full Python Script demo_ssd.py<../../scripts/detection/ssd/demo_ssd.py>`
#
# Example usage:
#
# .. code-block:: python
#
#     python demo_ssd.py --gpus 0 --network ssd_300_vgg16_atrous_voc --data-shape 300
#     # you can use models on disk as well
#     python demo_ssd.py --gpus 0 --network ssd_300_vgg16_atrous_voc --data-shape 300 --pretrained ./ssd_300.params


######################################################################
# You can also download this tutorial
# -----------------------------------
