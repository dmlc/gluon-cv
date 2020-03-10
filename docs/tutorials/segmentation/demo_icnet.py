"""7. Test with ICNet Pre-trained Models for Multi-Human Parsing
======================================

This is a quick demo of using GluonCV ICNet model for multi-human parsing on real-world images.
Please follow the `installation guide <../index.html>`_ to install MXNet and GluonCV if not yet.
"""
import mxnet as mx
from mxnet import image
from mxnet.gluon.data.vision import transforms
import gluoncv
# using cpu
ctx = mx.cpu(0)


##############################################################################
# Prepare the image
# -----------------
#
# download the example image
url = 'https://github.com/KuangHaofei/GluonCV_Test/blob/master/' + \
      'mhp_v1/demo%20images/1528.jpg?raw=true'
filename = 'mhp_v1_example.jpg'
gluoncv.utils.download(url, filename, True)


##############################################################################
# load the image
img = image.imread(filename)

from matplotlib import pyplot as plt
plt.imshow(img.asnumpy())
plt.show()

##############################################################################
# normalize the image using dataset mean
from gluoncv.data.transforms.presets.segmentation import test_transform
img = test_transform(img, ctx)

##############################################################################
# Load the pre-trained model and make prediction
# ----------------------------------------------
#
# get pre-trained model
model = gluoncv.model_zoo.get_model('icnet_resnet50_mhpv1', pretrained=True)

##############################################################################
# make prediction using single scale
output = model.predict(img)
predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()

##############################################################################
# Add color pallete for visualization
from gluoncv.utils.viz import get_color_pallete
import matplotlib.image as mpimg
mask = get_color_pallete(predict, 'mhpv1')
mask.save('output.png')

##############################################################################
# show the predicted mask
mmask = mpimg.imread('output.png')
plt.imshow(mmask)
plt.show()
