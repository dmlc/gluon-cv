"""7. Test with ICNet Pre-trained Models for Multi-Human Parsing
======================================

This is a quick demo of using GluonCV ICNet model for multi-human parsing on real-world images.
Please follow the `installation guide <../../index.html#installation>`__
to install MXNet and GluonCV if not yet.
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
# Let's first download the example image,

url = 'https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/segmentation/mhpv1_examples/1.jpg'
filename = 'mhp_v1_example.jpg'
gluoncv.utils.download(url, filename, True)


##############################################################################
# Then we load the image and visualize it,

img = image.imread(filename)

from matplotlib import pyplot as plt
plt.imshow(img.asnumpy())
plt.show()

##############################################################################
# We normalize the image using dataset mean and standard deviation,

from gluoncv.data.transforms.presets.segmentation import test_transform
img = test_transform(img, ctx)

##############################################################################
# Load the pre-trained model and make prediction
# ----------------------------------------------
#
# Next, we get a pre-trained model from our model zoo,

model = gluoncv.model_zoo.get_model('icnet_resnet50_mhpv1', pretrained=True)

##############################################################################
# We directly make semantic predictions on the image,

output = model.predict(img)
predict = mx.nd.squeeze(mx.nd.argmax(output, 1)).asnumpy()

##############################################################################
# In the end, we add color pallete for visualizing the predicted mask,

from gluoncv.utils.viz import get_color_pallete
import matplotlib.image as mpimg
mask = get_color_pallete(predict, 'mhpv1')
mask.save('output.png')
mmask = mpimg.imread('output.png')
plt.imshow(mmask)
plt.show()
