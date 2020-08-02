"""01. Predict depth from a single image with pre-trained Monodepth2 models
===========================================================================

This is a quick demo of using GluonCV Monodepth2 model for KITTI on real-world images.
Please follow the `installation guide <../../index.html#installation>`__
to install MXNet and GluonCV if not yet.
"""
import numpy as np

import mxnet as mx
from mxnet.gluon.data.vision import transforms
import gluoncv
# using cpu
ctx = mx.cpu(0)


##############################################################################
# Prepare the image
# -----------------
#
# Let's first download the example image,

url = 'https://raw.githubusercontent.com/KuangHaofei/GluonCV_Test/master/monodepthv2/tutorials/test_img.png'
filename = 'test_img.png'
gluoncv.utils.download(url, filename, True)


##############################################################################
# Then we load the image and visualize it,

import PIL.Image as pil
img = pil.open(filename).convert('RGB')

from matplotlib import pyplot as plt
plt.imshow(img)
plt.show()

##############################################################################
# We resize the image make it has the same input size with pretrained model,
# and transfer the image to NDArray,

original_width, original_height = img.size
feed_height = 192
feed_width = 640

img = img.resize((feed_width, feed_height), pil.LANCZOS)
img = transforms.ToTensor()(mx.nd.array(img)).expand_dims(0).as_in_context(context=ctx)

##############################################################################
# Load the pre-trained model and make prediction
# ----------------------------------------------
#
# Next, we get a pre-trained model from our model zoo,

model = gluoncv.model_zoo.get_model('monodepth2_resnet18_kitti_stereo_640x192',
                                    pretrained_base=False, ctx=ctx, pretrained=True)

##############################################################################
# We directly make disparity map predictions on the image, and resize it to input size

outputs = model.predict(img)
disp = outputs[("disp", 0)]
disp_resized = mx.nd.contrib.BilinearResize2D(disp, height=original_height, width=original_width)

##############################################################################
# In the end, we add normalized color map for visualizing the predicted disparity map,

import matplotlib as mpl
import matplotlib.cm as cm
disp_resized_np = disp_resized.squeeze().as_in_context(mx.cpu()).asnumpy()
vmax = np.percentile(disp_resized_np, 95)
normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
im = pil.fromarray(colormapped_im)
im.save('test_output.png')

import matplotlib.image as mpimg
disp_map = mpimg.imread('test_output.png')
plt.imshow(disp_map)
plt.show()
