"""Getting Started with Pre-trained Model on CIFAR10
===================================================

`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`__ is a
labeled dataset of tiny (32x32) images, collected by Alex Krizhevsky,
Vinod Nair, and Geoffrey Hinton. It is widely used as a benchmark in
computer vision research.

|image-cifar10|

.. |image-cifar10| image:: https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/datasets/cifar10.png

In this tutorial, we will demonstrate how to load a pre-trained model in``Gluon``,
and classify images from the Internet or your local disk.

Step by Step
------------------

Let's first play with a pre-trained cifar model with a few lines of python code.

First, please follow the `installation guide <../index.html>`__
to install MXNet and GluonVision if not yet.
"""

import mxnet as mx
import matplotlib.pyplot as plt

from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms
from gluonvision import utils
from gluonvision.model_zoo import get_model

################################################################
#
# Download the example image:

url = 'https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/classification/plane-draw.jpeg'
im_fname = utils.download(url)

with open(im_fname, 'rb') as f:
    img = image.imdecode(f.read())

plt.imshow(img.asnumpy())
plt.show()

################################################################
# In case you cannot recognize it, the image is a poorly-drawn airplane :)
#
# Now we can define transformation for the image.

transform_fn = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

################################################################
# The transformation function does three things:
# resize the image to 32x32, transpose the image to Channel*Height*Width,
# and normalize the image with CIFAR10 parameters.
#
# How's the transformed image look like?

img = transform_fn(img)
plt.imshow(nd.transpose(img, (1,2,0)).asnumpy())
plt.show()

################################################################
# You can't recognize anything? Neither do I. Don't panic!
# The transformation makes it more "model-friendly", instead of "human-friendly".
#
# Next, we load a pre-trained model.

net = get_model('cifar_resnet110_v2', classes=10, pretrained=True)

################################################################
#
# Finally, we prepare the image and feed it to the model

img_transformed = nd.zeros((1, 3, 32, 32))
img_transformed[0,:,:,:] = img
pred = net(img_transformed)

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
ind = nd.argmax(pred, axis=1).astype('int')
print('The input picture is classified to be [%s], with probability %.3f.'%
      (class_names[ind.asscalar()], nd.softmax(pred)[0][ind].asscalar()))

################################################################
# Play with the scripts
# ---------------------
#
# We provide you a script to load a pre-trained model from us and
# predict on any image on your disk.
#
# :download:`Download Python Script demo_cifar10.py<../../../scripts/classification/cifar/demo_cifar10.py>`
#
# Feel free to feed in your own image to see how well it does the job.
# Keep in mind that ``CIFAR10`` is relatively small and has only 10
# classes. Models trained on ``CIFAR10`` only recognize objects from those
# 10 classes, therefore it may surprise you if we feed one image to the model
# which doesn't belong to any of the 10 classes
#
# For instance we have the following photo of Mt. Baker:
#
# |image-mtbaker|
#
# ::
#
#     python demo_cifar10.py --model cifar_resnet110_v2 --input-pic mt_baker.jpg
#
# The result is:
#
# ::
#
#     The input picture is classified to be [airplane], with probability 0.857.
#
# Next Step
# ---------
#
# Congratulations! You’ve just finished reading our first tutorial. We
# have a lot more others to help you learn more and get familiar with
# ``gluonvision``.
#
# If you would like to dig deeper in the topic of ``CIFAR10`` training,
# feel free to read `the next tutorial on CIFAR10 <dive_deep_cifar10.html>`__.
#
# Or, if you would like to try a more real-world-friendly demo, i.e. models trained
# on ImageNet, please read `Getting Started with ImageNet Pre-trained Models <demo_imagenet.html>`__.
#
# .. |image-mtbaker| image:: https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/classification/mt_baker.jpg
