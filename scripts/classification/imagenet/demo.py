"""Training Your First Classification Model on ImageNet
===================================================

```ImageNet`` <http://www.image-net.org/>`__ is a
large labeled dataset of real-world images. It is the most
well-known dataset for computer vision tasks.

In this tutorial, we will demonstrate how a well-trained model
classifies real life images.

Specifically, we offer a script to load a pretrained ``ResNet50_v2`` model, and
a list of models we haveself.

Different from ``CIFAR10``, training a model on ImageNet is much more difficult.
We will have more discussions on ImageNet in other tutorials.

Demo and Benchmark
------------------

A model trained on ImageNet can classify images into 1000 classes, this makes it
much more powerful than the one we showed in ```CIFAR10`` tutorial.

With this script, you can load a pre-trained model and predict on any image.

Let's take the photo of Mt. Baker again as the first photo.

|image0|

We can make prediction by

::

    python demo.py --model ResNet50_v2 --input-pic mt_baker.jpg

And the model thinks that

::

    The input picture is classified to be
    	[volcano], with probability 0.558.
    	[alp], with probability 0.398.
    	[valley], with probability 0.018.
    	[lakeside], with probability 0.006.
    	[mountain_tent], with probability 0.006.

This time it does a perfect job. Note that we have listed the top five
possible classes, because with 1000 classes the model may not always rate the
correct answer with the highest rank. Besides the top-1 prediction, we also
consider top-5 accuracy as a measurement of how well a model can predict.

Model Zoo
---------

We train various models and store them on cloud as a “zoo of the
models”. Users can pick the model with regards to the accuracy and model
complexity. Please check this
````page`` <https://mxnet.incubator.apache.org/api/python/gluon/model_zoo.html>`__
for a complete list of models.

Here’s a table for the pre-trained models in the ``ResNet`` family:

+--------------+--------+--------+
| Model        | Top-1  | Top-5  |
+==============+========+========+
| resnet18_v1  | 0.6803 | 0.8818 |
+--------------+--------+--------+
| resnet34_v1  | 0.7202 | 0.9066 |
+--------------+--------+--------+
| resnet50_v1  | 0.7540 | 0.9266 |
+--------------+--------+--------+
| resnet101_v1 | 0.7693 | 0.9334 |
+--------------+--------+--------+
| resnet152_v1 | 0.7727 | 0.9353 |
+--------------+--------+--------+
| resnet18_v2  | 0.6961 | 0.8901 |
+--------------+--------+--------+
| resnet34_v2  | 0.7324 | 0.9125 |
+--------------+--------+--------+
| resnet50_v2  | 0.7622 | 0.9297 |
+--------------+--------+--------+
| resnet101_v2 | 0.7747 | 0.9375 |
+--------------+--------+--------+
| resnet152_v2 | 0.7833 | 0.9409 |
+--------------+--------+--------+

Next Step
---------

Congratulations! You’ve just finished reading our first tutorial. We
have a lot more others to help you learn more and get familiar with
``gluonvision``.

If you would like to dig deeper in the topic of ``CIFAR10`` training,
feel free to read `the next tutorial on ``CIFAR10`` <>`__.

Or, if you would like to try a more powerful demo, i.e. models trained
on ImageNet, please read `xxx <>`__.

.. |image0| image:: mt_baker.jpeg

"""
from __future__ import division

import argparse, time, logging, random, math

import numpy as np
import mxnet as mx
import matplotlib.pyplot as plt

from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms

from gluonvision.model_zoo import get_model

parser = argparse.ArgumentParser(description='Predict ImageNet classes from a given image')
parser.add_argument('--model', type=str, required=True,
                    help='name of the model to use')
parser.add_argument('--saved-params', type=str, default='',
                    help='path to the saved model parameters')
parser.add_argument('--input-pic', type=str, required=True,
                    help='path to the input picture')
opt = parser.parse_args()

classes = 1000
with open('imagenet_labels.txt', 'r') as f:
    class_names = [l.strip('\n') for l in f.readlines()]

context = [mx.cpu()]

# Load Model
model_name = opt.model
pretrained = True if opt.saved_params == '' else False
kwargs = {'classes': classes, 'pretrained': pretrained}
net = get_model(model_name, **kwargs)

if not pretrained:
    net.load_params(opt.saved_params, ctx = context)

# Load Images
with open(opt.input_pic, 'rb') as f:
    img = image.imdecode(f.read())

# Transform
transform_fn = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img_transformed = nd.zeros((1, 3, 224, 224))
img_transformed[0,:,:,:] = transform_fn(img)
pred = net(img_transformed)

topK = 5
ind = nd.topk(pred, k=topK)[0].astype('int')
print('The input picture is classified to be')
for i in range(topK):
    print('\t[%s], with probability %.3f.'%
          (class_names[ind[i].asscalar()], nd.softmax(pred)[0][ind[i]].asscalar()))
