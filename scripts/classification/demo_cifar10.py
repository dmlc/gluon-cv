from __future__ import division

import argparse, time, logging, random, math

import numpy as np
import mxnet as mx
import matplotlib.pyplot as plt

from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms

from gluonvision.model_zoo import get_model

parser = argparse.ArgumentParser(description='Predict CIFAR10 classes from a given image')
parser.add_argument('--model', type=str, required=True,
                    help='name of the model to use')
parser.add_argument('--pretrained', action='store_true',
                    help='use the official pretrained model')
parser.add_argument('--saved-params', type=str, default='',
                    help='path to the saved model parameters')
parser.add_argument('--input-pic', type=str, required=True,
                    help='path to the input picture')
parser.add_argument('--num-gpus', type=int, default=0,
                    help='number of GPUs to predict')
parser.add_argument('--width-factor', type=float, default=1,
                    help='width factor for wide resnet. default is 1.')
parser.add_argument('--drop-rate', type=float, default=0,
                    help='drop rate for wide resnet. default is 0.')
opt = parser.parse_args()

classes = 10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

num_gpus = opt.num_gpus
context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]

# Load Model
model_name = opt.model
pretrained = True if opt.pretrained else False
if model_name.startswith('cifar_wideresnet'):
    kwargs = {'classes': classes,
              'drop_rate': opt.drop_rate, 'width_factor': opt.width_factor}
else:
    kwargs = {'classes': classes}
net = get_model(model_name, **kwargs)

if opt.saved_params is not None:
    net.load_params(opt.saved_params, ctx = context)

# Load Images
with open(opt.input_pic, 'rb') as f:
    img = image.imdecode(f.read())

# Transform
transform_fn = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

img_transformed = nd.zeros((1, 3, 32, 32))
img_transformed[0,:,:,:] = transform_fn(img)
pred = net(img_transformed)

ind = nd.argmax(pred, axis=1).astype('int')
print('The input picture is classified to be [%s], with probability %.3f.'%
      (class_names[ind.asscalar()], nd.softmax(pred)[0][ind].asscalar()))
