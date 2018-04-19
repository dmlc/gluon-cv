"""Training Your First Classification Model on CIFAR10
===================================================


`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`__ is a
labeled dataset of tiny (32x32) images, collected by Alex Krizhevsky,
Vinod Nair, and Geoffrey Hinton. It is widely used as a benchmark in
computer vision research.

|image-cifar10|

In this tutorial, we will demonstrate how to use ``Gluon`` to train a
model from scratch and reproduce the performance from papers.

In the following content, we will demonstrate how to

-  How well can our model predict
-  train a model

Demo and Benchmark
------------------

Before busying with training and parameter tuning, you may want to get
an idea of what the result may look like.

Here

Let's first play with a pre-trained cifar model with a few lines of python code.

First, please follow the `installation guide <../index.html>`__
to install MXNet and GluonVision if not yetself.
"""

import mxnet as mx
import matplotlib.pyplot as plt

from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms
from gluonvision import utils
from gluonvision.model_zoo import get_model

################################################################
# Prepare the image
# -----------------
#
# Download the example image:

url = 'https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/classification/plane-draw.jpeg'
im_fname = utils.download(url)

with open(im_fname, 'rb') as f:
    img = image.imdecode(f.read())

plt.imshow(img.asnumpy())
plt.show()

################################################################
# In case you cannot recognize it, the image is a poorly-drawn plane :)
#
# Now we can define transformation for the image.

transform_fn = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

################################################################
# The transformation function does two jobs: resize the image to 32x32 and
# normalize the image. How's the normalized image look like?
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
# |image1|
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
# To experience a more real world friendly demo, please checkout models
# trained on `ImageNet <demo_imagenet.html>`__.
#
# Train Your First Model
# ----------------------
#
# In the demo, we have used a pretrained model. So how did we train it?
#
# :download:`Download Python Script train_cifar10.py<../../../scripts/classification/cifar/train_cifar10.py>`
#
# We trained the models with ``train_cifar10.py``.
# It takes a bunch of parameters to control the model training process. To
# start, you can try the following command:
#
# ::
#
#     python train_cifar10.py --num-epochs 240 --mode hybrid --num-gpus 1 -j 8 --batch-size 128\
#         --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 80,160 --model cifar_resnet20_v1
#
# This command trains a ``ResNet20_V2`` model for 240 epochs on one GPU.
# The batch size is 128. We decay the learning rate by a factor of 10 at the 80-th and 160-th epoch.
# The script prints information for each epoch so that we can have a sense
# of the progress and watchout for any unexpected issues.
#
# ::
#
#     INFO:root:[Epoch 0] train=0.367448 val=0.460800 loss=1.735358 time: 12.739688
#     INFO:root:[Epoch 1] train=0.492027 val=0.524600 loss=1.409553 time: 12.500988
#     INFO:root:[Epoch 2] train=0.556891 val=0.640400 loss=1.241357 time: 12.994388
#     INFO:root:[Epoch 3] train=0.595152 val=0.658900 loss=1.145049 time: 12.342926
#     INFO:root:[Epoch 4] train=0.620733 val=0.680900 loss=1.075090 time: 13.098537
#     INFO:root:[Epoch 5] train=0.640966 val=0.700200 loss=1.017329 time: 12.360461
#     ...
#
# The dataset and the model are relatively small, thus it won’t take you
# too long to train the model. If you don't have a GPU yet, you can still try to
# train with your CPU with MKLDNN. One can install MXNet with MKLDNN with
#
#
# ::
#
#     pip install --pre mxnet-mkl
#
#
# After the installation, one can run the following command:
#
# ::
#
#     python train_cifar10.py --num-epochs 240 --mode hybrid --num-gpus 0 -j 1 --batch-size 128\
#         --wd 0.0001 --lr 0.1 --lr-decay 0.1 --lr-decay-epoch 80,160 --model cifar_resnet20_v1
#
# Here we change the values of ``--num-gpus`` to 0 and ``-j`` to 1, to only use CPU for training and use one thread
# for data loader.
#
# This is a brief comparison of performance on our side :
#
# -  13 seconds with one V100 GPU per epoch, and 8 CPU threads.
# -  70 seconds with one 8-thread CPU per epoch, with MKLDNN enabled.
#
# With limited computational power, it is good in practice to firstly test
# a few epochs to ensure everything works, then leave it running for a
# night, and wake up to see the result :)
#
# After the training, the accuracy is expect to be around 91%. To get a
# better accuracy, we can train a ``ResNet110_V2`` model instead by
# ``--model cifar_resnet110_v2``, at the cost of around 4 times of the
# training time. With ``ResNet110_V2``, we expect the accuracy to be
# around 94%.
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
# Or, if you would like to try a more powerful demo, i.e. models trained
# on ImageNet, please read `Getting Started with ImageNet Pre-trained Models <demo_imagenet.html>`__.
#
# .. |image-cifar10| image:: https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/classification/cifar10.png
# .. |image0| image:: https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/classification/plane-draw.jpeg
# .. |image1| image:: https://raw.githubusercontent.com/dmlc/web-data/master/gluonvision/classification/mt_baker.jpg
