"""Prepare the ImageNet dataset
============================

The `ImageNet <http://www.image-net.org/>`_ project contains millions of images
and thounds of objects for image classification. It is widely used in the
research community to demonstrate if new proposed models are be able to achieve
the state-of-the-art results.

.. image:: https://www.fanyeong.com/wp-content/uploads/2018/01/v2-718f95df083b2d715ee29b018d9eb5c2_r.jpg
   :width: 500 px

The dataset are multiple versions. The commonly used one for image
classification is the dataset provided in `ILSVRC 2012
<http://www.image-net.org/challenges/LSVRC/2012/>`_. This tutorial will go
through the steps of preparing this dataset to be used by GluonCV.

.. note::

   You need at least 300 GB disk space to download and extract the dataset. SSD
   (Solid-state disks) is prefered over HDD because of the better performance
   on reading and writing small objects (images).

Download the Dataset
--------------------

First to go to the `download page <http://www.image-net.org/download-images>`_
(you may need to register an account), and then find the download link for
ILSVRC2012. Next go to the download page to download the following two files:

======================== ======
Filename                 Size
======================== ======
ILSVRC2012_img_train.tar 138 GB
ILSVRC2012_img_val.tar   6.3 GB
======================== ======

Preprocess the Dataset
----------------------

Assume the two tar files are downloaded in the folder ``~/ILSVRC2012``. We can use the
following command to prepare the dataset automatically.

.. code-block:: bash

   # assume your current working directory is git cloned gluon-cv
   python scripts/datasets/imagenet.py --download-dir ~/ILSVRC2012

.. hint::

   You can download the python script :download:`imagenet.py<../../../scripts/datasets/imagenet.py>`
   and subdirectory information for 50000 validation images: :download:`imagenet_val_maps.pklz<../../../scripts/datasets/imagenet_val_maps.pklz>`.
   Put the pklz file in the same directory as the python script.

.. note::

   Extracting the images may take tens of minutes to a few hours. E.g., it takes
   about 30min on AWS EBS General Purpose SSD (gp2).

In default it will extract the images into ``~/.mxnet/datasets/imagenet``. You
can specify a different place by using ``--target-dir``.


How to Read the Dataset
-----------------------

The prepared dataset can be loaded by :py:class:`gluoncv.data.ImageNet`
directly. Here is an example that randomly reads 128 images each time and
performs randomized resizing and cropping.
"""


from gluoncv.data import ImageNet
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import transforms

train_trans = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor()
])

# You need to specify ``root`` for ImageNet if you extracted the images into
# a different folder
train_data = DataLoader(
    ImageNet(train=True).transform_first(train_trans),
    batch_size=128, shuffle=True)

#########################################################################
for x, y in train_data:
    print((x.shape, y.shape))
    break


#########################################################################
# Peek some validation images
from gluoncv.utils import viz
val_dataset = ImageNet(train=False)
viz.plot_image(val_dataset[1234][0])  # index 0 is image, 1 is label
viz.plot_image(val_dataset[4567][0])
