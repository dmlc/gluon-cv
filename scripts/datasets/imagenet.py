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
through the steps of preparing this dataset to be used by GluonVision.

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

   python scripts/datasets/imagenet.py --download-dir ~/ILSVRC2012

.. note::

   Extracting the images may take tens of minutes to a few hours. E.g., it takes
   about 30min on AWS EBS General Purpose SSD (gp2).

In default it will extract the images into ``~/.mxnet/datasets/imagenet``. You
can specify a different place by using ``--target-dir``.


How to Read the Dataset
-----------------------

The prepared dataset can be loaded by :py:class:`gluonvision.data.ImageNet`
directly. Here is an example that randomly reads 128 images each time and
performs randomized resizing and cropping.

.. code-block:: python

   from gluonvision.data import ImageNet
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

   for x, y in train_data:
       print((x.shape, y.shape))
       break

The outputs of the above example will be::

   ((128, 3, 224, 224), (128,))

Dive Deep into Source Codes
---------------------------

The main job this script does is extract the images in the tar files into a
target directory. The training and validation images are stored in folders
``train`` and ``val``, respectively. The images belong to a same class are
stored in the same sub-folder.

"""

import os
import argparse
import tarfile
import pickle
from tqdm import tqdm
from mxnet.gluon.utils import check_sha1

_TARGET_DIR = os.path.expanduser('~/.mxnet/datasets/imagenet')
_TRAIN_TAR = 'ILSVRC2012_img_train.tar'
_TRAIN_TAR_SHA1 = '43eda4fe35c1705d6606a6a7a633bc965d194284'
_VAL_TAR = 'ILSVRC2012_img_val.tar'
_VAL_TAR_SHA1 = '5f3f73da3395154b60528b2b2a2caf2374f5f178'

def parse_args():
    parser = argparse.ArgumentParser(
        description='Setup the ImageNet dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--download-dir', required=True,
                        help="The directory that contains downloaded tar files")
    parser.add_argument('--target-dir', default=_TARGET_DIR,
                        help="The directory to store extracted images")
    parser.add_argument('--checksum', action='store_true',
                        help="If check integrity before extracting.")
    args = parser.parse_args()
    return args

def check_file(filename, checksum, sha1):
    if not os.path.exists(filename):
        raise ValueError('File not found: '+filename)
    if checksum and not check_sha1(filename, sha1):
        raise ValueError('Corrupted file: '+filename)

def extract_train(tar_fname, target_dir):
    os.makedirs(target_dir)
    with tarfile.open(tar_fname) as tar:
        print("Extracting "+tar_fname+"...")
        # extract each class one-by-one
        pbar = tqdm(total=len(tar.getnames()))
        for class_tar in tar:
            pbar.set_description('Extract '+class_tar.name)
            tar.extract(class_tar, target_dir)
            class_fname = os.path.join(target_dir, class_tar.name)
            class_dir = os.path.splitext(class_fname)[0]
            os.mkdir(class_dir)
            with tarfile.open(class_fname) as f:
                f.extractall(class_dir)
            os.remove(class_fname)
            pbar.update(1)
        pbar.close()

def extract_val(tar_fname, target_dir):
    os.makedirs(target_dir)
    print('Extracting ' + tar_fname)
    with tarfile.open(tar_fname) as tar:
        tar.extractall(target_dir)
    # move images to proper subfolders
    val_maps_file = os.path.join(os.path.dirname(__file__), 'imagenet_val_maps.pkl')
    with open(val_maps_file, 'rb') as f:
        dirs, mappings = pickle.load(f)
    for d in dirs:
        os.makedirs(os.path.join(target_dir, d))
    for m in mappings:
        os.rename(os.path.join(target_dir, m[0]), os.path.join(target_dir, m[1], m[0]))

def main():
    args = parse_args()

    target_dir = os.path.expanduser(args.target_dir)
    if os.path.exists(target_dir):
        raise ValueError('Target dir ['+target_dir+'] exists. Remove it first')

    tar_dir = os.path.expanduser(args.tar_dir)
    train_tar_fname = os.path.join(tar_dir, _TRAIN_TAR)
    check_file(train_tar_fname, args.checksum, _TRAIN_TAR_SHA1)
    val_tar_fname = os.path.join(tar_dir, _VAL_TAR)
    check_file(val_tar_fname, args.checksum, _VAL_TAR_SHA1)

    extract_train(train_tar_fname, os.path.join(target_dir, 'train'))
    extract_val(val_tar_fname, os.path.join(target_dir, 'val'))

if __name__ == '__main__':
    main()
