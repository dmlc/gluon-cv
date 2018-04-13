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
   (Solid-state disks) are prefered over HDD because of the better performance
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

Assume the two tar files are downloaded in the folder ``~/ILSVRC2012_tar``, and
we plan to store the extracted images in ``~/ILSVRC2012``. We can use the
following command to do it automatically.

.. code-block:: bash

   python scripts/datasets/setup_imagenet.py \
       --download-dir ~/ILSVRC2012_tar --path ~/ILSVRC2012

.. note::

   Extracting the images may take tens of minutes to a few hours. E.g., it takes
   about 30min on AWS EBS General Purpose SSD (gp2).

If you already extracted the download tar files, you only need to specify
``--path``. For example, assume all images are extracted in ``~/ILSVRC2012``,
then you can simply run the following command to prepare the dataset.

.. code-block:: bash

   python scripts/datasets/setup_imagenet.py --path ~/ILSVRC2012

Dive Deep into Source Codes
---------------------------

"""

import os
import argparse
import tarfile
from tqdm import tqdm
from gluonvision.utils import makedirs
from mxnet.gluon.utils import check_sha1

_TARGET_DIR = os.path.expanduser('~/.mxnet/datasets/imagenet')

def parse_args():
    parser = argparse.ArgumentParser(
        description='Initialize imagenet dataset.',
        epilog='Example: python setup_imagenet.py ~/datasets/VOCdevkit --download',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', required=True, help='dataset directory on disk')
    parser.add_argument('--download-dir', type=str, default='',
                        help='directory of downloaded *.tar files')
    parser.add_argument('--check-sha1', action='store_true',
                        help='check sha1 checksum before untar, since there is '
                        'risk of corrupted files. Use with cautious because this can be very slow.')
    args = parser.parse_args()
    return args

def untar_imagenet_train(filename, target_dir, checksum=False):
    train_dir = os.path.join(os.path.expanduser(target_dir), 'train')
    if checksum:
        if not check_sha1(filename, '43eda4fe35c1705d6606a6a7a633bc965d194284'):
            raise RuntimeError("Corrupted training file: {}".format(filename))

    makedirs(train_dir)
    with tarfile.open(filename) as tar:
        print("Extracting train data, this may take very long time depending on I/O perf.")
        files = tar.getnames()
        assert len(files) == 1000
        with tqdm(total=1000) as pbar:
            for obj in tar:
                if not obj:
                    break
                tar.extract(obj, train_dir)
                iname = os.path.join(train_dir, obj.name)
                with tarfile.open(iname) as itar:
                    dst_dir = os.path.splitext(iname)[0]
                    makedirs(dst_dir)
                    itar.extractall(dst_dir)
                # remove separate tarfiles
                os.remove(iname)
                pbar.update(1)

def untar_imagenet_val(filename, target_dir, checksum=False):
    val_dir = os.path.join(os.path.expanduser(target_dir), 'val')
    if checksum:
        if not check_sha1(filename, '5f3f73da3395154b60528b2b2a2caf2374f5f178'):
            raise RuntimeError("Corrupted training file: {}".format(filename))

    makedirs(val_dir)
    with tarfile.open(filename) as tar:
        tar.extractall(val_dir)

def symlink_val(val_dir, dst_dir):
    """Symlink individual validation images to 1000 directories."""
    import pickle
    val_maps_file = os.path.join(os.path.dirname(__file__), 'imagenet_val_maps.pkl')
    with open(val_maps_file, 'rb') as f:
        dirs, mappings = pickle.load(f)
    assert len(dirs) == 1000, "Require 1000 dir names"
    assert len(mappings) == 50000, "Require 50000 image->dir mappings"

    val_dir = os.path.expanduser(val_dir)
    dst_dir = os.path.expanduser(dst_dir)
    for d in dirs:
        makedirs(os.path.join(dst_dir, d))

    for m in mappings:
        os.symlink(os.path.join(val_dir, m[0]), os.path.join(dst_dir, m[1], m[0]))

if __name__ == '__main__':
    args = parse_args()
    path = os.path.expanduser(args.path)
    if not os.path.isdir(path) or not os.path.isdir(os.path.join(path, 'train')) \
        or not os.path.isdir(os.path.join(path, 'train', 'n04429376')) \
        or not os.path.isdir(os.path.join(path, 'val')):
        if not args.download_dir:
            raise ValueError(('{} is not a valid ImageNet directory, make sure it is present.'
                              ' Or you can point "--download-dir" to tar files'.format(path)))
        else:
            train_tarfile = os.path.join(args.download_dir, 'ILSVRC2012_img_train.tar')
            untar_imagenet_train(train_tarfile, path, args.check_sha1)
            val_tarfile = os.path.join(args.download_dir, 'ILSVRC2012_img_val.tar')
            untar_imagenet_val(val_tarfile, path, args.check_sha1)

    # make symlink
    if os.path.isdir(_TARGET_DIR):
        os.remove(_TARGET_DIR)
    makedirs(_TARGET_DIR)
    os.symlink(os.path.join(args.path, 'train'), os.path.join(_TARGET_DIR, 'train'))
    symlink_val(os.path.join(args.path, 'val'), os.path.join(_TARGET_DIR, 'val'))
