"""Prepare PASCAL VOC datasets
==============================

`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ contains a collection of
datasets for object detection. The most commonly adopted version for
benchmarking is using *2007 trainval* and *2012 trainval* for training and *2007
test* for validation.  This tutorial will walk you through the steps for
preparing this dataset to be used by GluonVision.

.. image:: http://host.robots.ox.ac.uk/pascal/VOC/pascal2.png

Prepare the dataset
-------------------

The easiest way is simply running this script, which will automatically download
and extract the data into ``~/.mxnet/datasets/voc``.


.. code-block:: bash

    python scripts/datasets/pascal_voc.py

.. note::

   You need 8.4 GB disk space to download and extract this dataset. SSD is
   preferred over HDD because of its better performance.

.. note::

   The total time to prepare the dataset depends on your Internet speed and disk
   performance. For example, it often takes 10min on AWS EC2 with EBS.

If you have already downloaded the following required files, whose URLs can be
obtained from the source codes at the end of this tutorial,

===========================  ======
Filename                     Size
===========================  ======
VOCtrainval_06-Nov-2007.tar  439 MB
VOCtest_06-Nov-2007.tar      430 MB
VOCtrainval_11-May-2012.tar  1.9 GB
benchmark.tgz                1.4 GB
===========================  ======

then you can specify the folder name through ``--download-dir`` to avoid
download them again.  For example,

.. code-block:: python

   python scripts/datasets/pascal_voc.py --download-dir ~/voc_downloads

How to load the dataset
-----------------------

Load image and label from Pascal VOC is quite straight-forward

.. code:: python

    from gluonvision.data import VOCDetection
    train_dataset = VOCDetection(splits=[(2007, 'trainval'), (2012, 'trainval')])
    val_dataset = VOCDetection(splits=[(2007, 'test')])
    print('Training images:', len(train_dataset))
    print('Validation images:', len(val_dataset))

Output::

    Training images: 16551
    Validation images: 4952

Check the first example:

.. code:: python

    train_image, train_label = train_dataset[0]
    bboxes = train_label[:, :4]
    cids = train_label[:, 4:5]
    print('image size:', train_image.shape)
    print('bboxes:', bboxes.shape, 'class ids:', cids.shape)

Output::

    image size: (375, 500, 3)
    bboxes: (5, 4) class ids: (5, 1)

Dive deep into source codes
---------------------------

The implementation of pascal_voc.py is straightforward. It simply downloads and
extract the data.
"""
import os
import shutil
import argparse
import tarfile
from gluonvision.utils import download, makedirs

_TARGET_DIR = os.path.expanduser('~/.mxnet/datasets/voc')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Initialize PASCAL VOC dataset.',
        epilog='Example: python setup_pascal_voc.py ~/datasets/VOCdevkit --download',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', required=True, help='dataset directory on disk')
    parser.add_argument('--download', action='store_true', help='try download if set')
    parser.add_argument('--overwrite', action='store_true', help='overwrite downloaded if set')
    args = parser.parse_args()
    return args

#####################################################################################
# Download and extract VOC datasets into ``path``

def download_voc(path, overwrite=False):
    _DOWNLOAD_URLS = [
        ('http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
         '34ed68851bce2a36e2a223fa52c661d592c66b3c'),
        ('http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',
         '41a8d6e12baa5ab18ee7f8f8029b9e11805b4ef1'),
        ('http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
         '4e443f8a2eca6b1dac8a6c57641b67dd40621a49')]
    download_dir = os.path.join(path, 'downloads')
    makedirs(download_dir)
    for url, checksum in _DOWNLOAD_URLS:
        filename = download(url, path=download_dir, overwrite=overwrite, sha1_hash=checksum)
        # extract
        with tarfile.open(filename) as tar:
            tar.extractall(path=path)


#####################################################################################
# Download and extract the VOC augmented segementation dataset into ``path``

def download_aug(path, overwrite=False):
    _AUG_DOWNLOAD_URLS = [
        ('http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz', '7129e0a480c2d6afb02b517bb18ac54283bfaa35')]
    download_dir = os.path.join(path, 'downloads')
    makedirs(download_dir)
    for url, checksum in _AUG_DOWNLOAD_URLS:
        filename = download(url, path=download_dir, overwrite=overwrite, sha1_hash=checksum)
        # extract
        with tarfile.open(filename) as tar:
            tar.extractall(path=path)
            shutil.move(os.path.join(path, 'benchmark_RELEASE'),
                        os.path.join(path, 'VOCaug'))
            filenames = ['VOCaug/dataset/train.txt', 'VOCaug/dataset/val.txt']
            # generate trainval.txt
            with open(os.path.join(path, 'VOCaug/dataset/trainval.txt'), 'w') as outfile:
                for fname in filenames:
                    fname = os.path.join(path, fname)
                    with open(fname) as infile:
                        for line in infile:
                            outfile.write(line)


if __name__ == '__main__':
    args = parse_args()
    path = os.path.expanduser(args.path)
    if not os.path.isdir(path) or not os.path.isdir(os.path.join(path, 'VOC2007')) \
        or not os.path.isdir(os.path.join(path, 'VOC2012')):
        if not args.download:
            raise ValueError(('{} is not a valid directory, make sure it is present.'
                              ' Or you can try "--download" to grab it'.format(path)))
        else:
            download_voc(path, overwrite=args.overwrite)
            shutil.move(os.path.join(path, 'VOCdevkit', 'VOC2007'), os.path.join(path, 'VOC2007'))
            shutil.move(os.path.join(path, 'VOCdevkit', 'VOC2012'), os.path.join(path, 'VOC2012'))
            shutil.rmtree(os.path.join(path, 'VOCdevkit'))

    if not os.path.isdir(os.path.join(path, 'VOCaug')):
        if not args.download:
            raise ValueError(('{} is not a valid directory, make sure it is present.'
                              ' Or you can try "--download" to grab it'.format(path)))
        else:
            download_aug(path, overwrite=args.overwrite)

    # make symlink
    makedirs(os.path.expanduser('~/.mxnet/datasets'))
    if os.path.isdir(_TARGET_DIR):
        os.remove(_TARGET_DIR)
    os.symlink(args.path, _TARGET_DIR)
