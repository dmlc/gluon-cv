"""
Prepare PASCAL VOC datasets
==============================

`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ is a common source of data for object
detection and very popular to be served as performance benchmarks.

.. image:: https://github.com/zhreshold/gluonvision-tutorials/blob/master/images/pascal2.png?raw=true

There are many years of data released in the past. The most common combination for
object detection is (2007 trainval + 2012 trainval) and validate on (2007 test).
Test data of year 2012 is not public and requires you to register on the site.

This tutorial will walk you thourgh the preparation steps in order to let GluonVision to recognize
the dataset on your disk automatically. Once finished, you won't need to specify the path and can totoally forget about it.

Download the dataset
--------------------
The included script can automatically download the dataset for you. Depending on your
internet speed, it may take 5 min to several hours.

Assume you want to store the dataset in ``~/pascal_voc``, simply run

.. code-block:: bash

    python scripts/datasets/pascal_voc.py --path ~/pascal_voc --download

How to use Pascal VOC as object detection dataset
-------------------------------------------------

Load image and label from Pascal VOC is quite straight-forward

.. code:: python

    from gluonvision.data import VOCDetection
    train_dataset = VOCDetection(splits=[(2007, 'trainval'), (2012, 'trainval')])
    val_dataset = VOCDetection(splits=[(2007, 'test')])
    print('Training images:', len(train_dataset))
    print('Validation images:', len(val_dataset))

Training and validation images:

.. parsed-literal::

    Training images: 16551
    Validation images: 4952

Images and labels loaded from dataset:

.. code:: python

    train_image, train_label = train_dataset[0]
    bboxes = train_label[:, :4]
    cids = train_label[:, 4:5]
    print('image:', train_image.shape)
    print('bboxes:', bboxes.shape, 'class ids:', cids.shape)


.. parsed-literal::

    image: (375, 500, 3)
    bboxes: (5, 4) class ids: (5, 1)

Dive deep into preparation script
---------------------------------

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
