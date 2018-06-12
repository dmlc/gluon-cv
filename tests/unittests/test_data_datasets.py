from __future__ import print_function
from __future__ import division

import mxnet as mx
import numpy as np

import gluoncv as gcv
from gluoncv import data
import os.path as osp


def test_pascal_voc_detection():
    if not osp.isdir(osp.expanduser('~/.mxnet/datasets/voc')):
        return

    train = data.VOCDetection(splits=((2007, 'trainval'), (2012, 'trainval')))
    name = str(train)
    val = data.VOCDetection(splits=((2007, 'test'), ))
    name = str(val)

    assert train.classes == val.classes

    for _ in range(10):
        index = np.random.randint(0, len(train))
        _ = train[index]

    for _ in range(10):
        index = np.random.randint(0, len(val))
        _ = val[index]


def test_coco_detection():
    if not osp.isdir(osp.expanduser('~/.mxnet/datasets/coco')):
        return

    # use valid only, loading training split is very slow
    val = data.COCODetection(splits=('instances_val2017'))
    name = str(val)
    assert len(val.classes) > 0

    for _ in range(10):
        index = np.random.randint(0, len(val))
        _ = val[index]


def test_voc_segmentation():
    if not osp.isdir(osp.expanduser('~/.mxnet/datasets/voc')):
        return

    # use valid only, loading training split is very slow
    val = data.VOCSegmentation(split='train')
    name = str(val)
    assert len(val.classes) > 0

    for _ in range(10):
        index = np.random.randint(0, len(val))
        _ = val[index]

def test_voc_aug_segmentation():
    if not osp.isdir(osp.expanduser('~/.mxnet/datasets/voc')):
        return

    # use valid only, loading training split is very slow
    val = data.VOCAugSegmentation(split='train')
    name = str(val)
    assert len(val.classes) > 0

    for _ in range(10):
        index = np.random.randint(0, len(val))
        _ = val[index]

def test_ade_segmentation():
    if not osp.isdir(osp.expanduser('~/.mxnet/datasets/ade')):
        return

    # use valid only, loading training split is very slow
    val = data.ADE20KSegmentation(split='train')
    name = str(val)

    for _ in range(10):
        index = np.random.randint(0, len(val))
        _ = val[index]

if __name__ == '__main__':
    import nose
    nose.runmodule()
