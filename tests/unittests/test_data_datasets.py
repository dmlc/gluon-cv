from __future__ import print_function
from __future__ import division

import os
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

def test_coco_instance():
    if not osp.isdir(osp.expanduser('~/.mxnet/datasets/coco')):
        return

    # use valid only, loading training split is very slow
    val = data.COCOInstance(splits=('instances_val2017',))
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

def test_citys_segmentation():
    if not osp.isdir(osp.expanduser('~/.mxnet/datasets/ade')):
        return

    # use valid only, loading training split is very slow
    val = data.CitySegmentation(split='train')
    name = str(val)

    for _ in range(10):
        index = np.random.randint(0, len(val))
        _ = val[index]

def test_lst_detection():
    dog_label = [130, 220, 320, 530]
    bike_label = [115, 120, 580, 420]
    car_label = [480, 80, 700, 170]
    all_boxes = np.array([dog_label, bike_label, car_label])
    all_ids = np.array([0, 1, 2])
    class_names = ['dog', 'bike', 'car']
    im_fname = gcv.utils.download('https://github.com/dmlc/web-data/blob/master/' +
                              'gluoncv/datasets/dog.jpg?raw=true',
                              path='dog.jpg')
    img = mx.image.imread(im_fname)

    def write_line(img_path, im_shape, boxes, ids, idx):
        h, w, c = im_shape
        # for header, we use minimal length 2, plus width and height
        # with A: 4, B: 5, C: width, D: height
        A = 4
        B = 5
        C = w
        D = h
        # concat id and bboxes
        labels = np.hstack((ids.reshape(-1, 1), boxes)).astype('float')
        # normalized bboxes (recommended)
        labels[:, (1, 3)] /= float(w)
        labels[:, (2, 4)] /= float(h)
        # flatten
        labels = labels.flatten().tolist()
        str_idx = [str(idx)]
        str_header = [str(x) for x in [A, B, C, D]]
        str_labels = [str(x) for x in labels]
        str_path = [img_path]
        line = '\t'.join(str_idx + str_header + str_labels + str_path) + '\n'
        return line

    with open('val.lst', 'w') as fw:
        for i in range(4):
            line = write_line('dog.jpg', img.shape, all_boxes, all_ids, i)
            print(line)
            fw.write(line)
    from gluoncv.data import LstDetection
    lst_dataset = LstDetection('val.lst', root=os.path.expanduser('.'))
    assert len(lst_dataset) == 4
    lst_dataset.transform(lambda x, y : (x, y))
    try:
        os.remove('dog.jpg')
        os.remove('val.lst')
    except IOError:
        pass

if __name__ == '__main__':
    import nose
    nose.runmodule()
