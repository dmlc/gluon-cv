from __future__ import absolute_import
import os
import numpy as np
import mxnet as mx
from ..recordio.detection import _transform_label


class ListDetection(object):
    def __init__(self, filename, root='', flag=1, coord_normalized=True):
        self._flag = flag
        self._coord_normalized = coord_normalized
        self._items = []
        self._labels = []
        full_path = os.path.expanduser(filename)
        with open(full_path) as fin:
            for line in iter(fin.readline, ''):
                line = line.strip().split('\t')
                label = np.array(line[1:-1]).astype('float')
                im_path = os.path.join(root, line[-1])
                self._items.append(im_path)
                self._labels.append(label)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        im_path = self._items[idx]
        img = mx.image.imread(im_path, self._flag)
        h, w, _ = img.shape
        label = self._labels[idx]
        if self._coord_normalized:
            label = _transform_label(label, h, w)
        else:
            label = _transform_label(label)
        return img, label
