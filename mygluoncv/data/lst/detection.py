"""Detection Dataset from LST file."""
from __future__ import absolute_import
import os
import numpy as np
import mxnet as mx
from mxnet.gluon.data import Dataset
from ..recordio.detection import _transform_label


class LstDetection(Dataset):
    """Detection dataset loaded from LST file and raw images.
    LST file is a pure text file but with special label format.

    Checkout :ref:`lst_record_dataset` for tutorial of how to prepare this file.

    Parameters
    ----------
    filename : type
        Description of parameter `filename`.
    root : str
        Relative image root folder for filenames in LST file.
    flag : int, default is 1
        Use 1 for color images, and 0 for gray images.
    coord_normalized : boolean
        Indicate whether bounding box coordinates haved been normalized to (0, 1) in labels.
        If so, we will rescale back to absolute coordinates by multiplying width or height.

    """
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
