from __future__ import absolute_import
import os, numbers

from mxnet.gluon.data import dataset
from mxnet import image, nd


class ImageTxtDataset(dataset.Dataset):
    def __init__(self, items, flag=1, transform=None):
        self._flag = flag
        self._transform = transform
        self.items = items

    def __getitem__(self, idx):
        fpath = self.items[idx][0]
        img = image.imread(fpath, self._flag)
        label = self.items[idx][1]
        if self._transform is not None:
            img = self._transform(img)
        return img, label

    def __len__(self):
        return len(self.items)
