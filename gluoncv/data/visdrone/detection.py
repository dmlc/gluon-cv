"""VisDrone2019 object detection dataset."""
from __future__ import absolute_import
from __future__ import division
import os
import logging
import numpy as np
import mxnet as mx
from ..base import VisionDataset


class VisDroneDetection(VisionDataset):
    """VisDrone2019-DET Dataset.

    Parameters
    ----------
    root : str, default '~/mxnet/datasets/visdrone'
        Path to folder storing the dataset.
    splits : list of str, default ['train']
        Candidates can be: train, val.
    transform : callable, default None
        A function that takes data and label and transforms them. Refer to
        :doc:`./transforms` for examples.

        A transform function for object detection should take label into consideration,
        because any geometric modification will require label to be modified.
    """
    CLASSES = ('pedestrian', 'person', 'bicycle', 'car', 'van',
               'truck', 'tricycle', 'awning-tricycle', 'bus',
               'motor', 'others')

    def __init__(self, root=os.path.join('~', '.mxnet', 'datasets', 'visdrone'),
                 splits=('train',),
                 transform=None):
        super(VisDroneDetection, self).__init__(root)
        self._im_shapes = {}
        self._root = os.path.expanduser(root)
        self._transform = transform
        self._splits = splits
        self._anno_path = os.path.join('{}', 'annotations', '{}.txt')
        self._image_path = os.path.join('{}', 'images', '{}.jpg')
        self._images_dir = os.path.join('{}', 'images')
        self._items = self._load_items(self._splits)
        self.num_classes = len(self.CLASSES)
        self.index_map = dict(zip(self.CLASSES, range(self.num_class)))

    def __str__(self):
        detail = ','.join([str(s) for s in self._splits])
        return self.__class__.__name__ + '(' + detail + ')'

    @property
    def classes(self):
        """Category names."""
        return type(self).CLASSES

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        img_id = self._items[idx]
        img_path = self._image_path.format(*img_id)
        img = mx.image.imread(img_path, 1)
        height, width, _ = img.shape
        label = self._load_label(idx, height, width)
        if self._transform is not None:
            return self._transform(img, label)
        return img, label.copy()

    def _load_items(self, splits):
        """Load individual image indices from splits."""
        ids = list()
        for name in splits:
            root = os.path.join(self._root, 'VisDrone2019-DET-' + name)
            images_dir = self._images_dir.format(root)
            images = [f[:-4] for f in os.listdir(images_dir)
                      if os.path.isfile(os.path.join(images_dir, f)) and f[-3:] == 'jpg']
            ids += [(root, line.strip()) for line in images]
        return ids

    def _load_label(self, idx, height=None, width=None):
        """Parse csv file and return labels."""
        img_id = self._items[idx]
        anno_path = self._anno_path.format(*img_id)
        height = float(height) if height else None
        width = float(width) if width else None
        if idx not in self._im_shapes:
            # store the shapes for later usage
            self._im_shapes[idx] = (width, height)
        with open(anno_path, 'r') as f:
            annotations = f.read().splitlines()
        label = []
        for ann_line in annotations:
            ann_info = ann_line.split(',')
            xmin, ymin = float(ann_info[0]), float(ann_info[1])
            w, h = float(ann_info[2]), float(ann_info[3])
            xmax, ymax = xmin + w, ymin + h
            cls_id = int(ann_info[5]) - 1
            try:
                if not height and not width:
                    self._validate_label(xmin, ymin, xmax, ymax, width, height)
            except AssertionError as e:
                raise RuntimeError("Invalid label at {}, {}".format(anno_path, e))
            label.append([xmin, ymin, xmax, ymax, cls_id])
        return np.array(label)

    def _validate_label(self, xmin, ymin, xmax, ymax, width, height):
        """Validate labels."""
        assert 0 <= xmin < width, "xmin must in [0, {}), given {}".format(width, xmin)
        assert 0 <= ymin < height, "ymin must in [0, {}), given {}".format(height, ymin)
        assert xmin < xmax <= width, "xmax must in (xmin, {}], given {}".format(width, xmax)
        assert ymin < ymax <= height, "ymax must in (ymin, {}], given {}".format(height, ymax)

    def _preload_labels(self):
        """Preload all labels into memory."""
        logging.debug("Preloading %s labels into memory...", str(self))
        return [self._load_label(idx) for idx in range(len(self))]
