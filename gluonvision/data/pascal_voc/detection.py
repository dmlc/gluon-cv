"""Pascal VOC object detection dataset."""
from __future__ import division
import os
import logging
import numpy as np
try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET
import mxnet as mx
from ..base import VisionDataset


class VOCDetection(VisionDataset):
    """Pascal VOC detection Dataset.

    Parameters
    ----------
    root : string
        Path to VOCdevkit folder. Default is '~/mxnet/datasets/voc'
    splits : list of tuples
        List of combinations of (year, name), e.g. [(2007, 'trainval'), (2012, 'train')].
        For years, candidates can be: 2007, 2012.
        For names, candidates can be: 'train', 'val', 'trainval', 'test'.
    flag : {0, 1}, default 1
        If 0, always convert images to greyscale.
        If 1, always convert images to colored (RGB).
    transform : callable, optional
        A function that takes data and label and transforms them::

            transform = lambda data, label: (data.astype(np.float32)/255, label)

        A transform function for object detection should take label into consideration,
        because any geometric modification will require label to be modified.
    index_map : dict, optional
        If provided as dict, class indecies are mapped by looking up in the dict.
        Otherwise will use alphabetic indexing for all classes from 0 to 19.
    preload : bool
        All labels will be parsed and loaded into memory at initialization.
        This will allow early check for errors, and will be significantly faster.
    normalize_bbox : bool
        Whether or not normalize box coordinates to range (0, 1) by dividing width/height.

    """
    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
               'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
    def __init__(self, root=os.path.join('~', '.mxnet', 'datasets', 'voc'),
                 splits='train', flag=1, transform=None, index_map=None,
                 preload=True, normalize_bbox=False):
        super(VOCDetection, self).__init__(root)
        self._normalize_bbox = normalize_bbox
        self._im_shapes = {}
        self._root = os.path.expanduser(root)
        self._flag = flag
        self._transform = transform
        self._splits = splits
        self._items = self._load_items(splits)
        self._anno_path = os.path.join('{}', 'Annotations', '{}.xml')
        self._image_path = os.path.join('{}', 'JPEGImages', '{}.jpg')
        self.index_map = index_map or dict(zip(self.classes, range(self.num_class)))
        self._label_cache = self._preload_labels() if preload else None

    def __str__(self):
        detail = ','.join([str(s[0]) + s[1] for s in self._splits])
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
        label = self._label_cache[idx] if self._label_cache else self._load_label(idx)
        img = mx.image.imread(img_path, self._flag)
        if self._transform is not None:
            return self._transform(img, label)
        return img, label

    def _load_items(self, splits):
        """Load individual image indices from splits."""
        ids = []
        for year, name in splits:
            root = os.path.join(self._root, 'VOC' + str(year))
            lf = os.path.join(root, 'ImageSets', 'Main', name + '.txt')
            with open(lf, 'r') as f:
                ids += [(root, line.strip()) for line in f.readlines()]
        return ids

    def _load_label(self, idx):
        """Parse xml file and return labels."""
        img_id = self._items[idx]
        anno_path = self._anno_path.format(*img_id)
        root = ET.parse(anno_path).getroot()
        size = root.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)
        if idx not in self._im_shapes:
            # store the shapes for later usage
            self._im_shapes[idx] = (width, height)
        label = []
        for obj in root.iter('object'):
            difficult = int(obj.find('difficult').text)
            cls_name = obj.find('name').text.strip().lower()
            if cls_name not in self.classes:
                continue
            cls_id = self.index_map[cls_name]
            xml_box = obj.find('bndbox')
            xmin = (float(xml_box.find('xmin').text) - 1)
            ymin = (float(xml_box.find('ymin').text) - 1)
            xmax = (float(xml_box.find('xmax').text) - 1)
            ymax = (float(xml_box.find('ymax').text) - 1)
            try:
                self._validate_label(xmin, ymin, xmax, ymax, width, height)
            except AssertionError as e:
                raise RuntimeError("Invalid label at {}, {}".format(anno_path, e))
            if self._normalize_bbox:
                label.append([xmin / width, ymin / height, xmax / width,
                              ymax / height, cls_id, difficult])
            else:
                label.append([xmin, ymin, xmax, ymax, cls_id, difficult])
        return np.array(label)

    def _validate_label(self, xmin, ymin, xmax, ymax, width, height):
        """Validate labels."""
        assert xmin >= 0 and xmin < width, (
            "xmin must in [0, {}), given {}".format(width, xmin))
        assert ymin >= 0 and ymin < height, (
            "ymin must in [0, {}), given {}".format(height, ymin))
        assert xmax > xmin and xmax <= width, (
            "xmax must in (xmin, {}], given {}".format(width, xmax))
        assert ymax > ymin and ymax <= height, (
            "ymax must in (ymin, {}], given {}".format(height, ymax))

    def _preload_labels(self):
        """Preload all labels into memory."""
        logging.debug("Preloading %s labels into memory...", str(self))
        return [self._load_label(idx) for idx in range(len(self))]
