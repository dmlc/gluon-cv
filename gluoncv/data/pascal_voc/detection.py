"""Pascal VOC object detection dataset."""
from __future__ import absolute_import
from __future__ import division

import glob
import logging
import os
import warnings

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
    root : str, default '~/mxnet/datasets/voc'
        Path to folder storing the dataset.
    splits : list of tuples, default ((2007, 'trainval'), (2012, 'trainval'))
        List of combinations of (year, name)
        For years, candidates can be: 2007, 2012.
        For names, candidates can be: 'train', 'val', 'trainval', 'test'.
    transform : callable, default None
        A function that takes data and label and transforms them. Refer to
        :doc:`./transforms` for examples.

        A transform function for object detection should take label into consideration,
        because any geometric modification will require label to be modified.
    index_map : dict, default None
        In default, the 20 classes are mapped into indices from 0 to 19. We can
        customize it by providing a str to int dict specifying how to map class
        names to indices. Use by advanced users only, when you want to swap the orders
        of class labels.
    preload_label : bool, default True
        If True, then parse and load all labels into memory during
        initialization. It often accelerate speed but require more memory
        usage. Typical preloaded labels took tens of MB. You only need to disable it
        when your dataset is extremely large.
    """
    CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
               'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

    def __init__(self, root=os.path.join('~', '.mxnet', 'datasets', 'voc'),
                 splits=((2007, 'trainval'), (2012, 'trainval')),
                 transform=None, index_map=None, preload_label=True):
        super(VOCDetection, self).__init__(root)
        self._im_shapes = {}
        self._root = os.path.expanduser(root)
        self._transform = transform
        self._splits = splits
        self._items = self._load_items(splits)
        self._anno_path = os.path.join('{}', 'Annotations', '{}.xml')
        self._image_path = os.path.join('{}', 'JPEGImages', '{}.jpg')
        self.index_map = index_map or dict(zip(self.classes, range(self.num_class)))
        self._label_cache = self._preload_labels() if preload_label else None

    def __str__(self):
        detail = ','.join([str(s[0]) + s[1] for s in self._splits])
        return self.__class__.__name__ + '(' + detail + ')'

    @property
    def classes(self):
        """Category names."""
        try:
            self._validate_class_names(self.CLASSES)
        except AssertionError as e:
            raise RuntimeError("Class names must not contain {}".format(e))
        return type(self).CLASSES

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        img_id = self._items[idx]
        img_path = self._image_path.format(*img_id)
        label = self._label_cache[idx] if self._label_cache else self._load_label(idx)
        img = mx.image.imread(img_path, 1)
        if self._transform is not None:
            return self._transform(img, label)
        return img, label.copy()

    def _load_items(self, splits):
        """Load individual image indices from splits."""
        ids = []
        for subfolder, name in splits:
            root = os.path.join(
                self._root, ('VOC' + str(subfolder)) if isinstance(subfolder, int) else subfolder)
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
            try:
                difficult = int(obj.find('difficult').text)
            except ValueError:
                difficult = 0
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
                label.append([xmin, ymin, xmax, ymax, cls_id, difficult])
            except AssertionError as e:
                logging.warning("Invalid label at %s, %s", anno_path, e)
        return np.array(label)

    def _validate_label(self, xmin, ymin, xmax, ymax, width, height):
        """Validate labels."""
        assert 0 <= xmin < width, "xmin must in [0, {}), given {}".format(width, xmin)
        assert 0 <= ymin < height, "ymin must in [0, {}), given {}".format(height, ymin)
        assert xmin < xmax <= width, "xmax must in (xmin, {}], given {}".format(width, xmax)
        assert ymin < ymax <= height, "ymax must in (ymin, {}], given {}".format(height, ymax)

    def _validate_class_names(self, class_list):
        """Validate class names."""
        assert all(c.islower() for c in class_list), "uppercase characters"
        stripped = [c for c in class_list if c.strip() != c]
        if stripped:
            warnings.warn('white space removed for {}'.format(stripped))

    def _preload_labels(self):
        """Preload all labels into memory."""
        logging.debug("Preloading %s labels into memory...", str(self))
        return [self._load_label(idx) for idx in range(len(self))]


class CustomVOCDetection(VOCDetection):
    """Custom Pascal VOC detection Dataset.
    Classes are generated from dataset
    generate_classes : bool, default False
        If True, generate class labels base on the annotations instead of the default classe labels.
    """

    def __init__(self, generate_classes=False, **kwargs):
        super(CustomVOCDetection, self).__init__(**kwargs)
        if generate_classes:
            self.CLASSES = self._generate_classes()

    def _generate_classes(self):
        classes = set()
        all_xml = glob.glob(os.path.join(self._root, 'Annotations', '*.xml'))
        for each_xml_file in all_xml:
            tree = ET.parse(each_xml_file)
            root = tree.getroot()
            for child in root:
                if child.tag == 'object':
                    for item in child:
                        if item.tag == 'name':
                            classes.add(item.text)
        classes = sorted(list(classes))
        return classes


class CustomVOCDetectionBase(VOCDetection):
    """Base class for custom Dataset which follows protocol/formatting of the well-known VOC object detection dataset.

    Parameters
    ----------
    class: tuple of classes, default = None
        We reuse the neural network weights if the corresponding class appears in the pretrained model.
        Otherwise, we randomly initialize the neural network weights for new classes.
    root : str, default '~/mxnet/datasets/voc'
        Path to folder storing the dataset.
    splits : list of tuples, default ((2007, 'trainval'), (2012, 'trainval'))
        List of combinations of (year, name)
        For years, candidates can be: 2007, 2012.
        For names, candidates can be: 'train', 'val', 'trainval', 'test'.
    transform : callable, default = None
        A function that takes data and label and transforms them. Refer to
        :doc:`./transforms` for examples.
        A transform function for object detection should take label into consideration,
        because any geometric modification will require label to be modified.
    index_map : dict, default = None
        By default, the 20 classes are mapped into indices from 0 to 19. We can
        customize it by providing a str to int dict specifying how to map class
        names to indices. This is only for advanced users, when you want to swap the orders
        of class labels.
    preload_label : bool, default = True
        If True, then parse and load all labels into memory during
        initialization. It often accelerate speed but require more memory
        usage. Typical preloaded labels took tens of MB. You only need to disable it
        when your dataset is extremely large.
    """

    def __init__(self, classes=None, root=os.path.join('~', '.mxnet', 'datasets', 'voc'),
                 splits=((2007, 'trainval'), (2012, 'trainval')),
                 transform=None, index_map=None, preload_label=True):

        # update classes
        if classes:
            self._set_class(classes)
        super(CustomVOCDetectionBase, self).__init__(root=root,
                                                     splits=splits,
                                                     transform=transform,
                                                     index_map=index_map,
                                                     preload_label=False)
        self._items_new = [self._items[each_id] for each_id in range(len(self._items)) if self._check_valid(each_id)]
        self._items = self._items_new
        self._label_cache = self._preload_labels() if preload_label else None

    @classmethod
    def _set_class(cls, classes):
        cls.CLASSES = classes

    def _load_items(self, splits):
        """Load individual image indices from splits."""
        ids = []
        for subfolder, name in splits:
            root = os.path.join(self._root, subfolder) if subfolder else self._root
            lf = os.path.join(root, 'ImageSets', 'Main', name + '.txt')
            with open(lf, 'r') as f:
                ids += [(root, line.strip()) for line in f.readlines()]
        return ids

    def _check_valid(self, idx, allow_difficult=True):
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
        for obj in root.iter('object'):
            try:
                difficult = int(obj.find('difficult').text)
            except ValueError:
                difficult = 0
            cls_name = obj.find('name').text.strip().lower()
            if cls_name not in self.classes:
                continue
            if difficult and not allow_difficult:
                continue
            # cls_id = self.index_map[cls_name]
            xml_box = obj.find('bndbox')
            xmin = (float(xml_box.find('xmin').text) - 1)
            ymin = (float(xml_box.find('ymin').text) - 1)
            xmax = (float(xml_box.find('xmax').text) - 1)
            ymax = (float(xml_box.find('ymax').text) - 1)

            if not ((0 <= xmin < width) and (0 <= ymin < height) \
                and (xmin < xmax <= width) and (ymin < ymax <= height)):
                return False

        return True
