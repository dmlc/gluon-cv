"""MS COCO object detection dataset."""
from __future__ import absolute_import
from __future__ import division
import os
import logging
import numpy as np
import mxnet as mx
from .utils import try_import_pycocotools
from ..base import VisionDataset

__all__ = ['COCODetection']


class COCODetection(VisionDataset):
    CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
               'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
               'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
               'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
               'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
               'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
               'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
               'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
               'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
               'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    def __init__(self, root=os.path.join('~', '.mxnet', 'datasets', 'coco'),
                 splits=('instances_val2017',)):
        super(COCODetection, self).__init__(root)
        self._root = os.path.expanduser(root)
        splits = [splits] if isinstance(splits, str) else splits
        self._splits = splits
        # to avoid trouble, we always use continous IDs except dealing with cocoapi
        self.index_map = dict(zip(type(self).CLASSES, range(self.num_class)))
        self._json_id_to_continous = None
        self._continous_id_to_json = None
        items, labels = self._load_jsons()

    @property
    def classes(self):
        """Category names."""
        return type(self).CLASSES

    def _load_jsons(self):
        items = []
        labels = []
        # lazy import pycocotools
        try_import_pycocotools()
        from pycocotools.coco import COCO
        for s in self._splits:
            anno = os.path.join(self._root, 'annotations', s) + '.json'
            _coco = COCO(anno)
            classes = [c['name'] for c in _coco.loadCats(_coco.getCatIds())]
            if not classes == self.classes:
                raise ValueError("Incompatible category names with COCO: ")
            assert classes == self.classes
            json_id_to_continous = {
                v: k + 1 for k, v in enumerate(_coco.getCatIds())}
            if self._json_id_to_continous is None:
                self._json_id_to_continous = json_id_to_continous
            else:



        return items, labels
