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
    def __init__(self, root=os.path.join('~', '.mxnet', 'datasets', 'coco'),
                 splits=('coco_2017_test')):
        super(COCODetection, self).__init__(root)
        self._root = os.path.expanduser(root)
        # lazy import pycocotools
        try_import_pycocotools()
        from pycocotools.coco import COCO
        
