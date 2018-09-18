"""
This module provides data loaders and transfomers for popular vision datasets.
"""
from . import transforms
from . import batchify
from .imagenet.classification import ImageNet
from .dataloader import DetectionDataLoader, RandomTransformDataLoader
from .pascal_voc.detection import VOCDetection
from .mscoco.detection import COCODetection
from .mscoco.instance import COCOInstance
from .pascal_voc.segmentation import VOCSegmentation
from .pascal_aug.segmentation import VOCAugSegmentation
from .ade20k.segmentation import ADE20KSegmentation
from .segbase import get_segmentation_dataset, ms_batchify_fn
from .recordio.detection import RecordFileDetection
from .lst.detection import LstDetection
