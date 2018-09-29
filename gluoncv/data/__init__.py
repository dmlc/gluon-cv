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
from .mscoco.segmentation import COCOSegmentation
from .cityscapes import CitySegmentation
from .pascal_voc.segmentation import VOCSegmentation
from .pascal_aug.segmentation import VOCAugSegmentation
from .ade20k.segmentation import ADE20KSegmentation
from .segbase import ms_batchify_fn
from .recordio.detection import RecordFileDetection
from .lst.detection import LstDetection

datasets = {
    'ade20k': ADE20KSegmentation,
    'pascal_voc': VOCSegmentation,
    'pascal_aug': VOCAugSegmentation,
    'coco' : COCOSegmentation,
    'citys' : CitySegmentation,
}

def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    return datasets[name.lower()](**kwargs)
