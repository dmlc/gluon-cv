"""Dataset implementation for specific task(s)"""
import pandas as pd


class ObjectDetectionDataset(pd.DataFrame):
    def __ini__(self, data):
        super().__init__(data)

    @classmethod
    def from_iterable(cls, iterable):
        # lst is a python list with element pairs [(path, label, bbox), (path, label, bbox)...]
        raise NotImplementedError

    @classmethod
    def from_voc(cls, root, splits=None, classes=None):
        # construct from pascal VOC forma
        from ...data.pascal_voc.detection import CustomVOCDetectionBase


    @classmethod
    def from_coco(cls, path):
        # construct from COCO format
        raise NotImplementedError

    @classmethod
    def from_path_func(cls, fn):
        # create from a function
        raise NotImplementedError
