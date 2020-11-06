import os
from gluoncv import data

class COCODetectionTiny(data.COCODetection):
    CLASSES = ['bicycle', 'motorcycle']

    def __init__(self, root=os.path.join('~', '.mxnet', 'datasets', 'tiny_coco'),
                 splits=('instances_val2017_tiny',), **kwargs):
        super().__init__(root=root, splits=splits, **kwargs)

class COCOInstanceTiny(data.COCOInstance):
    CLASSES = ['bicycle', 'motorcycle']

    def __init__(self, root=os.path.join('~', '.mxnet', 'datasets', 'tiny_coco'),
                 splits=('instances_val2017_tiny',), **kwargs):
        super().__init__(root=root, splits=splits, **kwargs)

class VOCDetectionTiny(data.VOCDetection):
    CLASSES = ['motorbike', 'person']

    def __init__(self, root=os.path.join('~', '.mxnet', 'datasets', 'tiny_motorbike'),
                 splits=(('tiny_motorbike', 'trainval'),), **kwargs):
        super().__init__(root=root, splits=splits, **kwargs)

class VOCSegmentationTiny(data.VOCSegmentation):
    CLASSES = ['motorbike', 'person']
    BASE_DIR = 'tiny_motorbike'

    def __init__(self, root=os.path.expanduser(os.path.join('~', '.mxnet', 'datasets', 'tiny_motorbike')),
                 split='train', **kwargs):
        super().__init__(root=root, split=split, **kwargs)
