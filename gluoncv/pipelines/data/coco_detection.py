"""COCO detection pipeline"""
import os
from sacred import Ingredient

coco_detection = Ingredient('coco_detection')

@coco_detection.config
def cfg():
    root = os.path.expanduser(os.path.join('~', '.mxnet', 'datasets'))
    train_splits = 'instances_train2017'
    valid_splits = 'instances_val2017'
    valid_skip_empty = False
    data_shape = None
    cleanup = True
