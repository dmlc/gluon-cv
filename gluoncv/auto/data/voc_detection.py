"""VOC detection pipeline"""
# pylint: disable=unused-variable,missing-function-docstring
import os
import tempfile
from sacred import Ingredient


voc_detection = Ingredient('voc_detection')

@voc_detection.config
def cfg():
    root = os.path.expanduser(os.path.join('~', '.mxnet', 'datasets'))
    train_splits = 'instances_train2017'
    valid_splits = 'instances_val2017'
    valid_skip_empty = False
    data_shape = None
    cleanup = True

def load_voc_detection(root, train_splits, valid_splits, valid_skip_empty,
                       data_shape, cleanup=True, post_affine=None):
    train_dataset = COCODetection(root=os.path.join(root, 'coco'), splits=train_splits)
    val_dataset = COCODetection(root=os.path.join(root, 'coco'),
                                splits=valid_splits, skip_empty=valid_skip_empty)
    val_metric = COCODetectionMetric(val_dataset,
                                     tempfile.NamedTemporaryFile('w', delete=False).name,
                                     cleanup=cleanup,
                                     data_shape=data_shape,
                                     post_affine=post_affine)
    return train_dataset, val_dataset, val_metric
