"""CenterNet Estimator"""
import os
import warnings
import tempfile
import mxnet as mx
from mxnet import gluon
from ...data import COCODetection, VOCDetection
from ...data.transforms.presets.center_net import CenterNetDefaultTrainTransform
from ...data.transforms.presets.center_net import CenterNetDefaultValTransform, get_post_transform
from ...data.batchify import Tuple, Stack, Pad
from ...utils.metrics import COCODetectionMetric
from ...utils.metrics.accuracy import Accuracy
from ...utils import LRScheduler, LRSequential
from ...model_zoo.center_net import get_center_net
from ...loss import *

from ..data.coco_detection import coco_detection
from .base_estimator import BaseEstimator, set_default
from .common import train_hyperparams

from sacred import Experiment

ex = Experiment('center_net_default', ingredients=[coco_detection, train_hyperparams])

@coco_detection.config
def update_coco_detection():
    data_shape = (512, 512)  # override coco config for center_net

@coco_detection.capture
def load_dataset(root, train_splits, valid_splits, valid_skip_empty, data_shape, cleanup):
    train_dataset = COCODetection(root=os.path.join(root, 'coco'), splits=train_splits)
    val_dataset = COCODetection(root=os.path.join(root, 'coco'),
                                splits=valid_splits, skip_empty=valid_skip_empty)
    val_metric = COCODetectionMetric(val_dataset,
                                     tempfile.NamedTemporaryFile('w', delete=False).name,
                                     cleanup=cleanup,
                                     data_shape=data_shape,
                                     post_affine=get_post_transform)
    return train_dataset, val_dataset, val_metric

@ex.config
def model():
    base_network = 'dla34_deconv_dcnv2'

@set_default(ex)
class CenterNetEstimator(BaseEstimator):
    def __init__(self, config, logger=None):
        super(CenterNetEstimator, self).__init__(config, logger)
        print(self._cfg)
        self._cfg.train_hyperparams.batch_size = 256
        print(self._cfg)

    def _fit(self):
        pass

@ex.automain
def main(_config, _log):
    # main is the commandline entry for user w/o coding
    c = CenterNetEstimator(_config, _log)
    c.fit()
