"""Estimator implementations"""
from .utils import create_dummy_estimator
# FIXME: for quick test purpose only
try:
    import mxnet
    from .image_classification import ImageClassificationEstimator
    from .ssd import SSDEstimator
    from .yolo import YOLOv3Estimator
    from .faster_rcnn import FasterRCNNEstimator
    # from .mask_rcnn import MaskRCNNEstimator
    from .center_net import CenterNetEstimator
except ImportError:
    # create dummy placeholder estimator classes
    reason = 'gluoncv.auto.estimators.{} requires mxnet to be installed which is missing.'
    ImageClassificationEstimator = create_dummy_estimator(
        'ImageClassificationEstimator', reason)
    SSDEstimator = create_dummy_estimator(
        'SSDEstimator', reason)
    YOLOv3Estimator = create_dummy_estimator(
        'YOLOv3Estimator', reason)
    FasterRCNNEstimator = create_dummy_estimator(
        'FasterRCNNEstimator', reason)
    CenterNetEstimator = create_dummy_estimator(
        'CenterNetEstimator', reason)

try:
    import timm
    import torch
    from .torch_image_classification import TorchImageClassificationEstimator
except ImportError:
    reason = 'This estimator requires torch/timm to be installed which is missing.'
    TorchImageClassificationEstimator = create_dummy_estimator(
        'TorchImageClassificationEstimator', reason)
