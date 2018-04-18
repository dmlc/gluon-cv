"""Gluon Vision Model Zoo"""
# pylint: disable=wildcard-import
from .model_zoo import get_model
from .dilated import dilatedresnetv0, dilatedresnetv2
from .ssd import *
from .cifarresnet import *
from .cifarwideresnet import *
from .segbase import SegBaseModel, SegEvalModel
from .fcn import *
from .pspnet import *
