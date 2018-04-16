"""Gluon Vision Model Zoo"""
# pylint: disable=wildcard-import
from .model_zoo import get_model
from .dilated import dilatedresnetv1, dilatedresnetv2
from .ssd import *
from .cifarresnet import *
from .cifarwideresnet import *
from .segbase import SegBaseModel
from .fcn import FCN
from .pspnet import PSPNet
