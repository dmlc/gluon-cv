"""DLA network with Deconvolution layers for CenterNet object detection."""
from __future__ import absolute_import

import warnings
import math

import mxnet as mx
from mxnet.context import cpu
from mxnet.gluon import nn
from mxnet.gluon import contrib
from .. model_zoo import get_model
