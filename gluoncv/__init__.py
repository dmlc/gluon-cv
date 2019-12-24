# coding: utf-8
# pylint: disable=wrong-import-position
"""GluonCV: a deep learning vision toolkit powered by Gluon."""
from __future__ import absolute_import

__version__ = '0.6.0'

from . import data
from . import model_zoo
from . import nn
from . import utils
from .utils.version import _require_mxnet_version
from . import loss

_require_mxnet_version('1.4.0')
