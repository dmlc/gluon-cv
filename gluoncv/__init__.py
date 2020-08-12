# coding: utf-8
# pylint: disable=wrong-import-position
"""GluonCV: a deep learning vision toolkit powered by Gluon."""
from __future__ import absolute_import

__version__ = '0.9.0'

from .utils.version import _require_mxnet_version, _deprecate_python2
_deprecate_python2()
_require_mxnet_version('1.4.0', '2.0.0')

from . import data
from . import model_zoo
from . import nn
from . import utils

from . import loss
