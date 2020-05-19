# coding: utf-8
# pylint: disable=wrong-import-position
"""GluonCV: a deep learning vision toolkit powered by Gluon."""
from __future__ import absolute_import

__version__ = '0.8.0'

from .utils.version import _require_mxnet_version, _deprecate_python2
_deprecate_python2()
_require_mxnet_version('1.4.0')
try:
    # pylint: disable=unused-import
    from mxnet import metric
except ImportError:
    from mxnet.gluon import metric
    import mxnet
    mxnet.metric = metric

from . import data
from . import model_zoo
from . import nn
from . import utils

from . import loss
