# coding: utf-8
# pylint: disable=wrong-import-position
"""GluonCV: a deep learning vision toolkit powered by Gluon."""
from __future__ import absolute_import

__version__ = '0.8.0'

from .utils.version import _require_mxnet_version, _deprecate_python2, _mxnet_v2_legacy_hook
_deprecate_python2()
_require_mxnet_version('1.4.0')
_mxnet_v2_legacy_hook()

from . import data
from . import model_zoo
from . import nn
from . import utils

from . import loss
