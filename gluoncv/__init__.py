# coding: utf-8
# pylint: disable=wrong-import-position
"""GluonCV: a deep learning vision toolkit powered by Gluon."""
from __future__ import absolute_import

from .check import _deprecate_python2
from .check import _require_mxnet_version, _require_pytorch_version

__version__ = '0.9.0'

_deprecate_python2()

# optionally depend on mxnet or pytorch
try:
    _require_mxnet_version('1.4.0', '2.0.0')
    from . import data
    from . import model_zoo
    from . import nn
    from . import utils
    from . import loss
except ImportError:
    try:
        _require_pytorch_version('1.4.0', '2.0.0')
        from .torch import data
        from .torch import model_zoo
        from .torch import utils
    except ImportError:
        raise ImportError('Unable to import modules due to missing `mxnet` & `torch`. '
                          'You should install at least one deep learning framework.')
