# coding: utf-8
# pylint: disable=wrong-import-position
"""GluonCV: a deep learning vision toolkit powered by Gluon."""
from __future__ import absolute_import

from .check import _deprecate_python2
from .check import _require_mxnet_version, _require_pytorch_version

__version__ = '0.11.0'

_deprecate_python2()

# optionally depend on mxnet or pytorch
_found_mxnet = _found_pytorch = False
try:
    _require_mxnet_version('1.4.0', '2.0.0')
    from . import data
    from . import model_zoo
    from . import nn
    from . import utils
    from . import loss
    _found_mxnet = True
except ImportError:
    pass

try:
    _require_pytorch_version('1.4.0', '2.0.0')
    _found_pytorch = True
except ImportError:
    pass

if not any((_found_mxnet, _found_pytorch)):
    raise ImportError('Unable to import modules due to missing `mxnet` & `torch`. '
                      'You should install at least one deep learning framework.')

if all((_found_mxnet, _found_pytorch)):
    import warnings
    import mxnet as mx
    import torch
    warnings.warn(f'Both `mxnet=={mx.__version__}` and `torch=={torch.__version__}` are installed. '
                  'You might encounter increased GPU memory footprint if both framework are used at the same time.')
