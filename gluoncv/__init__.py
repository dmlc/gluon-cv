# coding: utf-8
# pylint: disable=wrong-import-position
"""GluonCV: a deep learning vision toolkit powered by Gluon."""
from __future__ import absolute_import

from .check import _deprecate_python2
from .backend import (
    get_backend,
    set_backend,
    lock_backend,
    unlock_backend,
    is_backend_locked,
    is_available,
    get_version,
    get_module,
    backend_context,
    require_backend,
    check_backend_available,
    Backend,
    BackendManager,
)

__version__ = '0.11.0'

_deprecate_python2()

_manager = BackendManager()

if _manager.mxnet_available:
    from . import data
    from . import model_zoo
    from . import nn
    from . import utils
    from . import loss

if not any((_manager.mxnet_available, _manager.pytorch_available)):
    raise ImportError('Unable to import modules due to missing `mxnet` & `torch`. '
                      'You should install at least one deep learning framework.')

if all((_manager.mxnet_available, _manager.pytorch_available)):
    import warnings
    warnings.warn(
        f'Both `mxnet=={_manager.mxnet_version}` and `torch=={_manager.pytorch_version}` are installed. '
        'You might encounter increased GPU memory footprint if both framework are used at the same time. '
        'Use gluoncv.set_backend() to explicitly select a backend.'
    )
