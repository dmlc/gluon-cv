"""GluonCV Utility functions."""
# pylint: disable=wildcard-import,exec-used,wrong-import-position
from __future__ import absolute_import

import types

def import_dummy_module(code, name):
    # create blank module
    module = types.ModuleType(name)
    # populate the module with code
    exec(code, module.__dict__)
    return module

dummy_module = """
def __getattr__(name):
    raise AttributeError(f"gluoncv.utils.{__name__} module requires mxnet which is missing.")
"""


from . import bbox
from . import random
from . import filesystem
try:
    import mxnet
    from . import viz
    from . import metrics
    from . import parallel
    from .lr_scheduler import LRSequential, LRScheduler
    from .export_helper import export_block, export_tvm
    from .sync_loader_helper import split_data, split_and_load
except ImportError:
    viz = import_dummy_module(dummy_module, 'viz')
    metrics = import_dummy_module(dummy_module, 'metrics')
    parallel = import_dummy_module(dummy_module, 'parallel')
    LRSequential, LRScheduler = None, None
    export_block, export_tvm = None, None
    split_data, split_and_load = None, None

from .download import download, check_sha1
from .filesystem import makedirs, try_import_dali, try_import_cv2
from .bbox import bbox_iou
from .block import recursive_visit, set_lr_mult, freeze_bn
from .plot_history import TrainingHistory
from .version import *
