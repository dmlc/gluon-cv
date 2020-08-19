"""n SSD"""

import gluoncv as gcv

gcv.utils.check_version('0.8.0')
from gluoncv.auto.estimators.yolo import YoloEstimator
from gluoncv.auto.estimators.yolo import ex

try:
    import horovod.mxnet as hvd
except ImportError:
    hvd = None


@ex.automain
def main(_config, _log):
    # main is the commandline entry for user w/o coding
    c = YoloEstimator(_config, _log)
    c.fit()
