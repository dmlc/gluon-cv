"""Train YOLOv3"""

import gluoncv as gcv

gcv.utils.check_version('0.8.0')
from gluoncv.auto.estimators.yolo import YOLOv3Estimator
from gluoncv.auto.estimators.yolo import ex

try:
    import horovod.mxnet as hvd
except ImportError:
    hvd = None


@ex.automain
def main(_config, _log):
    # main is the commandline entry for user w/o coding
    c = YOLOv3Estimator(_config, _log)
    c.fit()
