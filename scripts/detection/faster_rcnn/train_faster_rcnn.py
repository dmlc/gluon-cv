"""Train Faster-RCNN end to end."""
import os

# disable autotune
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_GPU_MEM_POOL_TYPE'] = 'Round'
os.environ['MXNET_GPU_MEM_POOL_ROUND_LINEAR_CUTOFF'] = '26'
os.environ['MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD'] = '999'
os.environ['MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD'] = '25'
os.environ['MXNET_GPU_COPY_NTHREADS'] = '1'
os.environ['MXNET_OPTIMIZER_AGGREGATION_SIZE'] = '54'

import gluoncv as gcv

gcv.utils.check_version('0.7.0')
from gluoncv.auto.estimators.rcnn import FasterRCNNEstimator
from gluoncv.auto.estimators.rcnn import ex

try:
    import horovod.mxnet as hvd
except ImportError:
    hvd = None


@ex.automain
def main(_config, _log):
    # main is the commandline entry for user w/o coding
    c = FasterRCNNEstimator(_config, _log)
    c.fit()
