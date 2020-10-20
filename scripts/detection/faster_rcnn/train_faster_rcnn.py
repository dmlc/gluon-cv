"""Train Faster R-CNN end to end."""
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
gcv.utils.check_version('0.8.0')
from gluoncv.auto.estimators.faster_rcnn import FasterRCNNEstimator
from gluoncv.auto.estimators.faster_rcnn import ex
from gluoncv.auto.data import url_data
from gluoncv.auto.tasks import ObjectDetection as Task

@ex.automain
def main(_config, _log):
    root = url_data('https://autogluon.s3.amazonaws.com/datasets/tiny_motorbike.zip')
    dataset = Task.Dataset.from_voc(root)
    # train_data, val_data, test_data = dataset.random_split(test_size=0.1, val_size=0.1)

    # main is the commandline entry for user w/o coding
    c = FasterRCNNEstimator(_config, _log)
    c.fit(train_data=dataset)
