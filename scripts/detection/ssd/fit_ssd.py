"""Train SSD"""

import gluoncv as gcv
gcv.utils.check_version('0.8.0')
from gluoncv.auto.estimators.ssd import SSDEstimator
from gluoncv.auto.estimators.ssd import ex
from gluoncv.auto.data import url_data
from gluoncv.auto.tasks import ObjectDetection as Task


@ex.automain
def main(_config, _log):
    root = url_data('https://autogluon.s3.amazonaws.com/datasets/tiny_motorbike.zip')
    dataset = Task.Dataset.from_voc(root)
    # train_data, val_data, test_data = dataset.random_split(test_size=0.1, val_size=0.1)

    # main is the commandline entry for user w/o coding
    c = SSDEstimator(_config, _log)
    c.fit(train_data=dataset)
