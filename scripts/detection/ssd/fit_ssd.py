import logging

import gluoncv as gcv
gcv.utils.check_version('0.8.0')

from gluoncv.auto.estimators import SSDEstimator
from gluoncv.auto.tasks.utils import config_to_nested
from d8.object_detection import Dataset


if __name__ == '__main__':
    # specify hyperparameters
    config = {
        'dataset': 'sheep',
        'gpus': [0, 1, 2, 3, 4, 5, 6, 7],
        'estimator': 'ssd',
        'base_network': 'resnet50_v1',
        'data_shape': 512,
        'batch_size': 64,  # range [8, 16, 32, 64]
        'epochs': 3
    }
    config = config_to_nested(config)
    config.pop('estimator')

    # specify dataset
    dataset = Dataset.get('sheep')
    train_data, valid_data = dataset.split(0.8)

    # specify estimator
    estimator = SSDEstimator(config)

    # fit estimator
    estimator.fit(train_data, valid_data)

    # evaluate auto estimator
    eval_map = estimator.evaluate(valid_data)
    logging.info('evaluation: mAP={}'.format(eval_map[-1][-1]))
