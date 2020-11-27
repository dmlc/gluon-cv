import logging

import gluoncv as gcv
gcv.utils.check_version('0.8.0')

from gluoncv.auto.estimators import CenterNetEstimator
from gluoncv.auto.tasks.utils import config_to_nested
from d8.object_detection import Dataset


if __name__ == '__main__':
    # specify hyperparameters
    config = {
        'dataset': 'sheep',
        'gpus': [0, 1, 2, 3],
        'estimator': CenterNetEstimator,
        'base_network': 'resnet50_v1b',
        'batch_size': 32,  # range [4, 32]
        'epochs': 3
    }
    config = config_to_nested(config)
    config.pop('estimator')

    # specify dataset
    dataset = Dataset.get('sheep')
    train_data, valid_data = dataset.split(0.8)

    # specify estimator
    estimator = CenterNetEstimator(config)

    # fit estimator
    estimator.fit(train_data, valid_data)

    # evaluate auto estimator
    eval_map = estimator.evaluate(valid_data)
    logging.info('evaluation: mAP={}'.format(eval_map[-1][-1]))
