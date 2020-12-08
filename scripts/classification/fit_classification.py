import logging

import gluoncv as gcv
gcv.utils.check_version('0.8.0')

from gluoncv.auto.estimators import ImageClassificationEstimator
from gluoncv.auto.tasks.utils import config_to_nested
from d8.image_classification import Dataset


if __name__ == '__main__':
    # specify hyperparameters
    config = {
        'dataset': 'boat',
        'gpus': [0, 1, 2, 3, 4, 5, 6, 7],
        'estimator': 'img_cls',
        'model': 'resnet50_v1b',
        'batch_size': 128,  # range [16, 32, 64, 128]
        'epochs': 3
    }
    config = config_to_nested(config)
    config.pop('estimator')

    # specify dataset
    dataset = Dataset.get('boat')
    train_data, valid_data = dataset.split(0.8)

    # specify estimator
    estimator = ImageClassificationEstimator(config)

    # fit estimator
    estimator.fit(train_data, valid_data)

    # evaluate auto estimator
    top1, top5 = estimator.evaluate(valid_data)
    logging.info('evaluation: top1={}, top5={}'.format(top1, top5))
