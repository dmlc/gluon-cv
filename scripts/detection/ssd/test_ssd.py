from gluoncv.auto.estimators import SSDEstimator
from gluoncv.auto.tasks.utils import config_to_nested
from d8.object_detection import Dataset

if __name__ == '__main__':
    dataset = Dataset.get('chess')
    train_data, valid_data = dataset.split(0.8)

    config = {
        'estimator': SSDEstimator,
        'batch_size': 16,
        'epochs': 5
    }
    config = config_to_nested(config)
    config.pop('estimator')

    estimator = SSDEstimator(config)
    estimator.fit(train_data, valid_data)
