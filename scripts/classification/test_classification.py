from gluoncv.auto.estimators import ImageClassificationEstimator
from gluoncv.auto.tasks.utils import config_to_nested
from d8.image_classification import Dataset

if __name__ == '__main__':
    dataset = Dataset.get('brain-tumor')
    train_data, valid_data = dataset.split(0.8)

    config = {
        'estimator': ImageClassificationEstimator,
        'batch_size': 16,
        'epochs': 5
    }
    config = config_to_nested(config)
    config.pop('estimator')

    estimator = ImageClassificationEstimator(config)
    estimator.fit(train_data, valid_data)