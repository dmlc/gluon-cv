import os
import argparse
import logging

import autogluon.core as ag
from gluoncv.auto.tasks import ImageClassification
from gluoncv.auto.estimators import ImageClassificationEstimator
from d8.image_classification import Dataset

os.environ['MXNET_ENABLE_GPU_P2P'] = '0'


if __name__ == '__main__':
    # user defined arguments
    parser = argparse.ArgumentParser(description='benchmark for image classification')
    parser.add_argument('--dataset', type=str, default='boat', help='dataset name')
    parser.add_argument('--num-trials', type=int, default=3, help='number of training trials')
    args = parser.parse_args()
    logging.info('user defined arguments: {}'.format(args))

    # specify hyperparameter search space
    config = {
        'task': 'classification',
        'dataset': args.dataset,
        'estimator': 'img_cls',
        'model': ag.Categorical('resnet50_v1', 'resnet101_v1',
                                'resnet50_v2', 'resnet101_v2',
                                'resnet50_v1b', 'resnet101_v1b',
                                'resnest50', 'resnest101'),
        'lr': ag.Real(1e-4, 1e-2, log=True),
        'batch_size': ag.Int(4, 7),  # [16, 32, 64, 128]
        'momentum': ag.Real(0.85, 0.95),
        'wd': ag.Real(1e-6, 1e-2, log=True),
        'epochs': 15,
        'num_trials': args.num_trials,
        'search_strategy': 'bayesopt'
    }

    # specify learning task
    task = ImageClassification(config)

    # specify dataset
    dataset = Dataset.get(args.dataset)
    train_data, valid_data = dataset.split(0.8)

    # fit auto estimator
    classifier = task.fit(train_data, valid_data)

    # evaluate auto estimator
    top1, top5 = classifier.evaluate(valid_data)
    logging.info('evaluation: top1={}, top5={}'.format(top1, top5))

    # save and load auto estimator
    classifier.save('classifier.pkl')
    classifier = ImageClassification.load('classifier.pkl')
