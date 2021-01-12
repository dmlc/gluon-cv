import os
import argparse
import logging

import autogluon.core as ag
from gluoncv.auto.tasks import ObjectDetection
from gluoncv.auto.estimators import SSDEstimator
from d8.object_detection import Dataset

os.environ['MXNET_ENABLE_GPU_P2P'] = '0'


if __name__ == '__main__':
    # user defined arguments
    parser = argparse.ArgumentParser(description='benchmark for object detection')
    parser.add_argument('--dataset', type=str, default='sheep', help='dataset name')
    parser.add_argument('--num-trials', type=int, default=3, help='number of training trials')
    args = parser.parse_args()
    logging.info('user defined arguments: {}'.format(args))

    # specify hyperparameter search space
    config = {
        'task': 'ssd',
        'dataset': args.dataset,
        'estimator': 'ssd',
        'base_network': None,
        'transfer': ag.Categorical('ssd_512_vgg16_atrous_coco',
                                   'ssd_300_resnet34_v1b_coco',
                                   'ssd_512_resnet50_v1_coco',
                                   'ssd_512_resnet101_v2_voc'),
        'lr': ag.Real(1e-4, 1e-2, log=True),
        'batch_size': ag.Int(3, 6),  # [8, 16, 32, 64]
        'momentum': ag.Real(0.85, 0.95),
        'wd': ag.Real(1e-6, 1e-2, log=True),
        'epochs': 20,
        'num_trials': args.num_trials,
        'search_strategy': 'bayesopt'
    }

    # specify learning task
    task = ObjectDetection(config)

    # specify dataset
    dataset = Dataset.get(args.dataset)
    train_data, valid_data = dataset.split(0.8)

    # fit auto estimator
    detector = task.fit(train_data, valid_data)

    # evaluate auto estimator
    eval_map = detector.evaluate(valid_data)
    logging.info('evaluation: mAP={}'.format(eval_map[-1][-1]))

    # save and load auto estimator
    detector.save('ssd_detector.pkl')
    detector = ObjectDetection.load('ssd_detector.pkl')
