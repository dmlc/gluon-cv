import os
import argparse
import logging

import cv2
import autogluon.core as ag
from gluoncv.auto.tasks import ObjectDetection
from gluoncv.auto.estimators import CenterNetEstimator
from d8.object_detection import Dataset

os.environ['MXNET_ENABLE_GPU_P2P'] = '0'


if __name__ == '__main__':
    # disable parallel computation for opencv
    cv2.setNumThreads(0)

    # user defined arguments
    parser = argparse.ArgumentParser(description='benchmark for object detection')
    parser.add_argument('--dataset', type=str, default='sheep', help='dataset name')
    parser.add_argument('--num-trials', type=int, default=3, help='number of training trials')
    args = parser.parse_args()
    logging.info('user defined arguments: {}'.format(args))

    # specify hyperparameter search space
    config = {
        'task': 'center_net',
        'dataset': args.dataset,
        'estimator': 'center_net',
        'base_network': None,
        'transfer': ag.Categorical('center_net_resnet18_v1b_coco',
                                   'center_net_resnet50_v1b_coco',
                                   'center_net_resnet101_v1b_coco'),
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
    detector.save('center_net_detector.pkl')
    detector = ObjectDetection.load('center_net_detector.pkl')
