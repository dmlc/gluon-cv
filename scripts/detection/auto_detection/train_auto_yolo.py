import os
import argparse
import logging

import autogluon.core as ag
from gluoncv.auto.tasks import ObjectDetection
from gluoncv.auto.estimators import YOLOv3Estimator
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
        'task': 'yolo3',
        'dataset': args.dataset,
        'estimator': 'yolo3',
        'base_network': None,
        'transfer': ag.Categorical('yolo3_darknet53_voc',
                                   'yolo3_darknet53_coco'),
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
    detector.save('yolo_detector.pkl')
    detector = ObjectDetection.load('yolo_detector.pkl')
