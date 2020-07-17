import argparse
import logging

import autogluon as ag
from autogluon.core.space import Categorical
from mxnet import gluon

from gluoncv.auto.estimators.rcnn import FasterRCNNEstimator
from gluoncv.auto.tasks.object_detection import ObjectDetection

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='benchmark for object detection')
    parser.add_argument('--dataset-name', type=str, default='voc', help="dataset name")
    parser.add_argument('--dataset-root', type=str, default='./',
                        help="root path to the downloaded dataset, only for custom datastet")
    parser.add_argument('--dataset-format', type=str, default='voc', help="dataset format")
    parser.add_argument('--index-file-name-trainval', type=str, default='',
                        help="name of txt file which contains images for training and validation ")
    parser.add_argument('--index-file-name-test', type=str, default='',
                        help="name of txt file which contains images for testing")
    parser.add_argument('--classes', type=tuple, default=None, help="classes for custom classes")
    parser.add_argument('--no-redownload', action='store_true',
                        help="whether need to re-download dataset")
    parser.add_argument('--meta-arch', type=str, default='faster_rcnn',
                        choices=['yolo3', 'faster_rcnn'], help="Meta architecture of the model")

    args = parser.parse_args()
    logging.info('args: {}'.format(args))

    time_limits = 7 * 24 * 60 * 60  # 7 days
    epochs = 20
    # use coco pre-trained model for custom datasets
    transfer = None if ('voc' in args.dataset_name) or ('coco' in args.dataset_name) else 'coco'
    if args.meta_arch == 'yolo3':
        kwargs = {'num_trials': 30, 'epochs': epochs,
                  'net': ag.Categorical('darknet53', 'mobilenet1.0'), 'meta_arch': args.meta_arch,
                  'lr': ag.Categorical(1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5), 'transfer': transfer,
                  'data_shape': ag.Categorical(320, 416), 'nthreads_per_trial': 16,
                  'ngpus_per_trial': 8, 'batch_size': 64,
                  'lr_decay_epoch': ag.Categorical('80,90', '85,95'),
                  'warmup_epochs': ag.Int(1, 10), 'warmup_iters': ag.Int(250, 1000),
                  'wd': ag.Categorical(1e-4, 5e-4, 2.5e-4), 'syncbn': ag.Bool(),
                  'label_smooth': ag.Bool(), 'time_limits': time_limits, 'dist_ip_addrs': []}
    elif args.meta_arch == 'faster_rcnn':
        kwargs = {'num_trials': 30, 'epochs': ag.Categorical(30, 40, 50, 60),
                  'net': ag.Categorical('resnest101', 'resnest50'),
                  'meta_arch': args.meta_arch,
                  'lr': ag.Categorical(0.005, 0.002, 2e-4, 5e-4), 'transfer': transfer,
                  'data_shape': (640, 800), 'nthreads_per_trial': 16,
                  'ngpus_per_trial': 4, 'batch_size': 4,
                  'lr_decay_epoch': ag.Categorical([24, 28], [35], [50, 55], [40], [45], [55],
                                                   [30, 35], [20]),
                  'warmup_iters': ag.Int(5, 500),
                  'wd': ag.Categorical(1e-4, 5e-4, 2.5e-4), 'syncbn': ag.Bool(),
                  'label_smooth': False, 'time_limits': time_limits, 'dist_ip_addrs': []}
    else:
        raise NotImplementedError('%s is not implemented.', args.meta_arch)
    default_args = {'dataset': 'voc', 'net': Categorical('mobilenet1.0'), 'meta_arch': 'yolo3',
                    'lr': Categorical(5e-4, 1e-4), 'loss': gluon.loss.SoftmaxCrossEntropyLoss(),
                    'split_ratio': 0.8, 'batch_size': 16, 'epochs': 50, 'num_trials': 2,
                    'nthreads_per_trial': 12, 'num_workers': 16, 'ngpus_per_trial': 1,
                    'hybridize': True, 'search_strategy': 'random', 'search_options': {},
                    'time_limits': None, 'verbose': False, 'transfer': 'coco', 'resume': '',
                    'checkpoint': 'checkpoint/exp1.ag', 'visualizer': 'none', 'dist_ip_addrs': [],
                    'grace_period': None, 'auto_search': True, 'seed': 223, 'data_shape': 416,
                    'start_epoch': 0, 'lr_mode': 'step', 'lr_decay': 0.1, 'lr_decay_period': 0,
                    'lr_decay_epoch': '160,180', 'warmup_lr': 0.0, 'warmup_epochs': 2,
                    'warmup_iters': 1000, 'warmup_factor': 1. / 3., 'momentum': 0.9, 'wd': 0.0005,
                    'log_interval': 100, 'save_prefix': '', 'save_interval': 10, 'val_interval': 1,
                    'num_samples': -1, 'no_random_shape': False, 'no_wd': False, 'mixup': False,
                    'no_mixup_epochs': 20, 'label_smooth': False, 'syncbn': False,
                    'reuse_pred_weights': True, 'horovod': False}
    vars(args).update(default_args)
    vars(args).update(kwargs)
    task = ObjectDetection(args, FasterRCNNEstimator)
    estimator = task.fit()
    test_map = estimator.evaluate()
    print("mAP on test dataset: {}".format(test_map[-1][-1]))
    print(test_map)
    estimator.save('final_model.model')
