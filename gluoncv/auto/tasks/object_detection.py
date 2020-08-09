import logging

import autogluon as ag
from autogluon.core.decorator import sample_config
from autogluon.scheduler.resource import get_cpu_count, get_gpu_count
from autogluon.task import BaseTask
from autogluon.utils import collect_params

from ... import utils as gutils
from ...model_zoo import vgg16_atrous_300, vgg16_atrous_512
from ..estimators.base_estimator import ConfigDict


__all__ = ['ObjectDetection']

@ag.args()
def _train_object_detection(args, reporter):
    # fix seed for mxnet, numpy and python builtin random generator.
    gutils.random.seed(args.seed)

    # training contexts
    if args.meta_arch == 'faster_rcnn':
        config = {'dataset': args.dataset, 'gpus': [0, 1, 2, 3], 'resume': '', 'save_prefix': '',
                  'save_interval': 1, 'horovod': False, 'num_workers': 16, 'kv_store': 'nccl',
                  'disable_hybridization': False, 'seed': 826994795,
                  'train': {'pretrained_base': True, 'batch_size': args.batch_size,
                            'start_epoch': 0, 'epochs': args.epochs, 'lr': args.lr, 'lr_decay': 0.1,
                            'lr_decay_epoch': args.lr_decay_epoch, 'lr_mode': 'step',
                            'lr_warmup': 500, 'lr_warmup_factor': 0.3333333333333333,
                            'momentum': 0.9, 'wd': 0.0001, 'rpn_train_pre_nms': 12000,
                            'rpn_train_post_nms': 2000, 'rpn_smoothl1_rho': 0.001,
                            'rpn_min_size': 1, 'rcnn_num_samples': 512,
                            'rcnn_pos_iou_thresh': 0.5, 'rcnn_pos_ratio': 0.25,
                            'rcnn_smoothl1_rho': 0.001, 'log_interval': 100, 'seed': 233,
                            'verbose': False, 'mixup': False, 'no_mixup_epochs': 20,
                            'executor_threads': 4},
                  'validation': {'rpn_test_pre_nms': 6000, 'rpn_test_post_nms': 1000,
                                 'val_interval': 1},
                  'faster_rcnn': {'backbone': args.net, 'nms_thresh': 0.5, 'nms_topk': -1,
                                  'roi_mode': 'align', 'roi_size': [7, 7],
                                  'strides': [4, 8, 16, 32, 64], 'clip': 4.14,
                                  'anchor_base_size': 16, 'anchor_aspect_ratio': [0.5, 1, 2],
                                  'anchor_scales': [2, 4, 8, 16, 32],
                                  'anchor_alloc_size': [384, 384], 'rpn_channel': 256,
                                  'rpn_nms_thresh': 0.7, 'max_num_gt': 100,
                                  'norm_layer': args.norm_layer, 'use_fpn': True,
                                  'num_fpn_filters': 256, 'num_box_head_conv': 4,
                                  'num_box_head_conv_filters': 256,
                                  'num_box_head_dense_filters': 1024, 'image_short': (640, 800),
                                  'image_max_size': 1333, 'custom_model': True, 'amp': False,
                                  'static_alloc': False}}
        # vars(args).update(kwargs)
    elif args.meta_arch == 'ssd':
        config = {'dataset': args.dataset, 'dataset_root': '~/.mxnet/datasets/', 'gpus': [0, 1, 2, 3, 4, 5, 6, 7], 'resume': '',
                  'save_prefix': '', 'save_interval': 1, 'horovod': False, 'num_workers': 16, 'seed': 826994795,
                  'train': {'batch_size': args.batch_size, 'start_epoch': 0, 'epochs': args.epochs,
                            'lr': args.lr, 'lr_decay': 0.1, 'lr_decay_epoch': args.lr_decay_epoch, 'lr_mode': 'step',
                            'momentum': 0.9, 'wd': 0.0005, 'log_interval': 100, 'seed': 233, 'dali': False},
                  'validation': {'val_interval': 1},
                  'ssd': {'backbone': args.net, 'data_shape': 300, 'features': vgg16_atrous_300,
                          'filters': None,
                          'sizes': [30, 60, 111, 162, 213, 264, 315],
                          'ratios': [[1, 2, 0.5]] + [[1, 2, 0.5, 3, 1.0/3]] * 3 + [[1, 2, 0.5]] * 2,
                          'steps': [8, 16, 32, 64, 100, 300],
                          'syncbn': False, 'amp': False}}
    else:
        raise NotImplementedError(args.meta_arch, 'is not implemented.')

    # disable auto_resume for HPO tasks
    if 'train' in config:
        config['train']['auto_resume'] = False
    else:
        config['train'] = {'auto_resume': False}

    try:
        if args.meta_arch == 'faster_rcnn' or 'ssd':
            estimator = args.estimator(config, reporter=reporter)
        else:
            raise NotImplementedError('%s' % args.meta_arch)
        # training
        estimator.fit()
    except Exception as e:
        return str(e)

    if args.final_fit:
        return {'model_params': collect_params(estimator.net)}

    return {}


class ObjectDetection(BaseTask):
    def __init__(self, config, estimator, logger=None):
        super(ObjectDetection, self).__init__()
        self._logger = logger if logger is not None else logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)
        self._estimator = estimator
        #self._config = config
        self._config = ConfigDict(config)
        nthreads_per_trial = get_cpu_count() if self._config.nthreads_per_trial > get_cpu_count() \
            else self._config.nthreads_per_trial
        if self._config.ngpus_per_trial > get_gpu_count():
            self._logger.warning(
                "The number of requested GPUs is greater than the number of available GPUs.")
        ngpus_per_trial = get_gpu_count() if self._config.ngpus_per_trial > get_gpu_count() \
            else self._config.ngpus_per_trial

        _train_object_detection.register_args(
            meta_arch=self._config.meta_arch, dataset=self._config.dataset, net=self._config.net,
            lr=self._config.lr,  num_gpus=self._config.ngpus_per_trial,
            batch_size=self._config.batch_size, split_ratio=self._config.split_ratio,
            epochs=self._config.epochs, num_workers=self._config.nthreads_per_trial,
            hybridize=self._config.hybridize, verbose=self._config.verbose, final_fit=False,
            seed=self._config.seed, data_shape=self._config.data_shape, start_epoch=0,
            transfer=self._config.transfer, lr_mode=self._config.lr_mode,
            lr_decay=self._config.lr_decay, lr_decay_period=self._config.lr_decay_period,
            lr_decay_epoch=self._config.lr_decay_epoch, warmup_lr=self._config.warmup_lr,
            warmup_epochs=self._config.warmup_epochs, warmup_iters=self._config.warmup_iters,
            warmup_factor=self._config.warmup_factor, momentum=self._config.momentum,
            wd=self._config.wd, log_interval=self._config.log_interval,
            save_prefix=self._config.save_prefix, save_interval=self._config.save_interval,
            val_interval=self._config.val_interval, num_samples=self._config.num_samples,
            no_random_shape=self._config.no_random_shape, no_wd=self._config.no_wd,
            mixup=self._config.mixup, no_mixup_epochs=self._config.no_mixup_epochs,
            label_smooth=self._config.label_smooth, resume=self._config.resume,
            syncbn=self._config.syncbn, reuse_pred_weights=self._config.reuse_pred_weights,
            horovod=self._config.horovod, gpus='0,1,2,3,4,5,6,7', use_fpn=True,
            norm_layer='syncbn' if self._config.syncbn else None, estimator=self._estimator
        )

        self._config.scheduler_options = {
            'resource': {'num_cpus': nthreads_per_trial, 'num_gpus': ngpus_per_trial},
            'checkpoint': self._config.checkpoint,
            'num_trials': self._config.num_trials,
            'time_out': self._config.time_limits,
            'resume': self._config.resume,
            'visualizer': self._config.visualizer,
            'time_attr': 'epoch',
            'reward_attr': 'map_reward',
            'dist_ip_addrs': self._config.dist_ip_addrs,
            'searcher': self._config.search_strategy,
            'search_options': self._config.search_options,
        }
        if self._config.search_strategy == 'hyperband':
            self._config.scheduler_options.update({
                'searcher': 'random',
                'max_t': self._config.epochs,
                'grace_period': self._config.grace_period if self._config.grace_period
                else self._config.epochs // 4})

    def fit(self):
        results = self.run_fit(_train_object_detection, self._config.search_strategy,
                               self._config.scheduler_options)
        self._logger.info(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> finish model fitting")
        best_config = sample_config(_train_object_detection.args, results['best_config'])
        self._logger.info('The best config: {}'.format(best_config))

        estimator = self._estimator(best_config)
        estimator.put_parameters(results.pop('model_params'))
        return estimator
