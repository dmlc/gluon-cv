import logging

import autogluon as ag
from autogluon.core.decorator import sample_config
from autogluon.scheduler.resource import get_cpu_count, get_gpu_count
from autogluon.task import BaseTask
from autogluon.utils import collect_params

from ..estimators.rcnn import FasterRCNNEstimator
from ... import utils as gutils

__all__ = ['ObjectDetection']


@ag.args()
def _train_object_detection(args, reporter):
    # fix seed for mxnet, numpy and python builtin random generator.
    gutils.random.seed(args.seed)

    # training contexts
    if args.meta_arch == 'yolo3':
        net_name = '_'.join((args.meta_arch, args.net, 'custom'))
    elif args.meta_arch == 'faster_rcnn':
        net_name = '_'.join(('custom', args.meta_arch, 'fpn'))
        kwargs = {'network': args.net, 'base_network_name': args.net,
                  'image_short': args.data_shape, 'max_size': 1000, 'nms_thresh': 0.5,
                  'nms_topk': -1, 'min_stage': 2, 'max_stage': 6, 'post_nms': -1,
                  'roi_mode': 'align', 'roi_size': (7, 7), 'strides': (4, 8, 16, 32, 64),
                  'clip': 4.14, 'rpn_channel': 256, 'anchor_scales': (2, 4, 8, 16, 32),
                  'anchor_aspect_ratio': (0.5, 1, 2), 'anchor_alloc_size': (384, 384),
                  'rpn_nms_thresh': 0.7, 'rpn_train_pre_nms': 12000, 'rpn_train_post_nms': 2000,
                  'rpn_test_pre_nms': 6000, 'rpn_test_post_nms': 1000, 'rpn_min_size': 1,
                  'per_device_batch_size': args.batch_size // args.num_gpus, 'num_sample': 512,
                  'rcnn_pos_iou_thresh': 0.5, 'rcnn_pos_ratio': 0.25, 'max_num_gt': 100,
                  'custom_model': True, 'no_pretrained_base': True, 'num_fpn_filters': 256,
                  'num_box_head_conv': 4, 'num_box_head_conv_filters': 256, 'amp': False,
                  'num_box_head_dense_filters': 1024, 'image_max_size': 1333, 'kv_store': 'nccl',
                  'anchor_base_size': 16, 'rcnn_num_samples': 512, 'rpn_smoothl1_rho': 0.001,
                  'rcnn_smoothl1_rho': 0.001, 'lr_warmup_factor': 1. / 3., 'lr_warmup': 500,
                  'executor_threads': 4, 'disable_hybridization': False, 'static_alloc': False}
        vars(args).update(kwargs)
    else:
        raise NotImplementedError(args.meta_arch, 'is not implemented.')

    if args.meta_arch == 'faster_rcnn':
        estimator = FasterRCNNEstimator(args, reporter=reporter)
    else:
        raise NotImplementedError('%s' % args.meta_arch)

    # training
    estimator.fit()

    if args.final_fit:
        return {'model_params': collect_params(estimator.net)}


class ObjectDetection(BaseTask):
    def __init__(self, config, logger=None):
        super(ObjectDetection, self).__init__()
        self._logger = logger if logger is not None else logging.getLogger(__name__)
        self._config = config
        nthreads_per_trial = get_cpu_count() if self._config.nthreads_per_trial > get_cpu_count() \
            else self._config.nthreads_per_trial
        if self._config.ngpus_per_trial > get_gpu_count():
            self._logger.warning(
                "The number of requested GPUs is greater than the number of available GPUs.")
        ngpus_per_trial = get_gpu_count() if self._config.ngpus_per_trial > get_gpu_count() \
            else self._config.ngpus_per_trial

        _train_object_detection.register_args(
            meta_arch=self._config.meta_arch, dataset=self._config.dataset, net=self._config.net,
            lr=self._config.lr, loss=self._config.loss, num_gpus=self._config.ngpus_per_trial,
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
            norm_layer='syncbn' if self._config.syncbn else None,
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
        self._logger.info('The best config: {}'.format(results['best_config']))

        estimator = FasterRCNNEstimator(best_config)
        estimator.load_parameters(results.pop('model_params'))
        return estimator
