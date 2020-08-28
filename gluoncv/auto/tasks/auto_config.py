"""Utils for auto configs"""
import copy
import collections.abc

import autogluon as ag

from ..estimators.base_estimator import BaseEstimator

def config_to_nested(config):
    if config['meta_arch'] == 'ssd':
        config_mapping = {
            'ssd': ['backbone', 'data_shape', 'filters', 'sizes', 'ratios', 'steps', 'syncbn',
                    'amp', 'custom_model'],
            'train': ['batch_size', 'start_epoch', 'epochs', 'lr', 'lr_decay', 'lr_decay_epoch',
                      'momentum', 'wd', 'log_interval', 'seed', 'dali'],
            'validation': ['val_interval']
        }
    elif config['meta_arch'] == 'faster_rcnn':
        config_mapping = {
            'faster_rcnn': ['backbone', 'nms_thresh', 'nms_topk', 'roi_mode', 'roi_size', 'strides', 'clip',
                            'anchor_base_size', 'anchor_aspect_ratio', 'anchor_scales', 'anchor_alloc_size',
                            'rpn_channel', 'rpn_nms_thresh', 'max_num_gt', 'norm_layer', 'use_fpn', 'num_fpn_filters',
                            'num_box_head_conv', 'num_box_head_conv_filters', 'num_box_head_dense_filters',
                            'image_short', 'image_max_size', 'custom_model', 'amp', 'static_alloc'],
            'train': ['pretrained_base', 'batch_size', 'start_epoch', 'epochs', 'lr', 'lr_decay',
                      'lr_decay_epoch', 'lr_mode', 'lr_warmup', 'lr_warmup_factor', 'momentum', 'wd',
                      'rpn_train_pre_nms', 'rpn_train_post_nms', 'rpn_smoothl1_rho', 'rpn_min_size',
                      'rcnn_num_samples', 'rcnn_pos_iou_thresh', 'rcnn_pos_ratio', 'rcnn_smoothl1_rho',
                      'log_interval', 'seed', 'verbose', 'mixup', 'no_mixup_epochs', 'executor_threads'],
            'validation': ['rpn_test_pre_nms', 'rpn_test_post_nms', 'val_interval']
        }
    elif config['meta_arch'] == 'yolo3':
        config_mapping = {
            'yolo3': ['base_network', 'scale', 'topk', 'root', 'wh_weight', 'center_reg_weight',
                      'data_shape', 'syncbn'],
            'train': ['dataset', 'pretrained_base', 'gpus', 'num_workers', 'resume', 'batch_size',
                      'epochs', 'start_epoch', 'lr', 'lr_mode', 'lr_decay', 'lr_decay_period', 'lr_decay_epoch',
                      'warmup_lr', 'warmup_epochs', 'momentum', 'wd', 'log_interval', 'save_interval', 'save_prefix',
                      'seed', 'num_samples', 'no_random_shape', 'no_wd', 'mixup', 'no_mixup_epochs', 'label_smooth',
                      'amp', 'horovod'],
            'validation': ['val_interval', 'test']
        }
    elif config['meta_arch'] == 'center_net':
        config_mapping = {
            'center_net': ['base_network', 'heads', 'scale', 'topk', 'root', 'wh_weight', 'center_reg_weight',
                           'data_shape'],
            'train': ['gpus', 'pretrained_base', 'batch_size', 'epochs', 'lr', 'lr_decay', 'lr_decay_epoch',
                      'lr_mode', 'warmup_lr', 'warmup_epochs', 'num_workers', 'resume', 'auto_resume',
                      'start_epoch', 'momentum', 'wd', 'save_interval', 'log_interval'],
            'validation': ['flip_test', 'nms_thresh', 'nms_topk', 'post_nms', 'num_workers',
                           'batch_size', 'interval']
        }
    else:
        raise NotImplementedError(config['meta_arch'], 'is not implemented.')

    nested_config = {}

    for k, v in config.items():
        if k in config_mapping[config['meta_arch']]:
            if config['meta_arch'] not in nested_config.keys():
                nested_config[config['meta_arch']] = {}
            nested_config[config['meta_arch']].update({k: v})
        elif k in config_mapping['train']:
            if 'train' not in nested_config.keys():
                nested_config['train'] = {}
            nested_config['train'].update({k: v})
        elif k in config_mapping['validation']:
            if 'validation' not in nested_config.keys():
                nested_config['validation'] = {}
            nested_config['validation'].update({k: v})
        else:
            nested_config.update({k: v})

    return nested_config

def config_to_space(config):
    space = ag.Dict()
    for k, v in config.items():
        if isinstance(v, dict):
            if k not in space.keys():
                space[k] = ag.Dict()
            space[k] = config_to_space(v)
        else:
            space.update({k: v})
    return space

def cfg_to_space(cfg, space):
    for k, v in cfg.items():
        if isinstance(v, dict):
            if k not in space.keys():
                space[k] = ag.Dict()
            cfg_to_space(v, space[k])
        else:
            space[k] = v

def recursive_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def auto_args(estimators, config):
    """Merge updated config to default, and convert to search space"""
    config = config_to_nested(config)
    if not isinstance(estimators, (tuple, list)):
        estimators = [estimators]
    _cfg = {}
    for estimator in estimators:
        assert issubclass(estimator, BaseEstimator), estimator
        cfg = copy.deepcopy(estimator._default_config)
        recursive_update(_cfg, cfg)
    # user custom search space
    recursive_update(_cfg, config)
    ag_space = ag.Dict()
    cfg_to_space(_cfg, ag_space)
    return ag_space
