"""Utils for auto tasks"""
import copy
import warnings
import numpy as np

import autogluon.core as ag

from ... import data as gdata
from ..estimators.base_estimator import BaseEstimator
from ..estimators import SSDEstimator, FasterRCNNEstimator, YOLOv3Estimator, CenterNetEstimator
from .dataset import ObjectDetectionDataset

def auto_suggest(config, estimator, logger):
    """
    Automatically suggest some hyperparameters based on the dataset statistics.
    """
    if estimator is None:
        estimator = [SSDEstimator, FasterRCNNEstimator, YOLOv3Estimator, CenterNetEstimator]
    elif isinstance(estimator, (tuple, list)):
        pass
    else:
        assert issubclass(estimator, BaseEstimator)
        estimator = [estimator]
    config['estimator'] = ag.Categorical(*estimator)

    # get dataset statistics
    # user needs to define a Dataset object "train_dataset" when using custom dataset
    train_dataset = config.get('train_dataset', None)
    print(train_dataset)
    try:
        if train_dataset is None:
            dataset_name = config.get('dataset', 'voc')
            dataset_root = config.get('dataset_root', '~/.mxnet/datasets/')
            if dataset_name == 'voc':
                train_dataset = gdata.VOCDetection(splits=[(2007, 'trainval'), (2012, 'trainval')])
            elif dataset_name == 'voc_tiny':
                train_dataset = gdata.CustomVOCDetectionBase(classes=('motorbike',),
                                                             root=dataset_root + 'tiny_motorbike',
                                                             splits=[('', 'trainval')])
            elif dataset_name == 'coco':
                train_dataset = gdata.COCODetection(splits=['instances_train2017'])
        elif isinstance(train_dataset, ObjectDetectionDataset):
            train_dataset = train_dataset.to_mxnet()
        else:
            logger.info('Unknown dataset, quit auto suggestion...')
            return
    except Exception as e:
        logger.info(f'Unexpected error: {e}, quit auto suggestion...')
        return

    # choose 100 examples to calculate average statistics
    num_examples = 100
    image_size_list = []
    num_objects_list = []
    bbox_size_list = []
    bbox_rel_size_list = []

    for i in range(num_examples):
        train_image, train_label = train_dataset[i]

        image_height = train_image.shape[0]
        image_width = train_image.shape[1]
        image_size = image_height * image_width
        image_size_list.append(image_size)

        bounding_boxes = train_label[:, :4]
        num_objects = bounding_boxes.shape[0]
        bbox_height = bounding_boxes[:, 3] - bounding_boxes[:, 1]
        bbox_width = bounding_boxes[:, 2] - bounding_boxes[:, 0]
        bbox_size = bbox_height * bbox_width
        bbox_rel_size = bbox_size / image_size
        num_objects_list.append(num_objects)
        bbox_size_list.append(np.mean(bbox_size))
        bbox_rel_size_list.append(np.mean(bbox_rel_size))

    num_images = len(train_dataset)
    image_size = np.mean(image_size_list)
    num_classes = len(train_dataset.CLASSES)
    num_objects = np.mean(num_objects_list)
    bbox_size = np.mean(bbox_size_list)
    bbox_rel_size = np.mean(bbox_rel_size_list)

    logger.info("Printing dataset statistics")
    logger.info("number of training images: %d", num_images)
    logger.info("average image size: %.2f", image_size)
    logger.info("number of total object classes: %d", num_classes)
    logger.info("average number of objects in an image: %.2f", num_objects)
    logger.info("average bounding box size: %.2f", bbox_size)
    logger.info("average bounding box relative size: %.2f", bbox_rel_size)

    # specify 3 parts of config: data preprocessing, model selection, training settings
    if bbox_rel_size < 0.2 or num_objects > 5:
        suggested_estimator = [FasterRCNNEstimator]
    else:
        suggested_estimator = [SSDEstimator, YOLOv3Estimator]

    config['lr'] = config.get('lr', ag.Categorical(1e-2, 5e-3, 1e-3, 5e-4, 1e-4))

    # estimator setting
    if estimator is None:
        estimator = suggested_estimator
    elif isinstance(estimator, (tuple, list)):
        pass
    else:
        assert issubclass(estimator, BaseEstimator)
        estimator = [estimator]
    config['estimator'] = ag.Categorical(*estimator)

def config_to_nested(config):
    """Convert config to nested version"""
    if 'meta_arch' not in config:
        if config['estimator'] == SSDEstimator:
            config['meta_arch'] = 'ssd'
        elif config['estimator'] == FasterRCNNEstimator:
            config['meta_arch'] = 'faster_rcnn'
        elif config['estimator'] == YOLOv3Estimator:
            config['meta_arch'] = 'yolo3'
        elif config['estimator'] == CenterNetEstimator:
            config['meta_arch'] = 'center_net'
        else:
            config['meta_arch'] = None
    else:
        pass

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
            'yolo3': ['backbone', 'filters', 'anchors', 'strides', 'data_shape', 'syncbn', 'no_random_shape',
                      'amp', 'custom_model'],
            'train': ['batch_size', 'epochs', 'start_epoch', 'lr', 'lr_mode', 'lr_decay', 'lr_decay_period',
                      'lr_decay_epoch', 'warmup_lr', 'warmup_epochs', 'momentum', 'wd', 'log_interval',
                      'seed', 'num_samples', 'no_wd', 'mixup', 'no_mixup_epochs', 'label_smooth'],
            'validation': ['val_interval']
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
        raise NotImplementedError('%s is not implemented.' % config['meta_arch'])

    nested_config = {}

    for k, v in config.items():
        if k in config_mapping[config['meta_arch']]:
            if config['meta_arch'] not in nested_config:
                nested_config[config['meta_arch']] = {}
            nested_config[config['meta_arch']].update({k: v})
        elif k in config_mapping['train']:
            if 'train' not in nested_config:
                nested_config['train'] = {}
            nested_config['train'].update({k: v})
        elif k in config_mapping['validation']:
            if 'validation' not in nested_config:
                nested_config['validation'] = {}
            nested_config['validation'].update({k: v})
        else:
            nested_config.update({k: v})

    return nested_config

def recursive_update(total_config, config):
    """update config recursively"""
    for k, v in config.items():
        if isinstance(v, dict):
            if k not in total_config:
                total_config[k] = {}
            recursive_update(total_config[k], v)
        else:
            total_config[k] = v

def config_to_space(config):
    """Convert config to ag.space"""
    space = ag.Dict()
    for k, v in config.items():
        if isinstance(v, dict):
            if k not in space:
                space[k] = ag.Dict()
            space[k] = config_to_space(v)
        else:
            space[k] = v
    return space

def auto_args(config, estimators):
    """
    Merge user defined config to estimator default config, and convert to search space

    Parameters
    ----------
    config: <class 'dict'>

    Returns
    -------
    ag_space: <class 'autogluon.core.space.Dict'>
    """
    total_config = {}

    # estimator default config
    if not isinstance(estimators, (tuple, list)):
        estimators = [estimators]
    for estimator in estimators:
        assert issubclass(estimator, BaseEstimator), estimator
        default_config = copy.deepcopy(estimator._default_config)  # <class 'dict'>
        recursive_update(total_config, default_config)

    # user defined config
    nested_config = config_to_nested(config)
    recursive_update(total_config, nested_config)

    # convert to search space
    ag_space = config_to_space(total_config)

    return ag_space
