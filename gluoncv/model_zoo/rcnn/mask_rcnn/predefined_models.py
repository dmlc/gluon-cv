"""Predefined Mask RCNN Model."""
from __future__ import absolute_import

import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon.contrib.nn import SyncBatchNorm

from ..mask_rcnn import get_mask_rcnn
from ....nn.feature import FPNFeatureExpander

__all__ = ['mask_rcnn_resnet50_v1b_coco',
           'mask_rcnn_fpn_resnet50_v1b_coco',
           'mask_rcnn_resnet101_v1d_coco',
           'mask_rcnn_fpn_resnet101_v1d_coco',
           'mask_rcnn_resnet18_v1b_coco',
           'mask_rcnn_fpn_resnet18_v1b_coco',
           'mask_rcnn_fpn_syncbn_resnet18_v1b_coco',
           'mask_rcnn_fpn_syncbn_mobilenet1_0_coco']


def mask_rcnn_resnet50_v1b_coco(pretrained=False, pretrained_base=True, **kwargs):
    r"""Mask RCNN model from the paper
    "He, K., Gkioxari, G., Doll&ar, P., & Girshick, R. (2017). Mask R-CNN"

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = mask_rcnn_resnet50_v1b_coco(pretrained=True)
    >>> print(model)
    """
    from ....model_zoo.resnetv1b import resnet50_v1b
    from ....data import COCODetection
    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet50_v1b(pretrained=pretrained_base, dilated=False, use_global_stats=True)
    features = nn.HybridSequential()
    top_features = nn.HybridSequential()
    for layer in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']:
        features.add(getattr(base_network, layer))
    for layer in ['layer4']:
        top_features.add(getattr(base_network, layer))
    train_patterns = '|'.join(['.*dense', '.*rpn', '.*mask',
                               '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv'])
    return get_mask_rcnn(
        name='resnet50_v1b', dataset='coco', pretrained=pretrained,
        features=features, top_features=top_features, classes=classes,
        mask_channels=256, rcnn_max_dets=1000,
        short=800, max_size=1333, train_patterns=train_patterns,
        nms_thresh=0.5, nms_topk=-1, post_nms=-1,
        roi_mode='align', roi_size=(14, 14), strides=16, clip=4.42,
        rpn_channel=1024, base_size=16, scales=(2, 4, 8, 16, 32),
        ratios=(0.5, 1, 2), alloc_size=(128, 128), rpn_nms_thresh=0.7,
        rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
        rpn_test_pre_nms=6000, rpn_test_post_nms=1000, rpn_min_size=1,
        num_sample=128, pos_iou_thresh=0.5, pos_ratio=0.25,
        **kwargs)


def mask_rcnn_fpn_resnet50_v1b_coco(pretrained=False, pretrained_base=True, **kwargs):
    r"""Mask RCNN model from the paper
    "He, K., Gkioxari, G., Doll&ar, P., & Girshick, R. (2017). Mask R-CNN"

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = mask_rcnn_resnet50_v1b_coco(pretrained=True)
    >>> print(model)
    """
    from ....model_zoo.resnetv1b import resnet50_v1b
    from ....data import COCODetection
    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet50_v1b(pretrained=pretrained_base, dilated=False, use_global_stats=True)
    features = FPNFeatureExpander(
        network=base_network,
        outputs=['layers1_relu8_fwd', 'layers2_relu11_fwd', 'layers3_relu17_fwd',
                 'layers4_relu8_fwd'], num_filters=[256, 256, 256, 256], use_1x1=True,
        use_upsample=True, use_elewadd=True, use_p6=True, no_bias=False, pretrained=pretrained_base)
    top_features = None
    box_features = nn.HybridSequential()
    box_features.add(nn.AvgPool2D(pool_size=(3, 3), strides=2, padding=1))  # reduce to 7x7
    for _ in range(2):
        box_features.add(nn.Dense(1024, weight_initializer=mx.init.Normal(0.01)),
                         nn.Activation('relu'))
    train_patterns = '|'.join(['.*dense', '.*rpn', '.*mask', 'P',
                               '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv'])
    return get_mask_rcnn(
        name='fpn_resnet50_v1b', dataset='coco', pretrained=pretrained,
        features=features, top_features=top_features, classes=classes,
        box_features=box_features, mask_channels=256, rcnn_max_dets=1000,
        short=800, max_size=1333, min_stage=2, max_stage=6,
        train_patterns=train_patterns, nms_thresh=0.5, nms_topk=-1,
        post_nms=-1, roi_mode='align', roi_size=(14, 14),
        strides=(4, 8, 16, 32, 64), clip=4.42, rpn_channel=1024, base_size=16,
        scales=(2, 4, 8, 16, 32), ratios=(0.5, 1, 2), alloc_size=(384, 384),
        rpn_nms_thresh=0.7, rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
        rpn_test_pre_nms=6000, rpn_test_post_nms=1000, rpn_min_size=1,
        num_sample=512, pos_iou_thresh=0.5, pos_ratio=0.25, target_roi_scale=2,
        num_fcn_convs=4, **kwargs)


def mask_rcnn_resnet101_v1d_coco(pretrained=False, pretrained_base=True, **kwargs):
    r"""Mask RCNN model from the paper
    "He, K., Gkioxari, G., Doll&ar, P., & Girshick, R. (2017). Mask R-CNN"

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `Ture`, this has no effect.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = mask_rcnn_resnet101_v1d_coco(pretrained=True)
    >>> print(model)
    """
    from ....model_zoo.resnetv1b import resnet101_v1d
    from ....data import COCODetection
    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet101_v1d(pretrained=pretrained_base, dilated=False, use_global_stats=True)
    features = nn.HybridSequential()
    top_features = nn.HybridSequential()
    for layer in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']:
        features.add(getattr(base_network, layer))
    for layer in ['layer4']:
        top_features.add(getattr(base_network, layer))
    train_patterns = '|'.join(['.*dense', '.*rpn', '.*mask',
                               '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv'])
    return get_mask_rcnn(
        name='resnet101_v1d', dataset='coco', pretrained=pretrained,
        features=features, top_features=top_features, classes=classes,
        mask_channels=256, rcnn_max_dets=1000,
        short=800, max_size=1333, train_patterns=train_patterns,
        nms_thresh=0.5, nms_topk=-1, post_nms=-1,
        roi_mode='align', roi_size=(14, 14), strides=16, clip=4.42,
        rpn_channel=1024, base_size=16, scales=(2, 4, 8, 16, 32),
        ratios=(0.5, 1, 2), alloc_size=(128, 128), rpn_nms_thresh=0.7,
        rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
        rpn_test_pre_nms=6000, rpn_test_post_nms=1000, rpn_min_size=1,
        num_sample=128, pos_iou_thresh=0.5, pos_ratio=0.25,
        **kwargs)


def mask_rcnn_fpn_resnet101_v1d_coco(pretrained=False, pretrained_base=True, **kwargs):
    r"""Mask RCNN model from the paper
    "He, K., Gkioxari, G., Doll&ar, P., & Girshick, R. (2017). Mask R-CNN"

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = mask_rcnn_fpn_resnet101_v1d_coco(pretrained=True)
    >>> print(model)
    """
    from ....model_zoo.resnetv1b import resnet101_v1d
    from ....data import COCODetection
    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet101_v1d(pretrained=pretrained_base, dilated=False, use_global_stats=True)
    features = FPNFeatureExpander(
        network=base_network,
        outputs=['layers1_relu8_fwd', 'layers2_relu11_fwd', 'layers3_relu68_fwd',
                 'layers4_relu8_fwd'], num_filters=[256, 256, 256, 256], use_1x1=True,
        use_upsample=True, use_elewadd=True, use_p6=True, no_bias=False, pretrained=pretrained_base)
    top_features = None
    box_features = nn.HybridSequential()
    box_features.add(nn.AvgPool2D(pool_size=(3, 3), strides=2, padding=1))  # reduce to 7x7
    for _ in range(2):
        box_features.add(nn.Dense(1024, weight_initializer=mx.init.Normal(0.01)),
                         nn.Activation('relu'))
    train_patterns = '|'.join(['.*dense', '.*rpn', '.*mask', 'P',
                               '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv'])
    return get_mask_rcnn(
        name='fpn_resnet101_v1d', dataset='coco', pretrained=pretrained,
        features=features, top_features=top_features, classes=classes,
        box_features=box_features, mask_channels=256, rcnn_max_dets=1000,
        short=800, max_size=1333, min_stage=2, max_stage=6,
        train_patterns=train_patterns, nms_thresh=0.5, nms_topk=-1,
        post_nms=-1, roi_mode='align', roi_size=(14, 14),
        strides=(4, 8, 16, 32, 64), clip=4.42, rpn_channel=1024, base_size=16,
        scales=(2, 4, 8, 16, 32), ratios=(0.5, 1, 2), alloc_size=(384, 384),
        rpn_nms_thresh=0.7, rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
        rpn_test_pre_nms=6000, rpn_test_post_nms=1000, rpn_min_size=1,
        num_sample=512, pos_iou_thresh=0.5, pos_ratio=0.25, target_roi_scale=2,
        num_fcn_convs=4, **kwargs)


def mask_rcnn_resnet18_v1b_coco(pretrained=False, pretrained_base=True, rcnn_max_dets=1000,
                                rpn_test_pre_nms=6000, rpn_test_post_nms=1000, **kwargs):
    r"""Mask RCNN model from the paper
    "He, K., Gkioxari, G., Doll&ar, P., & Girshick, R. (2017). Mask R-CNN"

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    rcnn_max_dets : int, default is 1000
        Number of rois to retain in RCNN.
    rpn_test_pre_nms : int, default is 6000
        Filter top proposals before NMS in testing of RPN.
    rpn_test_post_nms : int, default is 300
        Return top proposal results after NMS in testing of RPN.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = mask_rcnn_resnet18_v1b_coco(pretrained=True)
    >>> print(model)
    """
    from ....model_zoo.resnetv1b import resnet18_v1b
    from ....data import COCODetection
    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    rcnn_max_dets = rpn_test_post_nms if rcnn_max_dets > rpn_test_post_nms else rcnn_max_dets
    base_network = resnet18_v1b(pretrained=pretrained_base, dilated=False, use_global_stats=True)
    features = nn.HybridSequential()
    top_features = nn.HybridSequential()
    for layer in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']:
        features.add(getattr(base_network, layer))
    for layer in ['layer4']:
        top_features.add(getattr(base_network, layer))
    train_patterns = '|'.join(['.*dense', '.*rpn', '.*mask',
                               '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv'])
    return get_mask_rcnn(
        name='resnet18_v1b', dataset='coco', pretrained=pretrained,
        features=features, top_features=top_features, classes=classes,
        mask_channels=256, rcnn_max_dets=rcnn_max_dets,
        short=800, max_size=1333, train_patterns=train_patterns,
        nms_thresh=0.5, nms_topk=-1, post_nms=-1,
        roi_mode='align', roi_size=(14, 14), strides=16, clip=4.42,
        rpn_channel=1024, base_size=16, scales=(2, 4, 8, 16, 32),
        ratios=(0.5, 1, 2), alloc_size=(128, 128), rpn_nms_thresh=0.7,
        rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
        rpn_test_pre_nms=rpn_test_pre_nms, rpn_test_post_nms=rpn_test_post_nms,
        rpn_min_size=1, num_sample=256, pos_iou_thresh=0.5, pos_ratio=0.25,
        **kwargs)


def mask_rcnn_fpn_resnet18_v1b_coco(pretrained=False, pretrained_base=True, rcnn_max_dets=1000,
                                    rpn_test_pre_nms=6000, rpn_test_post_nms=1000, **kwargs):
    r"""Mask RCNN model from the paper
    "He, K., Gkioxari, G., Doll&ar, P., & Girshick, R. (2017). Mask R-CNN"

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    rcnn_max_dets : int, default is 1000
        Number of rois to retain in RCNN.
    rpn_test_pre_nms : int, default is 6000
        Filter top proposals before NMS in testing of RPN.
    rpn_test_post_nms : int, default is 300
        Return top proposal results after NMS in testing of RPN.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = mask_rcnn_fpn_resnet18_v1b_coco(pretrained=True)
    >>> print(model)
    """
    from ....model_zoo.resnetv1b import resnet18_v1b
    from ....data import COCODetection
    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    rcnn_max_dets = rpn_test_post_nms if rcnn_max_dets > rpn_test_post_nms else rcnn_max_dets
    base_network = resnet18_v1b(pretrained=pretrained_base, dilated=False, use_global_stats=True)
    features = FPNFeatureExpander(
        network=base_network,
        outputs=['layers1_relu3_fwd', 'layers2_relu3_fwd', 'layers3_relu3_fwd',
                 'layers4_relu3_fwd'], num_filters=[256, 256, 256, 256], use_1x1=True,
        use_upsample=True, use_elewadd=True, use_p6=True, no_bias=False, pretrained=pretrained_base)
    top_features = None
    box_features = nn.HybridSequential()
    for _ in range(2):
        box_features.add(nn.Dense(1024, weight_initializer=mx.init.Normal(0.01)),
                         nn.Activation('relu'))
    train_patterns = '|'.join(['.*dense', '.*rpn', '.*mask', 'P',
                               '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv'])
    return get_mask_rcnn(
        name='fpn_resnet18_v1b', dataset='coco', pretrained=pretrained,
        features=features, top_features=top_features, classes=classes,
        box_features=box_features, mask_channels=256, rcnn_max_dets=rcnn_max_dets,
        short=800, max_size=1333, min_stage=2, max_stage=6,
        train_patterns=train_patterns, nms_thresh=0.5, nms_topk=-1,
        post_nms=-1, roi_mode='align', roi_size=(7, 7),
        strides=(4, 8, 16, 32, 64), clip=4.42, rpn_channel=1024, base_size=16,
        scales=(2, 4, 8, 16, 32), ratios=(0.5, 1, 2), alloc_size=(384, 384),
        rpn_nms_thresh=0.7, rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
        rpn_test_pre_nms=rpn_test_pre_nms, rpn_test_post_nms=rpn_test_post_nms,
        rpn_min_size=1, num_sample=512, pos_iou_thresh=0.5, pos_ratio=0.25,
        target_roi_scale=2, num_fcn_convs=2, **kwargs)


def mask_rcnn_fpn_syncbn_resnet18_v1b_coco(pretrained=False, pretrained_base=True, num_devices=0,
                                           rcnn_max_dets=1000, rpn_test_pre_nms=6000,
                                           rpn_test_post_nms=1000, **kwargs):
    r"""Mask RCNN model from the paper
    "He, K., Gkioxari, G., Doll&ar, P., & Girshick, R. (2017). Mask R-CNN"

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    num_devices : int, default is 0
        Number of devices for sync batch norm layer. if less than 1, use all devices available.
    rcnn_max_dets : int, default is 1000
        Number of rois to retain in RCNN.
    rpn_test_pre_nms : int, default is 6000
        Filter top proposals before NMS in testing of RPN.
    rpn_test_post_nms : int, default is 300
        Return top proposal results after NMS in testing of RPN.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = mask_rcnn_fpn_syncbn_resnet18_v1b_coco(pretrained=True)
    >>> print(model)
    """
    from ....model_zoo.resnetv1b import resnet18_v1b
    from ....data import COCODetection
    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    rcnn_max_dets = rpn_test_post_nms if rcnn_max_dets > rpn_test_post_nms else rcnn_max_dets
    gluon_norm_kwargs = {'num_devices': num_devices} if num_devices >= 1 else {}
    sym_norm_kwargs = {'ndev': num_devices} if num_devices >= 1 else {}
    base_network = resnet18_v1b(pretrained=pretrained_base, dilated=False, use_global_stats=False,
                                norm_layer=SyncBatchNorm, norm_kwargs=gluon_norm_kwargs, **kwargs)
    features = FPNFeatureExpander(
        network=base_network,
        outputs=['layers1_relu3_fwd', 'layers2_relu3_fwd', 'layers3_relu3_fwd',
                 'layers4_relu3_fwd'], num_filters=[256, 256, 256, 256], use_1x1=True,
        use_upsample=True, use_elewadd=True, use_p6=True, no_bias=False, pretrained=pretrained_base,
        norm_layer=mx.sym.contrib.SyncBatchNorm, norm_kwargs=sym_norm_kwargs)
    top_features = None
    box_features = nn.HybridSequential()
    box_features.add(nn.Conv2D(256, 3, padding=1),
                     SyncBatchNorm(**gluon_norm_kwargs),
                     nn.Activation('relu'),
                     nn.Dense(1024, weight_initializer=mx.init.Normal(0.01)),
                     nn.Activation('relu'))
    train_patterns = '(?!.*moving)'
    return get_mask_rcnn(
        name='fpn_syncbn_resnet18_v1b', dataset='coco', pretrained=pretrained,
        features=features, top_features=top_features, classes=classes,
        box_features=box_features, mask_channels=256, rcnn_max_dets=rcnn_max_dets,
        short=(640, 800), max_size=1333, min_stage=2, max_stage=6,
        train_patterns=train_patterns, nms_thresh=0.5, nms_topk=-1,
        post_nms=-1, roi_mode='align', roi_size=(7, 7),
        strides=(4, 8, 16, 32, 64), clip=4.42, rpn_channel=1024, base_size=16,
        scales=(2, 4, 8, 16, 32), ratios=(0.5, 1, 2), alloc_size=(384, 384),
        rpn_nms_thresh=0.7, rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
        rpn_test_pre_nms=rpn_test_pre_nms, rpn_test_post_nms=rpn_test_post_nms,
        rpn_min_size=1, num_sample=512, pos_iou_thresh=0.5, pos_ratio=0.25,
        target_roi_scale=2, num_fcn_convs=2, norm_layer=SyncBatchNorm,
        norm_kwargs=gluon_norm_kwargs, **kwargs)


def mask_rcnn_fpn_syncbn_mobilenet1_0_coco(pretrained=False, pretrained_base=True, num_devices=0,
                                           rcnn_max_dets=1000, rpn_test_pre_nms=6000,
                                           rpn_test_post_nms=1000, **kwargs):
    r"""Mask RCNN model from the paper
    "He, K., Gkioxari, G., Doll&ar, P., & Girshick, R. (2017). Mask R-CNN"

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    num_devices : int, default is 0
        Number of devices for sync batch norm layer. if less than 1, use all devices available.
    rcnn_max_dets : int, default is 1000
        Number of rois to retain in RCNN.
    rpn_test_pre_nms : int, default is 6000
        Filter top proposals before NMS in testing of RPN.
    rpn_test_post_nms : int, default is 300
        Return top proposal results after NMS in testing of RPN.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = mask_rcnn_fpn_syncbn_mobilenet1_0_coco(pretrained=True)
    >>> print(model)
    """
    from ....model_zoo.mobilenet import mobilenet1_0
    from ....data import COCODetection
    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    rcnn_max_dets = rpn_test_post_nms if rcnn_max_dets > rpn_test_post_nms else rcnn_max_dets
    gluon_norm_kwargs = {'num_devices': num_devices} if num_devices >= 1 else {}
    sym_norm_kwargs = {'ndev': num_devices} if num_devices >= 1 else {}
    base_network = mobilenet1_0(pretrained=pretrained_base, norm_layer=SyncBatchNorm,
                                norm_kwargs=gluon_norm_kwargs, **kwargs)
    features = FPNFeatureExpander(
        network=base_network,
        outputs=['relu6_fwd', 'relu10_fwd', 'relu22_fwd', 'relu26_fwd'],
        num_filters=[256, 256, 256, 256], use_1x1=True,
        use_upsample=True, use_elewadd=True, use_p6=True, no_bias=False, pretrained=pretrained_base,
        norm_layer=mx.sym.contrib.SyncBatchNorm, norm_kwargs=sym_norm_kwargs)
    top_features = None
    box_features = nn.HybridSequential()
    box_features.add(nn.AvgPool2D(pool_size=(3, 3), strides=2, padding=1))  # reduce to 7x7
    box_features.add(nn.Conv2D(256, 3, padding=1),
                     SyncBatchNorm(**gluon_norm_kwargs),
                     nn.Activation('relu'),
                     nn.Dense(1024, weight_initializer=mx.init.Normal(0.01)),
                     nn.Activation('relu'))
    train_patterns = '(?!.*moving)'
    return get_mask_rcnn(
        name='fpn_syncbn_mobilenet1_0', dataset='coco', pretrained=pretrained, features=features,
        top_features=top_features, classes=classes, box_features=box_features, mask_channels=256,
        rcnn_max_dets=rcnn_max_dets, short=(640, 800), max_size=1333, min_stage=2, max_stage=6,
        train_patterns=train_patterns, nms_thresh=0.5, nms_topk=-1, post_nms=-1, roi_mode='align',
        roi_size=(14, 14), strides=(4, 8, 16, 32, 64), clip=4.42, rpn_channel=1024, base_size=16,
        scales=(2, 4, 8, 16, 32), ratios=(0.5, 1, 2), alloc_size=(384, 384),
        rpn_nms_thresh=0.7, rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
        rpn_test_pre_nms=rpn_test_pre_nms, rpn_test_post_nms=rpn_test_post_nms, rpn_min_size=1,
        num_sample=512, pos_iou_thresh=0.5, pos_ratio=0.25, target_roi_scale=2, num_fcn_convs=2,
        norm_layer=SyncBatchNorm, norm_kwargs=gluon_norm_kwargs, **kwargs)
