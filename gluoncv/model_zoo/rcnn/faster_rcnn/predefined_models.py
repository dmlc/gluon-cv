"""Predefined Faster RCNN Model."""
from __future__ import absolute_import

import warnings

import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon.contrib.nn import SyncBatchNorm

from ..faster_rcnn import get_faster_rcnn
from ....nn.feature import FPNFeatureExpander

__all__ = ['faster_rcnn_resnet50_v1b_voc',
           'faster_rcnn_resnet50_v1b_coco',
           'faster_rcnn_fpn_resnet50_v1b_coco',
           'faster_rcnn_fpn_syncbn_resnet50_v1b_coco',
           'faster_rcnn_fpn_syncbn_resnest50_coco',
           'faster_rcnn_resnet50_v1b_custom',
           'faster_rcnn_resnet101_v1d_voc',
           'faster_rcnn_resnet101_v1d_coco',
           'faster_rcnn_fpn_resnet101_v1d_coco',
           'faster_rcnn_fpn_syncbn_resnet101_v1d_coco',
           'faster_rcnn_fpn_syncbn_resnest101_coco',
           'faster_rcnn_resnet101_v1d_custom',
           'faster_rcnn_fpn_syncbn_resnest269_coco']


def faster_rcnn_resnet50_v1b_voc(pretrained=False, pretrained_base=True, **kwargs):
    r"""Faster RCNN model from the paper
    "Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster r-cnn: Towards
    real-time object detection with region proposal networks"

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
    >>> model = get_faster_rcnn_resnet50_v1b_voc(pretrained=True)
    >>> print(model)
    """
    from ....model_zoo.resnetv1b import resnet50_v1b
    from ....data import VOCDetection
    classes = VOCDetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet50_v1b(pretrained=pretrained_base, dilated=False,
                                use_global_stats=True, **kwargs)
    features = nn.HybridSequential()
    top_features = nn.HybridSequential()
    for layer in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']:
        features.add(getattr(base_network, layer))
    for layer in ['layer4']:
        top_features.add(getattr(base_network, layer))
    train_patterns = '|'.join(['.*dense', '.*rpn', '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv'])
    return get_faster_rcnn(
        name='resnet50_v1b', dataset='voc', pretrained=pretrained,
        features=features, top_features=top_features, classes=classes,
        short=600, max_size=1000, train_patterns=train_patterns,
        nms_thresh=0.3, nms_topk=400, post_nms=100,
        roi_mode='align', roi_size=(14, 14), strides=16, clip=None,
        rpn_channel=1024, base_size=16, scales=(2, 4, 8, 16, 32),
        ratios=(0.5, 1, 2), alloc_size=(128, 128), rpn_nms_thresh=0.7,
        rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
        rpn_test_pre_nms=6000, rpn_test_post_nms=300, rpn_min_size=16,
        num_sample=128, pos_iou_thresh=0.5, pos_ratio=0.25, max_num_gt=100,
        **kwargs)


def faster_rcnn_resnet50_v1b_coco(pretrained=False, pretrained_base=True, **kwargs):
    r"""Faster RCNN model from the paper
    "Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster r-cnn: Towards
    real-time object detection with region proposal networks"

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
    >>> model = get_faster_rcnn_resnet50_v1b_coco(pretrained=True)
    >>> print(model)
    """
    from ....model_zoo.resnetv1b import resnet50_v1b
    from ....data import COCODetection
    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet50_v1b(pretrained=pretrained_base, dilated=False,
                                use_global_stats=True, **kwargs)
    features = nn.HybridSequential()
    top_features = nn.HybridSequential()
    for layer in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']:
        features.add(getattr(base_network, layer))
    for layer in ['layer4']:
        top_features.add(getattr(base_network, layer))
    train_patterns = '|'.join(['.*dense', '.*rpn', '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv'])
    return get_faster_rcnn(
        name='resnet50_v1b', dataset='coco', pretrained=pretrained,
        features=features, top_features=top_features, classes=classes,
        short=800, max_size=1333, train_patterns=train_patterns,
        nms_thresh=0.5, nms_topk=-1, post_nms=-1,
        roi_mode='align', roi_size=(14, 14), strides=16, clip=4.14,
        rpn_channel=1024, base_size=16, scales=(2, 4, 8, 16, 32),
        ratios=(0.5, 1, 2), alloc_size=(128, 128), rpn_nms_thresh=0.7,
        rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
        rpn_test_pre_nms=6000, rpn_test_post_nms=1000, rpn_min_size=1,
        num_sample=128, pos_iou_thresh=0.5, pos_ratio=0.25,
        max_num_gt=100, **kwargs)


def faster_rcnn_fpn_resnet50_v1b_coco(pretrained=False, pretrained_base=True, **kwargs):
    r"""Faster RCNN model with FPN from the paper
    "Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster r-cnn: Towards
    real-time object detection with region proposal networks"
    "Lin, T., Dollar, P., Girshick, R., He, K., Hariharan, B., Belongie, S. (2016).
    Feature Pyramid Networks for Object Detection"

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
    >>> model = get_faster_rcnn_fpn_resnet50_v1b_coco(pretrained=True)
    >>> print(model)
    """
    from ....model_zoo.resnetv1b import resnet50_v1b
    from ....data import COCODetection
    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet50_v1b(pretrained=pretrained_base, dilated=False,
                                use_global_stats=True, **kwargs)
    features = FPNFeatureExpander(
        network=base_network,
        outputs=['layers1_relu8_fwd', 'layers2_relu11_fwd', 'layers3_relu17_fwd',
                 'layers4_relu8_fwd'], num_filters=[256, 256, 256, 256], use_1x1=True,
        use_upsample=True, use_elewadd=True, use_p6=True, no_bias=False, pretrained=pretrained_base)
    top_features = None
    # 2 FC layer before RCNN cls and reg
    box_features = nn.HybridSequential()
    for _ in range(2):
        box_features.add(nn.Dense(1024, weight_initializer=mx.init.Normal(0.01)))
        box_features.add(nn.Activation('relu'))

    train_patterns = '|'.join(
        ['.*dense', '.*rpn', '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv', 'P'])
    return get_faster_rcnn(
        name='fpn_resnet50_v1b', dataset='coco', pretrained=pretrained, features=features,
        top_features=top_features, classes=classes, box_features=box_features,
        short=800, max_size=1333, min_stage=2, max_stage=6, train_patterns=train_patterns,
        nms_thresh=0.5, nms_topk=-1, post_nms=-1, roi_mode='align', roi_size=(7, 7),
        strides=(4, 8, 16, 32, 64), clip=4.14, rpn_channel=1024, base_size=16,
        scales=(2, 4, 8, 16, 32), ratios=(0.5, 1, 2), alloc_size=(384, 384),
        rpn_nms_thresh=0.7, rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
        rpn_test_pre_nms=6000, rpn_test_post_nms=1000, rpn_min_size=1, num_sample=512,
        pos_iou_thresh=0.5, pos_ratio=0.25, max_num_gt=100, **kwargs)


def faster_rcnn_fpn_syncbn_resnet50_v1b_coco(pretrained=False, pretrained_base=True, num_devices=0,
                                             **kwargs):
    r"""Faster RCNN model with FPN from the paper
    "Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster r-cnn: Towards
    real-time object detection with region proposal networks"
    "Lin, T., Dollar, P., Girshick, R., He, K., Hariharan, B., Belongie, S. (2016).
    Feature Pyramid Networks for Object Detection"

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `Ture`, this has no effect.
    num_devices : int, default is 0
        Number of devices for sync batch norm layer. if less than 1, use all devices available.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_faster_rcnn_fpn_syncbn_resnet50_v1b_coco(pretrained=True)
    >>> print(model)
    """
    from ....model_zoo.resnetv1b import resnet50_v1b
    from ....data import COCODetection
    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    gluon_norm_kwargs = {'num_devices': num_devices} if num_devices >= 1 else {}
    base_network = resnet50_v1b(pretrained=pretrained_base, dilated=False, use_global_stats=False,
                                norm_layer=SyncBatchNorm, norm_kwargs=gluon_norm_kwargs, **kwargs)
    sym_norm_kwargs = {'ndev': num_devices} if num_devices >= 1 else {}
    features = FPNFeatureExpander(
        network=base_network,
        outputs=['layers1_relu8_fwd', 'layers2_relu11_fwd', 'layers3_relu17_fwd',
                 'layers4_relu8_fwd'], num_filters=[256, 256, 256, 256], use_1x1=True,
        use_upsample=True, use_elewadd=True, use_p6=True, no_bias=True, pretrained=pretrained_base,
        norm_layer=mx.sym.contrib.SyncBatchNorm, norm_kwargs=sym_norm_kwargs)
    top_features = None
    # 1 Conv 1 FC layer before RCNN cls and reg
    box_features = nn.HybridSequential()
    box_features.add(nn.Conv2D(256, 3, padding=1, use_bias=False),
                     SyncBatchNorm(**gluon_norm_kwargs),
                     nn.Activation('relu'),
                     nn.Dense(1024, weight_initializer=mx.init.Normal(0.01)),
                     nn.Activation('relu'))

    train_patterns = '(?!.*moving)'  # excluding symbol bn moving mean and var
    return get_faster_rcnn(
        name='fpn_syncbn_resnet50_v1b', dataset='coco', pretrained=pretrained, features=features,
        top_features=top_features, classes=classes, box_features=box_features,
        short=(640, 800), max_size=1333, min_stage=2, max_stage=6, train_patterns=train_patterns,
        nms_thresh=0.5, nms_topk=-1, post_nms=-1, roi_mode='align', roi_size=(7, 7),
        strides=(4, 8, 16, 32, 64), clip=4.14, rpn_channel=256, base_size=16,
        scales=(2, 4, 8, 16, 32), ratios=(0.5, 1, 2), alloc_size=(384, 384),
        rpn_nms_thresh=0.7, rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
        rpn_test_pre_nms=6000, rpn_test_post_nms=1000, rpn_min_size=1, num_sample=512,
        pos_iou_thresh=0.5, pos_ratio=0.25, max_num_gt=100, **kwargs)


def faster_rcnn_fpn_syncbn_resnest50_coco(pretrained=False, pretrained_base=True, num_devices=0,
                                          **kwargs):
    r"""Faster R-CNN with ResNeSt
    ResNeSt: Split Attention Network"

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `Ture`, this has no effect.
    num_devices : int, default is 0
        Number of devices for sync batch norm layer. if less than 1, use all devices available.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_faster_rcnn_fpn_syncbn_resnest50_coco(pretrained=True)
    >>> print(model)
    """
    from ....model_zoo.resnest import resnest50
    from ....data import COCODetection
    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    gluon_norm_kwargs = {'num_devices': num_devices} if num_devices >= 1 else {}
    base_network = resnest50(pretrained=pretrained_base, dilated=False, use_global_stats=False,
                             norm_layer=SyncBatchNorm, norm_kwargs=gluon_norm_kwargs)
    from gluoncv.nn.dropblock import set_drop_prob
    from functools import partial
    apply_drop_prob = partial(set_drop_prob, 0.0)
    base_network.apply(apply_drop_prob)
    sym_norm_kwargs = {'ndev': num_devices} if num_devices >= 1 else {}
    features = FPNFeatureExpander(
        network=base_network,
        outputs=['layers1_relu11_fwd', 'layers2_relu15_fwd', 'layers3_relu23_fwd',
                 'layers4_relu11_fwd'], num_filters=[256, 256, 256, 256], use_1x1=True,
        use_upsample=True, use_elewadd=True, use_p6=True, no_bias=True, pretrained=pretrained_base,
        norm_layer=mx.sym.contrib.SyncBatchNorm, norm_kwargs=sym_norm_kwargs)
    top_features = None
    # 1 Conv 1 FC layer before RCNN cls and reg
    box_features = nn.HybridSequential()
    for _ in range(4):
        box_features.add(nn.Conv2D(256, 3, padding=1, use_bias=False),
                         SyncBatchNorm(**gluon_norm_kwargs),
                         nn.Activation('relu'))
    box_features.add(nn.Dense(1024, weight_initializer=mx.init.Normal(0.01)),
                     nn.Activation('relu'))

    train_patterns = '(?!.*moving)'  # excluding symbol bn moving mean and var
    return get_faster_rcnn(
        name='fpn_syncbn_resnest50', dataset='coco', pretrained=pretrained, features=features,
        top_features=top_features, classes=classes, box_features=box_features,
        short=(640, 800), max_size=1333, min_stage=2, max_stage=6, train_patterns=train_patterns,
        nms_thresh=0.5, nms_topk=-1, post_nms=-1, roi_mode='align', roi_size=(7, 7),
        strides=(4, 8, 16, 32, 64), clip=4.14, rpn_channel=256, base_size=16,
        scales=(2, 4, 8, 16, 32), ratios=(0.5, 1, 2), alloc_size=(384, 384),
        rpn_nms_thresh=0.7, rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
        rpn_test_pre_nms=6000, rpn_test_post_nms=1000, rpn_min_size=1, num_sample=512,
        pos_iou_thresh=0.5, pos_ratio=0.25, max_num_gt=100, **kwargs)


def faster_rcnn_resnet50_v1b_custom(classes, transfer=None, pretrained_base=True,
                                    pretrained=False, **kwargs):
    r"""Faster RCNN model with resnet50_v1b base network on custom dataset.

    Parameters
    ----------
    classes : iterable of str
        Names of custom foreground classes. `len(classes)` is the number of foreground classes.
    transfer : str or None
        If not `None`, will try to reuse pre-trained weights from faster RCNN networks trained
        on other datasets.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Returns
    -------
    mxnet.gluon.HybridBlock
        Hybrid faster RCNN network.
    """
    if pretrained:
        warnings.warn("Custom models don't provide `pretrained` weights, ignored.")
    if transfer is None:
        from ....model_zoo.resnetv1b import resnet50_v1b
        base_network = resnet50_v1b(pretrained=pretrained_base, dilated=False,
                                    use_global_stats=True, **kwargs)
        features = nn.HybridSequential()
        top_features = nn.HybridSequential()
        for layer in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']:
            features.add(getattr(base_network, layer))
        for layer in ['layer4']:
            top_features.add(getattr(base_network, layer))
        train_patterns = '|'.join(['.*dense', '.*rpn', '.*down(2|3|4)_conv',
                                   '.*layers(2|3|4)_conv'])
        return get_faster_rcnn(
            name='resnet50_v1b', dataset='custom', pretrained=pretrained,
            features=features, top_features=top_features, classes=classes,
            train_patterns=train_patterns, **kwargs)
    else:
        from ...model_zoo import get_model
        net = get_model('faster_rcnn_resnet50_v1b_' + str(transfer), pretrained=True, **kwargs)
        reuse_classes = [x for x in classes if x in net.classes]
        net.reset_class(classes, reuse_weights=reuse_classes)
    return net


def faster_rcnn_resnet101_v1d_voc(pretrained=False, pretrained_base=True, **kwargs):
    r"""Faster RCNN model from the paper
    "Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster r-cnn: Towards
    real-time object detection with region proposal networks"

    Parameters
    ----------
    pretrained : bool, optional, default is False
        Load pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_faster_rcnn_resnet101_v1d_voc(pretrained=True)
    >>> print(model)
    """
    from ....model_zoo.resnetv1b import resnet101_v1d
    from ....data import VOCDetection
    classes = VOCDetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet101_v1d(pretrained=pretrained_base, dilated=False,
                                 use_global_stats=True, **kwargs)
    features = nn.HybridSequential()
    top_features = nn.HybridSequential()
    for layer in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']:
        features.add(getattr(base_network, layer))
    for layer in ['layer4']:
        top_features.add(getattr(base_network, layer))
    train_patterns = '|'.join(['.*dense', '.*rpn', '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv'])
    return get_faster_rcnn(
        name='resnet101_v1d', dataset='voc', pretrained=pretrained,
        features=features, top_features=top_features, classes=classes,
        short=600, max_size=1000, train_patterns=train_patterns,
        nms_thresh=0.3, nms_topk=400, post_nms=100,
        roi_mode='align', roi_size=(14, 14), strides=16, clip=None,
        rpn_channel=1024, base_size=16, scales=(2, 4, 8, 16, 32),
        ratios=(0.5, 1, 2), alloc_size=(128, 128), rpn_nms_thresh=0.7,
        rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
        rpn_test_pre_nms=6000, rpn_test_post_nms=300, rpn_min_size=16,
        num_sample=128, pos_iou_thresh=0.5, pos_ratio=0.25, max_num_gt=100,
        **kwargs)


def faster_rcnn_resnet101_v1d_coco(pretrained=False, pretrained_base=True, **kwargs):
    r"""Faster RCNN model from the paper
    "Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster r-cnn: Towards
    real-time object detection with region proposal networks"

    Parameters
    ----------
    pretrained : bool, optional, default is False
        Load pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_faster_rcnn_resnet101_v1d_coco(pretrained=True)
    >>> print(model)
    """
    from ....model_zoo.resnetv1b import resnet101_v1d
    from ....data import COCODetection
    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet101_v1d(pretrained=pretrained_base, dilated=False,
                                 use_global_stats=True, **kwargs)
    features = nn.HybridSequential()
    top_features = nn.HybridSequential()
    for layer in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']:
        features.add(getattr(base_network, layer))
    for layer in ['layer4']:
        top_features.add(getattr(base_network, layer))
    train_patterns = '|'.join(['.*dense', '.*rpn', '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv'])
    return get_faster_rcnn(
        name='resnet101_v1d', dataset='coco', pretrained=pretrained,
        features=features, top_features=top_features, classes=classes,
        short=800, max_size=1333, train_patterns=train_patterns,
        nms_thresh=0.5, nms_topk=-1, post_nms=-1,
        roi_mode='align', roi_size=(14, 14), strides=16, clip=4.14,
        rpn_channel=1024, base_size=16, scales=(2, 4, 8, 16, 32),
        ratios=(0.5, 1, 2), alloc_size=(128, 128), rpn_nms_thresh=0.7,
        rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
        rpn_test_pre_nms=6000, rpn_test_post_nms=1000, rpn_min_size=1,
        num_sample=128, pos_iou_thresh=0.5, pos_ratio=0.25, max_num_gt=100,
        **kwargs)


def faster_rcnn_fpn_resnet101_v1d_coco(pretrained=False, pretrained_base=True, **kwargs):
    r"""Faster RCNN model with FPN from the paper
    "Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster r-cnn: Towards
    real-time object detection with region proposal networks"
    "Lin, T., Dollar, P., Girshick, R., He, K., Hariharan, B., Belongie, S. (2016).
    Feature Pyramid Networks for Object Detection"

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
    >>> model = get_faster_rcnn_fpn_resnet101_v1d_coco(pretrained=True)
    >>> print(model)
    """
    from ....model_zoo.resnetv1b import resnet101_v1d
    from ....data import COCODetection
    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet101_v1d(pretrained=pretrained_base, dilated=False,
                                 use_global_stats=True, **kwargs)
    features = FPNFeatureExpander(
        network=base_network,
        outputs=['layers1_relu8_fwd', 'layers2_relu11_fwd', 'layers3_relu68_fwd',
                 'layers4_relu8_fwd'], num_filters=[256, 256, 256, 256], use_1x1=True,
        use_upsample=True, use_elewadd=True, use_p6=True, no_bias=False, pretrained=pretrained_base)
    top_features = None
    # 2 FC layer before RCNN cls and reg
    box_features = nn.HybridSequential()
    for _ in range(2):
        box_features.add(nn.Dense(1024, weight_initializer=mx.init.Normal(0.01)))
        box_features.add(nn.Activation('relu'))

    train_patterns = '|'.join(
        ['.*dense', '.*rpn', '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv', 'P'])
    return get_faster_rcnn(
        name='fpn_resnet101_v1d', dataset='coco', pretrained=pretrained, features=features,
        top_features=top_features, classes=classes, box_features=box_features,
        short=800, max_size=1333, min_stage=2, max_stage=6, train_patterns=train_patterns,
        nms_thresh=0.5, nms_topk=-1, post_nms=-1, roi_mode='align', roi_size=(7, 7),
        strides=(4, 8, 16, 32, 64), clip=4.14, rpn_channel=1024, base_size=16,
        scales=(2, 4, 8, 16, 32), ratios=(0.5, 1, 2), alloc_size=(384, 384),
        rpn_nms_thresh=0.7, rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
        rpn_test_pre_nms=6000, rpn_test_post_nms=1000, rpn_min_size=1, num_sample=512,
        pos_iou_thresh=0.5, pos_ratio=0.25, max_num_gt=100, **kwargs)


def faster_rcnn_fpn_syncbn_resnet101_v1d_coco(pretrained=False, pretrained_base=True, num_devices=0,
                                              **kwargs):
    r"""Faster RCNN model with FPN from the paper
    "Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster r-cnn: Towards
    real-time object detection with region proposal networks"
    "Lin, T., Dollar, P., Girshick, R., He, K., Hariharan, B., Belongie, S. (2016).
    Feature Pyramid Networks for Object Detection"

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `Ture`, this has no effect.
    num_devices : int, default is 0
        Number of devices for sync batch norm layer. if less than 1, use all devices available.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_faster_rcnn_fpn_syncbn_resnet101_v1d_coco(pretrained=True)
    >>> print(model)
    """
    from ....model_zoo.resnetv1b import resnet101_v1d
    from ....data import COCODetection
    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    gluon_norm_kwargs = {'num_devices': num_devices} if num_devices >= 1 else {}
    base_network = resnet101_v1d(pretrained=pretrained_base, dilated=False, use_global_stats=False,
                                 norm_layer=SyncBatchNorm, norm_kwargs=gluon_norm_kwargs, **kwargs)
    sym_norm_kwargs = {'ndev': num_devices} if num_devices >= 1 else {}
    features = FPNFeatureExpander(
        network=base_network,
        outputs=['layers1_relu8_fwd', 'layers2_relu11_fwd', 'layers3_relu68_fwd',
                 'layers4_relu8_fwd'], num_filters=[256, 256, 256, 256], use_1x1=True,
        use_upsample=True, use_elewadd=True, use_p6=True, no_bias=True, pretrained=pretrained_base,
        norm_layer=mx.sym.contrib.SyncBatchNorm, norm_kwargs=sym_norm_kwargs)
    top_features = None
    # 1 Conv 1 FC layer before RCNN cls and reg
    box_features = nn.HybridSequential()
    for _ in range(4):
        box_features.add(nn.Conv2D(256, 3, padding=1, use_bias=False),
                         SyncBatchNorm(**gluon_norm_kwargs),
                         nn.Activation('relu'))
    box_features.add(nn.Dense(1024, weight_initializer=mx.init.Normal(0.01)),
                     nn.Activation('relu'))

    train_patterns = '(?!.*moving)'  # excluding symbol bn moving mean and var
    return get_faster_rcnn(
        name='fpn_syncbn_resnet101_v1d', dataset='coco', pretrained=pretrained, features=features,
        top_features=top_features, classes=classes, box_features=box_features,
        short=(640, 800), max_size=1333, min_stage=2, max_stage=6, train_patterns=train_patterns,
        nms_thresh=0.5, nms_topk=-1, post_nms=-1, roi_mode='align', roi_size=(7, 7),
        strides=(4, 8, 16, 32, 64), clip=4.14, rpn_channel=256, base_size=16,
        scales=(2, 4, 8, 16, 32), ratios=(0.5, 1, 2), alloc_size=(384, 384),
        rpn_nms_thresh=0.7, rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
        rpn_test_pre_nms=6000, rpn_test_post_nms=1000, rpn_min_size=1, num_sample=512,
        pos_iou_thresh=0.5, pos_ratio=0.25, max_num_gt=100, **kwargs)


def faster_rcnn_fpn_syncbn_resnest101_coco(pretrained=False, pretrained_base=True, num_devices=0,
                                           **kwargs):
    r"""Faster R-CNN with ResNeSt
    ResNeSt: Split Attention Network"

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `Ture`, this has no effect.
    num_devices : int, default is 0
        Number of devices for sync batch norm layer. if less than 1, use all devices available.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_faster_rcnn_fpn_syncbn_resnest101_coco(pretrained=True)
    >>> print(model)
    """
    from ....model_zoo.resnest import resnest101
    from ....data import COCODetection
    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    gluon_norm_kwargs = {'num_devices': num_devices} if num_devices >= 1 else {}
    base_network = resnest101(pretrained=pretrained_base, dilated=False, use_global_stats=False,
                              norm_layer=SyncBatchNorm, norm_kwargs=gluon_norm_kwargs)
    from gluoncv.nn.dropblock import set_drop_prob
    from functools import partial
    apply_drop_prob = partial(set_drop_prob, 0.0)
    base_network.apply(apply_drop_prob)
    sym_norm_kwargs = {'ndev': num_devices} if num_devices >= 1 else {}
    features = FPNFeatureExpander(
        network=base_network,
        outputs=['layers1_relu11_fwd', 'layers2_relu15_fwd', 'layers3_relu91_fwd',
                 'layers4_relu11_fwd'], num_filters=[256, 256, 256, 256], use_1x1=True,
        use_upsample=True, use_elewadd=True, use_p6=True, no_bias=True, pretrained=pretrained_base,
        norm_layer=mx.sym.contrib.SyncBatchNorm, norm_kwargs=sym_norm_kwargs)
    top_features = None
    # 1 Conv 1 FC layer before RCNN cls and reg
    box_features = nn.HybridSequential()
    for _ in range(4):
        box_features.add(nn.Conv2D(256, 3, padding=1, use_bias=False),
                         SyncBatchNorm(**gluon_norm_kwargs),
                         nn.Activation('relu'))
    box_features.add(nn.Dense(1024, weight_initializer=mx.init.Normal(0.01)),
                     nn.Activation('relu'))

    train_patterns = '(?!.*moving)'  # excluding symbol bn moving mean and var
    return get_faster_rcnn(
        name='fpn_syncbn_resnest101', dataset='coco', pretrained=pretrained, features=features,
        top_features=top_features, classes=classes, box_features=box_features,
        short=(640, 800), max_size=1333, min_stage=2, max_stage=6, train_patterns=train_patterns,
        nms_thresh=0.5, nms_topk=-1, post_nms=-1, roi_mode='align', roi_size=(7, 7),
        strides=(4, 8, 16, 32, 64), clip=4.14, rpn_channel=256, base_size=16,
        scales=(2, 4, 8, 16, 32), ratios=(0.5, 1, 2), alloc_size=(384, 384),
        rpn_nms_thresh=0.7, rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
        rpn_test_pre_nms=6000, rpn_test_post_nms=1000, rpn_min_size=1, num_sample=512,
        pos_iou_thresh=0.5, pos_ratio=0.25, max_num_gt=100, **kwargs)


def faster_rcnn_resnet101_v1d_custom(classes, transfer=None, pretrained_base=True,
                                     pretrained=False, **kwargs):
    r"""Faster RCNN model with resnet101_v1d base network on custom dataset.

    Parameters
    ----------
    classes : iterable of str
        Names of custom foreground classes. `len(classes)` is the number of foreground classes.
    transfer : str or None
        If not `None`, will try to reuse pre-trained weights from faster RCNN networks trained
        on other datasets.
    pretrained_base : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Returns
    -------
    mxnet.gluon.HybridBlock
        Hybrid faster RCNN network.
    """
    if pretrained:
        warnings.warn("Custom models don't provide `pretrained` weights, ignored.")
    if transfer is None:
        from ....model_zoo.resnetv1b import resnet101_v1d
        base_network = resnet101_v1d(pretrained=pretrained_base, dilated=False,
                                     use_global_stats=True, **kwargs)
        features = nn.HybridSequential()
        top_features = nn.HybridSequential()
        for layer in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']:
            features.add(getattr(base_network, layer))
        for layer in ['layer4']:
            top_features.add(getattr(base_network, layer))
        train_patterns = '|'.join(['.*dense', '.*rpn', '.*down(2|3|4)_conv',
                                   '.*layers(2|3|4)_conv'])
        return get_faster_rcnn(
            name='resnet101_v1d', dataset='custom', pretrained=pretrained,
            features=features, top_features=top_features, classes=classes,
            train_patterns=train_patterns, **kwargs)
    else:
        from ....model_zoo import get_model
        net = get_model('faster_rcnn_resnet101_v1d_' + str(transfer), pretrained=True, **kwargs)
        reuse_classes = [x for x in classes if x in net.classes]
        net.reset_class(classes, reuse_weights=reuse_classes)
    return net


def faster_rcnn_fpn_syncbn_resnest269_coco(pretrained=False, pretrained_base=True, num_devices=0,
                                           **kwargs):
    r"""Faster R-CNN with ResNeSt
    ResNeSt: Split Attention Network"

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `Ture`, this has no effect.
    num_devices : int, default is 0
        Number of devices for sync batch norm layer. if less than 1, use all devices available.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_faster_rcnn_fpn_syncbn_resnest269_coco(pretrained=True)
    >>> print(model)
    """
    from ....model_zoo.resnest import resnest269
    from ....data import COCODetection
    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    gluon_norm_kwargs = {'num_devices': num_devices} if num_devices >= 1 else {}
    base_network = resnest269(pretrained=pretrained_base, dilated=False, use_global_stats=False,
                              norm_layer=SyncBatchNorm, norm_kwargs=gluon_norm_kwargs)
    from gluoncv.nn.dropblock import set_drop_prob
    from functools import partial
    apply_drop_prob = partial(set_drop_prob, 0.0)
    base_network.apply(apply_drop_prob)
    sym_norm_kwargs = {'ndev': num_devices} if num_devices >= 1 else {}
    features = FPNFeatureExpander(
        network=base_network,
        outputs=['layers1_relu11_fwd', 'layers2_relu119_fwd', 'layers3_relu191_fwd',
                 'layers4_relu31_fwd'], num_filters=[256, 256, 256, 256], use_1x1=True,
        use_upsample=True, use_elewadd=True, use_p6=True, no_bias=True, pretrained=pretrained_base,
        norm_layer=mx.sym.contrib.SyncBatchNorm, norm_kwargs=sym_norm_kwargs)
    top_features = None
    # 1 Conv 1 FC layer before RCNN cls and reg
    box_features = nn.HybridSequential()
    for _ in range(4):
        box_features.add(nn.Conv2D(256, 3, padding=1, use_bias=False),
                         SyncBatchNorm(**gluon_norm_kwargs),
                         nn.Activation('relu'))
    box_features.add(nn.Dense(1024, weight_initializer=mx.init.Normal(0.01)),
                     nn.Activation('relu'))

    train_patterns = '(?!.*moving)'  # excluding symbol bn moving mean and var
    return get_faster_rcnn(
        name='fpn_syncbn_resnest269', dataset='coco', pretrained=pretrained, features=features,
        top_features=top_features, classes=classes, box_features=box_features,
        short=(640, 864), max_size=1440, min_stage=2, max_stage=6, train_patterns=train_patterns,
        nms_thresh=0.5, nms_topk=-1, post_nms=-1, roi_mode='align', roi_size=(7, 7),
        strides=(4, 8, 16, 32, 64), clip=4.14, rpn_channel=256, base_size=16,
        scales=(2, 4, 8, 16, 32), ratios=(0.5, 1, 2), alloc_size=(384, 384),
        rpn_nms_thresh=0.7, rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
        rpn_test_pre_nms=6000, rpn_test_post_nms=1000, rpn_min_size=1, num_sample=512,
        pos_iou_thresh=0.5, pos_ratio=0.25, max_num_gt=100, **kwargs)
