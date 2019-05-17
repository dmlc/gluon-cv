"""SSD predefined models."""
from __future__ import absolute_import

import warnings
from .ssd import get_ssd
from .vgg_atrous import vgg16_atrous_300, vgg16_atrous_512
from ...data import VOCDetection

__all__ = ['ssd_300_vgg16_atrous_voc',
           'ssd_300_vgg16_atrous_coco',
           'ssd_300_vgg16_atrous_custom',
           'ssd_512_vgg16_atrous_voc',
           'ssd_512_vgg16_atrous_coco',
           'ssd_512_vgg16_atrous_custom',
           'ssd_512_resnet18_v1_voc',
           'ssd_512_resnet18_v1_coco',
           'ssd_512_resnet18_v1_custom',
           'ssd_512_resnet50_v1_voc',
           'ssd_512_resnet50_v1_coco',
           'ssd_512_resnet50_v1_custom',
           'ssd_512_resnet101_v2_voc',
           'ssd_512_resnet152_v2_voc',
           'ssd_512_mobilenet1_0_voc',
           'ssd_512_mobilenet1_0_coco',
           'ssd_512_mobilenet1_0_custom',
           'ssd_300_mobilenet0_25_voc',
           'ssd_300_mobilenet0_25_coco',
           'ssd_300_mobilenet0_25_custom']

def ssd_300_vgg16_atrous_voc(pretrained=False, pretrained_base=True, **kwargs):
    """SSD architecture with VGG16 atrous 300x300 base network for Pascal VOC.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized.

    Returns
    -------
    HybridBlock
        A SSD detection network.
    """
    classes = VOCDetection.CLASSES
    net = get_ssd('vgg16_atrous', 300, features=vgg16_atrous_300, filters=None,
                  sizes=[30, 60, 111, 162, 213, 264, 315],
                  ratios=[[1, 2, 0.5]] + [[1, 2, 0.5, 3, 1.0/3]] * 3 + [[1, 2, 0.5]] * 2,
                  steps=[8, 16, 32, 64, 100, 300],
                  classes=classes, dataset='voc', pretrained=pretrained,
                  pretrained_base=pretrained_base, **kwargs)
    return net

def ssd_300_vgg16_atrous_coco(pretrained=False, pretrained_base=True, **kwargs):
    """SSD architecture with VGG16 atrous 300x300 base network for COCO.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized.

    Returns
    -------
    HybridBlock
        A SSD detection network.
    """
    from ...data import COCODetection
    classes = COCODetection.CLASSES
    net = get_ssd('vgg16_atrous', 300, features=vgg16_atrous_300, filters=None,
                  sizes=[21, 45, 99, 153, 207, 261, 315],
                  ratios=[[1, 2, 0.5]] + [[1, 2, 0.5, 3, 1.0/3]] * 3 + [[1, 2, 0.5]] * 2,
                  steps=[8, 16, 32, 64, 100, 300],
                  classes=classes, dataset='coco', pretrained=pretrained,
                  pretrained_base=pretrained_base, **kwargs)
    return net

def ssd_300_vgg16_atrous_custom(classes, pretrained_base=True, pretrained=False,
                                transfer=None, **kwargs):
    """SSD architecture with VGG16 atrous 300x300 base network for COCO.

    Parameters
    ----------
    classes : iterable of str
        Names of custom foreground classes. `len(classes)` is the number of foreground classes.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized.
    transfer : str or None
        If not `None`, will try to reuse pre-trained weights from SSD networks trained on other
        datasets.

    Returns
    -------
    HybridBlock
        A SSD detection network.

    Example
    -------
    >>> net = ssd_300_vgg16_atrous_custom(classes=['a', 'b', 'c'], pretrained_base=True)
    >>> net = ssd_300_vgg16_atrous_custom(classes=['foo', 'bar'], transfer='coco')

    """
    if pretrained:
        warnings.warn("Custom models don't provide `pretrained` weights, ignored.")
    if transfer is None:
        kwargs['pretrained'] = False
        net = get_ssd('vgg16_atrous', 300, features=vgg16_atrous_300, filters=None,
                      sizes=[21, 45, 99, 153, 207, 261, 315],
                      ratios=[[1, 2, 0.5]] + [[1, 2, 0.5, 3, 1.0/3]] * 3 + [[1, 2, 0.5]] * 2,
                      steps=[8, 16, 32, 64, 100, 300],
                      classes=classes, dataset='',
                      pretrained_base=pretrained_base, **kwargs)
    else:
        from ...model_zoo import get_model
        net = get_model('ssd_300_vgg16_atrous_' + str(transfer), pretrained=True, **kwargs)
        reuse_classes = [x for x in classes if x in net.classes]
        net.reset_class(classes, reuse_weights=reuse_classes)
    return net

def ssd_512_vgg16_atrous_voc(pretrained=False, pretrained_base=True, **kwargs):
    """SSD architecture with VGG16 atrous 512x512 base network.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized.

    Returns
    -------
    HybridBlock
        A SSD detection network.
    """
    classes = VOCDetection.CLASSES
    net = get_ssd('vgg16_atrous', 512, features=vgg16_atrous_512, filters=None,
                  sizes=[51.2, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6],
                  ratios=[[1, 2, 0.5]] + [[1, 2, 0.5, 3, 1.0/3]] * 4 + [[1, 2, 0.5]] * 2,
                  steps=[8, 16, 32, 64, 128, 256, 512],
                  classes=classes, dataset='voc', pretrained=pretrained,
                  pretrained_base=pretrained_base, **kwargs)
    return net

def ssd_512_vgg16_atrous_coco(pretrained=False, pretrained_base=True, **kwargs):
    """SSD architecture with VGG16 atrous layers for COCO.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized.

    Returns
    -------
    HybridBlock
        A SSD detection network.
    """
    from ...data import COCODetection
    classes = COCODetection.CLASSES
    return get_ssd('vgg16_atrous', 512, features=vgg16_atrous_512, filters=None,
                   sizes=[51.2, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6],
                   ratios=[[1, 2, 0.5]] + [[1, 2, 0.5, 3, 1.0/3]] * 4 + [[1, 2, 0.5]] * 2,
                   steps=[8, 16, 32, 64, 128, 256, 512],
                   classes=classes, dataset='coco', pretrained=pretrained,
                   pretrained_base=pretrained_base, **kwargs)

def ssd_512_vgg16_atrous_custom(classes, pretrained_base=True, pretrained=False,
                                transfer=None, **kwargs):
    """SSD architecture with VGG16 atrous 300x300 base network for COCO.

    Parameters
    ----------
    classes : iterable of str
        Names of custom foreground classes. `len(classes)` is the number of foreground classes.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized.
    transfer : str or None
        If not `None`, will try to reuse pre-trained weights from SSD networks trained on other
        datasets.

    Returns
    -------
    HybridBlock
        A SSD detection network.

    Example
    -------
    >>> net = ssd_512_vgg16_atrous_custom(classes=['a', 'b', 'c'], pretrained_base=True)
    >>> net = ssd_512_vgg16_atrous_custom(classes=['foo', 'bar'], transfer='coco')

    """
    if pretrained:
        warnings.warn("Custom models don't provide `pretrained` weights, ignored.")
    if transfer is None:
        kwargs['pretrained'] = False
        net = get_ssd('vgg16_atrous', 512, features=vgg16_atrous_512, filters=None,
                      sizes=[51.2, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6],
                      ratios=[[1, 2, 0.5]] + [[1, 2, 0.5, 3, 1.0/3]] * 4 + [[1, 2, 0.5]] * 2,
                      steps=[8, 16, 32, 64, 128, 256, 512],
                      classes=classes, dataset='',
                      pretrained_base=pretrained_base, **kwargs)
    else:
        from ...model_zoo import get_model
        net = get_model('ssd_512_vgg16_atrous_' + str(transfer), pretrained=True, **kwargs)
        reuse_classes = [x for x in classes if x in net.classes]
        net.reset_class(classes, reuse_weights=reuse_classes)
    return net

def ssd_512_resnet18_v1_voc(pretrained=False, pretrained_base=True, **kwargs):
    """SSD architecture with ResNet v1 18 layers.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.

    Returns
    -------
    HybridBlock
        A SSD detection network.
    """
    classes = VOCDetection.CLASSES
    return get_ssd('resnet18_v1', 512,
                   features=['stage3_activation1', 'stage4_activation1'],
                   filters=[512, 512, 256, 256],
                   sizes=[51.2, 102.4, 189.4, 276.4, 363.52, 450.6, 492],
                   ratios=[[1, 2, 0.5]] + [[1, 2, 0.5, 3, 1.0/3]] * 3 + [[1, 2, 0.5]] * 2,
                   steps=[16, 32, 64, 128, 256, 512],
                   classes=classes, dataset='voc', pretrained=pretrained,
                   pretrained_base=pretrained_base, **kwargs)

def ssd_512_resnet18_v1_coco(pretrained=False, pretrained_base=True, **kwargs):
    """SSD architecture with ResNet v1 18 layers.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.

    Returns
    -------
    HybridBlock
        A SSD detection network.
    """
    from ...data import COCODetection
    classes = COCODetection.CLASSES
    return get_ssd('resnet18_v1', 512,
                   features=['stage3_activation1', 'stage4_activation1'],
                   filters=[512, 512, 256, 256],
                   sizes=[51.2, 102.4, 189.4, 276.4, 363.52, 450.6, 492],
                   ratios=[[1, 2, 0.5]] + [[1, 2, 0.5, 3, 1.0/3]] * 3 + [[1, 2, 0.5]] * 2,
                   steps=[16, 32, 64, 128, 256, 512],
                   classes=classes, dataset='coco', pretrained=pretrained,
                   pretrained_base=pretrained_base, **kwargs)

def ssd_512_resnet18_v1_custom(classes, pretrained_base=True, pretrained=False,
                               transfer=None, **kwargs):
    """SSD architecture with ResNet18 v1 512 base network for COCO.

    Parameters
    ----------
    classes : iterable of str
        Names of custom foreground classes. `len(classes)` is the number of foreground classes.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized.
    transfer : str or None
        If not `None`, will try to reuse pre-trained weights from SSD networks trained on other
        datasets.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.

    Returns
    -------
    HybridBlock
        A SSD detection network.

    Example
    -------
    >>> net = ssd_512_resnet18_v1_custom(classes=['a', 'b', 'c'], pretrained_base=True)
    >>> net = ssd_512_resnet18_v1_custom(classes=['foo', 'bar'], transfer='voc')

    """
    if pretrained:
        warnings.warn("Custom models don't provide `pretrained` weights, ignored.")
    if transfer is None:
        kwargs['pretrained'] = False
        net = get_ssd('resnet18_v1', 512,
                      features=['stage3_activation1', 'stage4_activation1'],
                      filters=[512, 512, 256, 256],
                      sizes=[51.2, 102.4, 189.4, 276.4, 363.52, 450.6, 492],
                      ratios=[[1, 2, 0.5]] + [[1, 2, 0.5, 3, 1.0/3]] * 3 + [[1, 2, 0.5]] * 2,
                      steps=[16, 32, 64, 128, 256, 512],
                      classes=classes, dataset='',
                      pretrained_base=pretrained_base, **kwargs)
    else:
        from ...model_zoo import get_model
        net = get_model('ssd_512_resnet18_v1_' + str(transfer), pretrained=True, **kwargs)
        reuse_classes = [x for x in classes if x in net.classes]
        net.reset_class(classes, reuse_weights=reuse_classes)
    return net

def ssd_512_resnet50_v1_voc(pretrained=False, pretrained_base=True, **kwargs):
    """SSD architecture with ResNet v1 50 layers.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.

    Returns
    -------
    HybridBlock
        A SSD detection network.
    """
    classes = VOCDetection.CLASSES
    return get_ssd('resnet50_v1', 512,
                   features=['stage3_activation5', 'stage4_activation2'],
                   filters=[512, 512, 256, 256],
                   sizes=[51.2, 102.4, 189.4, 276.4, 363.52, 450.6, 492],
                   ratios=[[1, 2, 0.5]] + [[1, 2, 0.5, 3, 1.0/3]] * 3 + [[1, 2, 0.5]] * 2,
                   steps=[16, 32, 64, 128, 256, 512],
                   classes=classes, dataset='voc', pretrained=pretrained,
                   pretrained_base=pretrained_base, **kwargs)

def ssd_512_resnet50_v1_coco(pretrained=False, pretrained_base=True, **kwargs):
    """SSD architecture with ResNet v1 50 layers for COCO.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.

    Returns
    -------
    HybridBlock
        A SSD detection network.
    """
    from ...data import COCODetection
    classes = COCODetection.CLASSES
    return get_ssd('resnet50_v1', 512,
                   features=['stage3_activation5', 'stage4_activation2'],
                   filters=[512, 512, 256, 256],
                   sizes=[51.2, 133.12, 215.04, 296.96, 378.88, 460.8, 542.72],
                   ratios=[[1, 2, 0.5]] + [[1, 2, 0.5, 3, 1.0/3]] * 3 + [[1, 2, 0.5]] * 2,
                   steps=[16, 32, 64, 128, 256, 512],
                   classes=classes, dataset='coco', pretrained=pretrained,
                   pretrained_base=pretrained_base, **kwargs)

def ssd_512_resnet50_v1_custom(classes, pretrained_base=True, pretrained=False,
                               transfer=None, **kwargs):
    """SSD architecture with ResNet50 v1 512 base network for custom dataset.

    Parameters
    ----------
    classes : iterable of str
        Names of custom foreground classes. `len(classes)` is the number of foreground classes.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized.
    transfer : str or None
        If not `None`, will try to reuse pre-trained weights from SSD networks trained on other
        datasets.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.

    Returns
    -------
    HybridBlock
        A SSD detection network.

    Example
    -------
    >>> net = ssd_512_resnet50_v1_custom(classes=['a', 'b', 'c'], pretrained_base=True)
    >>> net = ssd_512_resnet50_v1_custom(classes=['foo', 'bar'], transfer='voc')

    """
    if pretrained:
        warnings.warn("Custom models don't provide `pretrained` weights, ignored.")
    if transfer is None:
        kwargs['pretrained'] = False
        net = get_ssd('resnet50_v1', 512,
                      features=['stage3_activation5', 'stage4_activation2'],
                      filters=[512, 512, 256, 256],
                      sizes=[51.2, 133.12, 215.04, 296.96, 378.88, 460.8, 542.72],
                      ratios=[[1, 2, 0.5]] + [[1, 2, 0.5, 3, 1.0/3]] * 3 + [[1, 2, 0.5]] * 2,
                      steps=[16, 32, 64, 128, 256, 512],
                      classes=classes, dataset='',
                      pretrained_base=pretrained_base, **kwargs)
    else:
        from ...model_zoo import get_model
        net = get_model('ssd_512_resnet50_v1_' + str(transfer), pretrained=True, **kwargs)
        reuse_classes = [x for x in classes if x in net.classes]
        net.reset_class(classes, reuse_weights=reuse_classes)
    return net

def ssd_512_resnet101_v2_voc(pretrained=False, pretrained_base=True, **kwargs):
    """SSD architecture with ResNet v2 101 layers.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.

    Returns
    -------
    HybridBlock
        A SSD detection network.
    """
    classes = VOCDetection.CLASSES
    return get_ssd('resnet101_v2', 512,
                   features=['stage3_activation22', 'stage4_activation2'],
                   filters=[512, 512, 256, 256],
                   sizes=[51.2, 102.4, 189.4, 276.4, 363.52, 450.6, 492],
                   ratios=[[1, 2, 0.5]] + [[1, 2, 0.5, 3, 1.0/3]] * 3 + [[1, 2, 0.5]] * 2,
                   steps=[16, 32, 64, 128, 256, 512],
                   classes=classes, dataset='voc', pretrained=pretrained,
                   pretrained_base=pretrained_base, **kwargs)

def ssd_512_resnet152_v2_voc(pretrained=False, pretrained_base=True, **kwargs):
    """SSD architecture with ResNet v2 152 layers.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.

    Returns
    -------
    HybridBlock
        A SSD detection network.
    """
    classes = VOCDetection.CLASSES
    return get_ssd('resnet152_v2', 512,
                   features=['stage2_activation7', 'stage3_activation35', 'stage4_activation2'],
                   filters=[512, 512, 256, 256],
                   sizes=[51.2, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6],
                   ratios=[[1, 2, 0.5]] + [[1, 2, 0.5, 3, 1.0/3]] * 4 + [[1, 2, 0.5]] * 2,
                   steps=[8, 16, 32, 64, 128, 256, 512],
                   classes=classes, dataset='voc', pretrained=pretrained,
                   pretrained_base=pretrained_base, **kwargs)

def ssd_512_mobilenet1_0_voc(pretrained=False, pretrained_base=True, **kwargs):
    """SSD architecture with mobilenet1.0 base networks.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.

    Returns
    -------
    HybridBlock
        A SSD detection network.
    """
    classes = VOCDetection.CLASSES
    return get_ssd('mobilenet1.0', 512,
                   features=['relu22_fwd', 'relu26_fwd'],
                   filters=[512, 512, 256, 256],
                   sizes=[51.2, 102.4, 189.4, 276.4, 363.52, 450.6, 492],
                   ratios=[[1, 2, 0.5]] + [[1, 2, 0.5, 3, 1.0/3]] * 3 + [[1, 2, 0.5]] * 2,
                   steps=[16, 32, 64, 128, 256, 512],
                   classes=classes, dataset='voc', pretrained=pretrained,
                   pretrained_base=pretrained_base, **kwargs)

def ssd_512_mobilenet1_0_coco(pretrained=False, pretrained_base=True, **kwargs):
    """SSD architecture with mobilenet1.0 base networks for COCO.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.

    Returns
    -------
    HybridBlock
        A SSD detection network.
    """
    from ...data import COCODetection
    classes = COCODetection.CLASSES
    return get_ssd('mobilenet1.0', 512,
                   features=['relu22_fwd', 'relu26_fwd'],
                   filters=[512, 512, 256, 256],
                   sizes=[51.2, 102.4, 189.4, 276.4, 363.52, 450.6, 492],
                   ratios=[[1, 2, 0.5]] + [[1, 2, 0.5, 3, 1.0/3]] * 3 + [[1, 2, 0.5]] * 2,
                   steps=[16, 32, 64, 128, 256, 512],
                   classes=classes, dataset='coco', pretrained=pretrained,
                   pretrained_base=pretrained_base, **kwargs)

def ssd_512_mobilenet1_0_custom(classes, pretrained_base=True, pretrained=False,
                                transfer=None, **kwargs):
    """SSD architecture with mobilenet1.0 512 base network for custom dataset.

    Parameters
    ----------
    classes : iterable of str
        Names of custom foreground classes. `len(classes)` is the number of foreground classes.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized.
    transfer : str or None
        If not `None`, will try to reuse pre-trained weights from SSD networks trained on other
        datasets.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.

    Returns
    -------
    HybridBlock
        A SSD detection network.

    Example
    -------
    >>> net = ssd_512_mobilenet1_0_custom(classes=['a', 'b', 'c'], pretrained_base=True)
    >>> net = ssd_512_mobilenet1_0_custom(classes=['foo', 'bar'], transfer='voc')

    """
    if pretrained:
        warnings.warn("Custom models don't provide `pretrained` weights, ignored.")
    if transfer is None:
        kwargs['pretrained'] = False
        net = get_ssd('mobilenet1.0', 512,
                      features=['relu22_fwd', 'relu26_fwd'],
                      filters=[512, 512, 256, 256],
                      sizes=[51.2, 102.4, 189.4, 276.4, 363.52, 450.6, 492],
                      ratios=[[1, 2, 0.5]] + [[1, 2, 0.5, 3, 1.0/3]] * 3 + [[1, 2, 0.5]] * 2,
                      steps=[16, 32, 64, 128, 256, 512],
                      classes=classes, dataset='',
                      pretrained_base=pretrained_base, **kwargs)
    else:
        from ...model_zoo import get_model
        net = get_model('ssd_512_mobilenet1.0_' + str(transfer), pretrained=True, **kwargs)
        reuse_classes = [x for x in classes if x in net.classes]
        net.reset_class(classes, reuse_weights=reuse_classes)
    return net

def ssd_300_mobilenet0_25_voc(pretrained=False, pretrained_base=True, **kwargs):
    """SSD architecture with mobilenet0.25 base networks.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.

    Returns
    -------
    HybridBlock
        A SSD detection network.
    """
    classes = VOCDetection.CLASSES
    return get_ssd('mobilenet0.25', 300,
                   features=['relu22_fwd', 'relu26_fwd'],
                   filters=[256, 256, 128, 128],
                   sizes=[21, 45, 99, 153, 207, 261, 315],
                   ratios=[[1, 2, 0.5]] + [[1, 2, 0.5, 3, 1.0/3]] * 3 + [[1, 2, 0.5]] * 2,
                   steps=[8, 16, 32, 64, 100, 300],
                   classes=classes, dataset='voc', pretrained=pretrained,
                   pretrained_base=pretrained_base, **kwargs)

def ssd_300_mobilenet0_25_coco(pretrained=False, pretrained_base=True, **kwargs):
    """SSD architecture with mobilenet0.25 base networks for COCO.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.

    Returns
    -------
    HybridBlock
        A SSD detection network.
    """
    from ...data import COCODetection
    classes = COCODetection.CLASSES
    return get_ssd('mobilenet0.25', 300,
                   features=['relu22_fwd', 'relu26_fwd'],
                   filters=[256, 256, 128, 128],
                   sizes=[21, 45, 99, 153, 207, 261, 315],
                   ratios=[[1, 2, 0.5]] + [[1, 2, 0.5, 3, 1.0/3]] * 3 + [[1, 2, 0.5]] * 2,
                   steps=[8, 16, 32, 64, 100, 300],
                   classes=classes, dataset='coco', pretrained=pretrained,
                   pretrained_base=pretrained_base, **kwargs)

def ssd_300_mobilenet0_25_custom(classes, pretrained_base=True, pretrained=False,
                                 transfer=None, **kwargs):
    """SSD architecture with mobilenet0.25 300 base network for custom dataset.

    Parameters
    ----------
    classes : iterable of str
        Names of custom foreground classes. `len(classes)` is the number of foreground classes.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized.
    transfer : str or None
        If not `None`, will try to reuse pre-trained weights from SSD networks trained on other
        datasets.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.

    Returns
    -------
    HybridBlock
        A SSD detection network.

    Example
    -------
    >>> net = ssd_300_mobilenet0_25_custom(classes=['a', 'b', 'c'], pretrained_base=True)
    >>> net = ssd_300_mobilenet0_25_custom(classes=['foo', 'bar'], transfer='voc')

    """
    if pretrained:
        warnings.warn("Custom models don't provide `pretrained` weights, ignored.")
    if transfer is None:
        kwargs['pretrained'] = False
        net = get_ssd('mobilenet0.25', 300,
                      features=['relu22_fwd', 'relu26_fwd'],
                      filters=[256, 256, 128, 128],
                      sizes=[21, 45, 99, 153, 207, 261, 315],
                      ratios=[[1, 2, 0.5]] + [[1, 2, 0.5, 3, 1.0/3]] * 3 + [[1, 2, 0.5]] * 2,
                      steps=[8, 16, 32, 64, 100, 300],
                      classes=classes, dataset='',
                      pretrained_base=pretrained_base, **kwargs)
    else:
        from ...model_zoo import get_model
        net = get_model('ssd_300_mobilenet0.25_' + str(transfer), pretrained=True, **kwargs)
        reuse_classes = [x for x in classes if x in net.classes]
        net.reset_class(classes, reuse_weights=reuse_classes)
    return net
