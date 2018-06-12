# pylint: disable=wildcard-import, unused-wildcard-import
"""Model store which handles pretrained models from both
mxnet.gluon.model_zoo.vision and gluoncv.models
"""
from mxnet import gluon
from .ssd import *
from .fcn import *
from .cifarresnet import *
from .cifarresnext import *
from .cifarwideresnet import *
from .resnetv1b import *
from .resnext import *
from .senet import *
from .se_resnet import *

__all__ = ['get_model']

def get_model(name, **kwargs):
    """Returns a pre-defined model by name

    Parameters
    ----------
    name : str
        Name of the model.
    pretrained : bool
        Whether to load the pretrained weights for model.
    classes : int
        Number of classes for the output layer.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Returns
    -------
    HybridBlock
        The model.
    """
    models = {
        'ssd_300_vgg16_atrous_voc': ssd_300_vgg16_atrous_voc,
        'ssd_512_vgg16_atrous_voc': ssd_512_vgg16_atrous_voc,
        'ssd_512_resnet18_v1_voc': ssd_512_resnet18_v1_voc,
        'ssd_512_resnet50_v1_voc': ssd_512_resnet50_v1_voc,
        'ssd_512_resnet101_v2_voc': ssd_512_resnet101_v2_voc,
        'ssd_512_resnet152_v2_voc': ssd_512_resnet152_v2_voc,
        'ssd_512_mobilenet1_0_voc': ssd_512_mobilenet1_0_voc,
        'cifar_resnet20_v1': cifar_resnet20_v1,
        'cifar_resnet56_v1': cifar_resnet56_v1,
        'cifar_resnet110_v1': cifar_resnet110_v1,
        'cifar_resnet20_v2': cifar_resnet20_v2,
        'cifar_resnet56_v2': cifar_resnet56_v2,
        'cifar_resnet110_v2': cifar_resnet110_v2,
        'cifar_wideresnet16_10': cifar_wideresnet16_10,
        'cifar_wideresnet28_10': cifar_wideresnet28_10,
        'cifar_wideresnet40_8': cifar_wideresnet40_8,
        'cifar_resnext29_32x4d': cifar_resnext29_32x4d,
        'cifar_resnext29_16x64d': cifar_resnext29_16x64d,
        'fcn_resnet50_voc' : get_fcn_voc_resnet50,
        'fcn_resnet101_voc' : get_fcn_voc_resnet101,
        'resnet18_v1b' : resnet18_v1b,
        'resnet34_v1b' : resnet34_v1b,
        'resnet50_v1b' : resnet50_v1b,
        'resnet101_v1b' : resnet101_v1b,
        'resnet152_v1b' : resnet152_v1b,
        'resnext50_32x4d' : resnext50_32x4d,
        'resnext101_32x4d' : resnext101_32x4d,
        'resnext101_64x4d' : resnext101_64x4d,
        'se_resnext50_32x4d' : se_resnext50_32x4d,
        'se_resnext101_32x4d' : se_resnext101_32x4d,
        'se_resnext101_64x4d' : se_resnext101_64x4d,
        'senet_52' : senet_52,
        'senet_103' : senet_103,
        'senet_154' : senet_154,
        'se_resnet18_v1' : se_resnet18_v1,
        'se_resnet34_v1' : se_resnet34_v1,
        'se_resnet50_v1' : se_resnet50_v1,
        'se_resnet101_v1' : se_resnet101_v1,
        'se_resnet152_v1' : se_resnet152_v1,
        'se_resnet18_v2' : se_resnet18_v2,
        'se_resnet34_v2' : se_resnet34_v2,
        'se_resnet50_v2' : se_resnet50_v2,
        'se_resnet101_v2' : se_resnet101_v2,
        'se_resnet152_v2' : se_resnet152_v2,
        }
    try:
        net = gluon.model_zoo.vision.get_model(name, **kwargs)
    except ValueError as e:
        name = name.lower()
        if name not in models:
            raise ValueError('%s\n\t%s' % (str(e), '\n\t'.join(sorted(models.keys()))))
        net = models[name](**kwargs)
    return net
