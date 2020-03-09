"""Pruned ResNetV1bs, implemented in Gluon."""
from __future__ import division
import json
import os
from mxnet.context import cpu
from mxnet.gluon import nn
from mxnet import ndarray
from ..resnetv1b import ResNetV1b
from ..resnetv1b import BasicBlockV1b
from ..resnetv1b import BottleneckV1b


__all__ = ['resnet18_v1b_89', 'resnet50_v1d_86', 'resnet50_v1d_48', 'resnet50_v1d_37',
           'resnet50_v1d_11', 'resnet101_v1d_76', 'resnet101_v1d_73']


def prune_gluon_block(net, prefix, params_shapes, params=None, pretrained=False, ctx=cpu(0)):
    """
    :param params_shapes: dictionary of shapes of convolutional weights
    :param prefix: prefix of the original resnet50_v1d
    :param pretrained: Boolean specifying if the pretrained model parameters needs to be loaded
    :param net: original network that is required to be pruned
    :param params: dictionary of parameters for the pruned network. Size of the parameters in
    this dictionary tells what
    should be the size of channels of each convolution layer.
    :param ctx: cpu(0)
    :return: "net"
    """
    for _, layer in net._children.items():
        if pretrained:
            if isinstance(layer, nn.BatchNorm):
                params_layer = layer._collect_params_with_prefix()
                for param_name in ['beta', 'gamma', 'running_mean', 'running_var']:
                    param_val = params[layer.name.replace(prefix, "resnetv1d") + "_" + param_name]
                    layer.params.get(param_name)._shape = param_val.shape
                    params_layer[param_name]._load_init(param_val, ctx=ctx)

        if isinstance(layer, nn.Conv2D):
            param_shape = params_shapes[layer.name.replace(prefix, "resnetv1d") + "_weight"]
            layer._channels = param_shape[0]
            layer._kwargs['num_filter'] = param_shape[0]

            params_layer = layer._collect_params_with_prefix()
            for param_name in ['weight']:
                param_shape = params_shapes[
                    layer.name.replace(prefix, "resnetv1d") + "_" + param_name]
                layer.params.get(param_name)._shape = param_shape
                if pretrained:
                    param_val = params[layer.name.replace(prefix, "resnetv1d") + "_" + param_name]
                    params_layer[param_name]._load_init(param_val, ctx=ctx)

        if isinstance(layer, nn.Dense):
            layer._in_units = params_shapes[layer.name.replace(prefix, "resnetv1d") + "_weight"][1]

            params_layer = layer._collect_params_with_prefix()
            for param_name in ['weight', 'bias']:
                param_shape = params_shapes[
                    layer.name.replace(prefix, "resnetv1d") + "_" + param_name]
                layer.params.get(param_name)._shape = param_shape
                if pretrained:
                    param_val = params[layer.name.replace(prefix, "resnetv1d") + "_" + param_name]
                    params_layer[param_name]._load_init(param_val, ctx=ctx)
        else:
            prune_gluon_block(layer, prefix, params_shapes, params, pretrained, ctx)


def resnet18_v1b_89(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    """Constructs a ResNetV1b-18_2.6x model. Uses resnet18_v1b construction from resnetv1b.py

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    """
    model = ResNetV1b(BasicBlockV1b, [2, 2, 2, 2], name_prefix='resnetv1b_', **kwargs)
    dirname = os.path.dirname(__file__)
    json_filename = os.path.join(dirname, 'resnet%d_v%db_%.1fx' % (18, 1, 2.6) + ".json")
    with open(json_filename, "r") as jsonFile:
        params_shapes = json.load(jsonFile)
    if pretrained:
        from ..model_store import get_model_file
        params_file = get_model_file('resnet%d_v%db_%.1fx' % (18, 1, 2.6), tag=pretrained,
                                     root=root)
        prune_gluon_block(model, model.name, params_shapes, params=ndarray.load(params_file),
                          pretrained=True, ctx=ctx)
    else:
        prune_gluon_block(model, model.name, params_shapes, params=None, pretrained=False, ctx=ctx)
    if pretrained:
        from ...data import ImageNet1kAttr
        attrib = ImageNet1kAttr()
        model.synset = attrib.synset
        model.classes = attrib.classes
        model.classes_long = attrib.classes_long
    return model


def resnet50_v1d_86(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    """Constructs a ResNetV1d-50_1.8x model. Uses resnet50_v1d construction from resnetv1b.py

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    """
    model = ResNetV1b(BottleneckV1b, [3, 4, 6, 3], deep_stem=True, avg_down=True,
                      name_prefix='resnetv1d_', **kwargs)
    dirname = os.path.dirname(__file__)
    json_filename = os.path.join(dirname, 'resnet%d_v%dd_%.1fx' % (50, 1, 1.8) + ".json")
    with open(json_filename, "r") as jsonFile:
        params_shapes = json.load(jsonFile)
    if pretrained:
        from ..model_store import get_model_file
        params_file = get_model_file('resnet%d_v%dd_%.1fx' % (50, 1, 1.8), tag=pretrained,
                                     root=root)
        prune_gluon_block(model, model.name, params_shapes, params=ndarray.load(params_file),
                          pretrained=True, ctx=ctx)
    else:
        prune_gluon_block(model, model.name, params_shapes, params=None, pretrained=False, ctx=ctx)

    if pretrained:
        from ...data import ImageNet1kAttr
        attrib = ImageNet1kAttr()
        model.synset = attrib.synset
        model.classes = attrib.classes
        model.classes_long = attrib.classes_long
    return model


def resnet50_v1d_48(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    """Constructs a ResNetV1d-50_3.6x model. Uses resnet50_v1d construction from resnetv1b.py

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    """
    model = ResNetV1b(BottleneckV1b, [3, 4, 6, 3], deep_stem=True, avg_down=True,
                      name_prefix='resnetv1d_', **kwargs)
    dirname = os.path.dirname(__file__)
    json_filename = os.path.join(dirname, 'resnet%d_v%dd_%.1fx' % (50, 1, 3.6) + ".json")
    with open(json_filename, "r") as jsonFile:
        params_shapes = json.load(jsonFile)
    if pretrained:
        from ..model_store import get_model_file
        params_file = get_model_file('resnet%d_v%dd_%.1fx' % (50, 1, 3.6), tag=pretrained,
                                     root=root)
        prune_gluon_block(model, model.name, params_shapes, params=ndarray.load(params_file),
                          pretrained=True, ctx=ctx)
    else:
        prune_gluon_block(model, model.name, params_shapes, params=None, pretrained=False, ctx=ctx)

    if pretrained:
        from ...data import ImageNet1kAttr
        attrib = ImageNet1kAttr()
        model.synset = attrib.synset
        model.classes = attrib.classes
        model.classes_long = attrib.classes_long
    return model


def resnet50_v1d_37(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    """Constructs a ResNetV1d-50_5.9x model. Uses resnet50_v1d construction from resnetv1b.py

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    """
    model = ResNetV1b(BottleneckV1b, [3, 4, 6, 3], deep_stem=True, avg_down=True,
                      name_prefix='resnetv1d_', **kwargs)
    dirname = os.path.dirname(__file__)
    json_filename = os.path.join(dirname, 'resnet%d_v%dd_%.1fx' % (50, 1, 5.9) + ".json")
    with open(json_filename, "r") as jsonFile:
        params_shapes = json.load(jsonFile)
    if pretrained:
        from ..model_store import get_model_file
        params_file = get_model_file('resnet%d_v%dd_%.1fx' % (50, 1, 5.9), tag=pretrained,
                                     root=root)
        prune_gluon_block(model, model.name, params_shapes, params=ndarray.load(params_file),
                          pretrained=True, ctx=ctx)
    else:
        prune_gluon_block(model, model.name, params_shapes, params=None, pretrained=False, ctx=ctx)

    if pretrained:
        from ...data import ImageNet1kAttr
        attrib = ImageNet1kAttr()
        model.synset = attrib.synset
        model.classes = attrib.classes
        model.classes_long = attrib.classes_long
    return model


def resnet50_v1d_11(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    """Constructs a ResNetV1d-50_8.8x model. Uses resnet50_v1d construction from resnetv1b.py

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    """
    model = ResNetV1b(BottleneckV1b, [3, 4, 6, 3], deep_stem=True, avg_down=True,
                      name_prefix='resnetv1d_', **kwargs)
    dirname = os.path.dirname(__file__)
    json_filename = os.path.join(dirname, 'resnet%d_v%dd_%.1fx' % (50, 1, 8.8) + ".json")
    with open(json_filename, "r") as jsonFile:
        params_shapes = json.load(jsonFile)
    if pretrained:
        from ..model_store import get_model_file
        params_file = get_model_file('resnet%d_v%dd_%.1fx' % (50, 1, 8.8), tag=pretrained,
                                     root=root)
        prune_gluon_block(model, model.name, params_shapes, params=ndarray.load(params_file),
                          pretrained=True, ctx=ctx)
    else:
        prune_gluon_block(model, model.name, params_shapes, params=None, pretrained=False, ctx=ctx)

    if pretrained:
        from ...data import ImageNet1kAttr
        attrib = ImageNet1kAttr()
        model.synset = attrib.synset
        model.classes = attrib.classes
        model.classes_long = attrib.classes_long
    return model


def resnet101_v1d_76(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    """Constructs a ResNetV1d-101_1.9x model. Uses resnet101_v1d construction from resnetv1b.py

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    """
    model = ResNetV1b(BottleneckV1b, [3, 4, 23, 3], deep_stem=True, avg_down=True,
                      name_prefix='resnetv1d_', **kwargs)
    dirname = os.path.dirname(__file__)
    json_filename = os.path.join(dirname, 'resnet%d_v%dd_%.1fx' % (101, 1, 1.9) + ".json")
    with open(json_filename, "r") as jsonFile:
        params_shapes = json.load(jsonFile)
    if pretrained:
        from ..model_store import get_model_file
        params_file = get_model_file('resnet%d_v%dd_%.1fx' % (101, 1, 1.9), tag=pretrained,
                                     root=root)
        prune_gluon_block(model, model.name, params_shapes, params=ndarray.load(params_file),
                          pretrained=True, ctx=ctx)
    else:
        prune_gluon_block(model, model.name, params_shapes, params=None, pretrained=False, ctx=ctx)

    if pretrained:
        from ...data import ImageNet1kAttr
        attrib = ImageNet1kAttr()
        model.synset = attrib.synset
        model.classes = attrib.classes
        model.classes_long = attrib.classes_long
    return model


def resnet101_v1d_73(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    """Constructs a ResNetV1d-101_2.2x model. Uses resnet101_v1d construction from resnetv1b.py

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    """
    model = ResNetV1b(BottleneckV1b, [3, 4, 23, 3], deep_stem=True, avg_down=True,
                      name_prefix='resnetv1d_', **kwargs)
    dirname = os.path.dirname(__file__)
    json_filename = os.path.join(dirname, 'resnet%d_v%dd_%.1fx' % (101, 1, 2.2) + ".json")
    with open(json_filename, "r") as jsonFile:
        params_shapes = json.load(jsonFile)
    if pretrained:
        from ..model_store import get_model_file
        params_file = get_model_file('resnet%d_v%dd_%.1fx' % (101, 1, 2.2), tag=pretrained,
                                     root=root)
        prune_gluon_block(model, model.name, params_shapes, params=ndarray.load(params_file),
                          pretrained=True, ctx=ctx)
    else:
        prune_gluon_block(model, model.name, params_shapes, params=None, pretrained=False, ctx=ctx)

    if pretrained:
        from ...data import ImageNet1kAttr
        attrib = ImageNet1kAttr()
        model.synset = attrib.synset
        model.classes = attrib.classes
        model.classes_long = attrib.classes_long
    return model
