# pylint: disable=abstract-method
"""Feature extraction blocks.
Feature or Multi-Feature extraction is a key component in object detection.
Class predictor/Box predictor are usually applied on feature layer(s).
A good feature extraction mechanism is critical to performance.
"""
from __future__ import absolute_import

import mxnet as mx
from mxnet.base import string_types
from mxnet.gluon import HybridBlock, SymbolBlock
from mxnet.symbol import Symbol
from mxnet.symbol.contrib import SyncBatchNorm


def _parse_network(network, outputs, inputs, pretrained, ctx, **kwargs):
    """Parse network with specified outputs and other arguments.

    Parameters
    ----------
    network : str or HybridBlock or Symbol
        Logic chain: load from gluoncv.model_zoo if network is string.
        Convert to Symbol if network is HybridBlock
    outputs : str or iterable of str
        The name of layers to be extracted as features.
    inputs : iterable of str
        The name of input datas.
    pretrained : bool
        Use pretrained parameters as in gluon.model_zoo
    ctx : Context
        The context, e.g. mxnet.cpu(), mxnet.gpu(0).

    Returns
    -------
    inputs : list of Symbol
        Network input Symbols, usually ['data']
    outputs : list of Symbol
        Network output Symbols, usually as features
    params : ParameterDict
        Network parameters.
    """
    inputs = list(inputs) if isinstance(inputs, tuple) else inputs
    for i, inp in enumerate(inputs):
        if isinstance(inp, string_types):
            inputs[i] = mx.sym.var(inp)
        assert isinstance(inputs[i], Symbol), "Network expects inputs are Symbols."
    if len(inputs) == 1:
        inputs = inputs[0]
    else:
        inputs = mx.sym.Group(inputs)
    params = None
    prefix = ''
    if isinstance(network, string_types):
        from ..model_zoo import get_model
        network = get_model(network, pretrained=pretrained, ctx=ctx, **kwargs)
    if isinstance(network, HybridBlock):
        params = network.collect_params()
        prefix = network._prefix
        network = network(inputs)
    assert isinstance(network, Symbol), \
        "FeatureExtractor requires the network argument to be either " \
        "str, HybridBlock or Symbol, but got %s" % type(network)

    if isinstance(outputs, string_types):
        outputs = [outputs]
    assert len(outputs) > 0, "At least one outputs must be specified."
    outputs = [out if out.endswith('_output') else out + '_output' for out in outputs]
    outputs = [network.get_internals()[prefix + out] for out in outputs]
    return inputs, outputs, params


class FeatureExtractor(SymbolBlock):
    """Feature extractor.

    Parameters
    ----------
    network : str or HybridBlock or Symbol
        Logic chain: load from gluoncv.model_zoo if network is string.
        Convert to Symbol if network is HybridBlock
    outputs : str or list of str
        The name of layers to be extracted as features
    inputs : list of str or list of Symbol
        The inputs of network.
    pretrained : bool
        Use pretrained parameters as in gluon.model_zoo
    ctx : Context
        The context, e.g. mxnet.cpu(), mxnet.gpu(0).
    """

    def __init__(self, network, outputs, inputs=('data',),
                 pretrained=False, ctx=mx.cpu(), **kwargs):
        inputs, outputs, params = _parse_network(
            network, outputs, inputs, pretrained, ctx, **kwargs)
        super(FeatureExtractor, self).__init__(outputs, inputs, params=params)


class FeatureExpander(SymbolBlock):
    """Feature extractor with additional layers to append.
    This is very common in vision networks where extra branches are attached to
    backbone network.

    Parameters
    ----------
    network : str or HybridBlock or Symbol
        Logic chain: load from gluoncv.model_zoo if network is string.
        Convert to Symbol if network is HybridBlock.
    outputs : str or list of str
        The name of layers to be extracted as features
    num_filters : list of int
        Number of filters to be appended.
    use_1x1_transition : bool
        Whether to use 1x1 convolution between attached layers. It is effective
        reducing network size.
    use_bn : bool
        Whether to use BatchNorm between attached layers.
    reduce_ratio : float
        Channel reduction ratio of the transition layers.
    min_depth : int
        Minimum channel number of transition layers.
    global_pool : bool
        Whether to use global pooling as the last layer.
    pretrained : bool
        Use pretrained parameters as in gluon.model_zoo if `True`.
    ctx : Context
        The context, e.g. mxnet.cpu(), mxnet.gpu(0).
    inputs : list of str
        Name of input variables to the network.

    """

    def __init__(self, network, outputs, num_filters, use_1x1_transition=True,
                 use_bn=True, reduce_ratio=1.0, min_depth=128, global_pool=False,
                 pretrained=False, ctx=mx.cpu(), inputs=('data',), **kwargs):
        inputs, outputs, params = _parse_network(
            network, outputs, inputs, pretrained, ctx, **kwargs)
        # append more layers
        y = outputs[-1]
        weight_init = mx.init.Xavier(rnd_type='gaussian', factor_type='out', magnitude=2)
        for i, f in enumerate(num_filters):
            if use_1x1_transition:
                num_trans = max(min_depth, int(round(f * reduce_ratio)))
                y = mx.sym.Convolution(
                    y, num_filter=num_trans, kernel=(1, 1), no_bias=use_bn,
                    name='expand_trans_conv{}'.format(i), attr={'__init__': weight_init})
                if use_bn:
                    y = mx.sym.BatchNorm(y, name='expand_trans_bn{}'.format(i))
                y = mx.sym.Activation(y, act_type='relu', name='expand_trans_relu{}'.format(i))
            y = mx.sym.Convolution(
                y, num_filter=f, kernel=(3, 3), pad=(1, 1), stride=(2, 2),
                no_bias=use_bn, name='expand_conv{}'.format(i), attr={'__init__': weight_init})
            if use_bn:
                y = mx.sym.BatchNorm(y, name='expand_bn{}'.format(i))
            y = mx.sym.Activation(y, act_type='relu', name='expand_relu{}'.format(i))
            outputs.append(y)
        if global_pool:
            outputs.append(mx.sym.Pooling(y, pool_type='avg', global_pool=True, kernel=(1, 1)))
        super(FeatureExpander, self).__init__(outputs, inputs, params)


class FPNFeatureExpander(SymbolBlock):
    """Feature extractor with additional layers to append.
    This is specified for ``Feature Pyramid Network for Object Detection``
    which implement ``Top-down pathway and lateral connections``.

    Parameters
    ----------
    network : str or HybridBlock or Symbol
        Logic chain: load from gluon.model_zoo.vision if network is string.
        Convert to Symbol if network is HybridBlock.
    outputs : str or list of str
        The name of layers to be extracted as features
    num_filters : list of int e.g. [256, 256, 256, 256]
        Number of filters to be appended.
    use_1x1 : bool
        Whether to use 1x1 convolution
    use_upsample : bool
        Whether to use upsample
    use_elewadd : float
        Whether to use element-wise add operation
    use_p6 : bool
        Whether use P6 stage, this is used for RPN experiments in ori paper
    p6_conv : bool
        Whether to use convolution for P6 stage, if it is enabled, or just max pooling.
    no_bias : bool
        Whether use bias for Convolution operation.
    norm_layer : HybridBlock or SymbolBlock
        Type of normalization layer.
    norm_kwargs : dict
        Arguments for normalization layer.
    pretrained : bool
        Use pretrained parameters as in gluon.model_zoo if `True`.
    ctx : Context
        The context, e.g. mxnet.cpu(), mxnet.gpu(0).
    inputs : list of str
        Name of input variables to the network.

    """

    def __init__(self, network, outputs, num_filters, use_1x1=True, use_upsample=True,
                 use_elewadd=True, use_p6=False, p6_conv=True, no_bias=True, pretrained=False,
                 norm_layer=None, norm_kwargs=None, ctx=mx.cpu(), inputs=('data',)):
        inputs, outputs, params = _parse_network(network, outputs, inputs, pretrained, ctx)
        if norm_kwargs is None:
            norm_kwargs = {}
        # e.g. For ResNet50, the feature is :
        # outputs = ['stage1_activation2', 'stage2_activation3',
        #            'stage3_activation5', 'stage4_activation2']
        # with regard to [conv2, conv3, conv4, conv5] -> [C2, C3, C4, C5]
        # append more layers with reversed order : [P5, P4, P3, P2]
        y = outputs[-1]
        base_features = outputs[::-1]
        num_stages = len(num_filters) + 1  # usually 5
        weight_init = mx.init.Xavier(rnd_type='uniform', factor_type='in', magnitude=1.)
        tmp_outputs = []
        # num_filter is 256 in ori paper
        for i, (bf, f) in enumerate(zip(base_features, num_filters)):
            if i == 0:
                if use_1x1:
                    y = mx.sym.Convolution(y, num_filter=f, kernel=(1, 1), pad=(0, 0),
                                           stride=(1, 1), no_bias=no_bias,
                                           name="P{}_conv_lat".format(num_stages - i),
                                           attr={'__init__': weight_init})
                    if norm_layer is not None:
                        if norm_layer is SyncBatchNorm:
                            norm_kwargs['key'] = "P{}_lat_bn".format(num_stages - i)
                            norm_kwargs['name'] = "P{}_lat_bn".format(num_stages - i)
                        y = norm_layer(y, **norm_kwargs)
                if use_p6 and p6_conv:
                    # method 2 : use conv (Deformable use this)
                    y_p6 = mx.sym.Convolution(y, num_filter=f, kernel=(3, 3), pad=(1, 1),
                                              stride=(2, 2), no_bias=no_bias,
                                              name='P{}_conv1'.format(num_stages + 1),
                                              attr={'__init__': weight_init})
                    if norm_layer is not None:
                        if norm_layer is SyncBatchNorm:
                            norm_kwargs['key'] = "P{}_pre_bn".format(num_stages + 1)
                            norm_kwargs['name'] = "P{}_pre_bn".format(num_stages + 1)
                        y_p6 = norm_layer(y_p6, **norm_kwargs)
            else:
                if use_1x1:
                    bf = mx.sym.Convolution(bf, num_filter=f, kernel=(1, 1), pad=(0, 0),
                                            stride=(1, 1), no_bias=no_bias,
                                            name="P{}_conv_lat".format(num_stages - i),
                                            attr={'__init__': weight_init})
                    if norm_layer is not None:
                        if norm_layer is SyncBatchNorm:
                            norm_kwargs['key'] = "P{}_conv1_bn".format(num_stages - i)
                            norm_kwargs['name'] = "P{}_conv1_bn".format(num_stages - i)
                        bf = norm_layer(bf, **norm_kwargs)
                if use_upsample:
                    y = mx.sym.UpSampling(y, scale=2, sample_type='nearest',
                                          name="P{}_upsp".format(num_stages - i))

                if use_elewadd:
                    # make two symbol alignment
                    # method 1 : mx.sym.Crop
                    # y = mx.sym.Crop(*[y, bf], name="P{}_clip".format(num_stages-i))
                    # method 2 : mx.sym.slice_like
                    y = mx.sym.slice_like(y, bf, axes=(2, 3),
                                          name="P{}_clip".format(num_stages - i))
                    y = mx.sym.ElementWiseSum(bf, y, name="P{}_sum".format(num_stages - i))
            # Reduce the aliasing effect of upsampling described in ori paper
            out = mx.sym.Convolution(y, num_filter=f, kernel=(3, 3), pad=(1, 1), stride=(1, 1),
                                     no_bias=no_bias, name='P{}_conv1'.format(num_stages - i),
                                     attr={'__init__': weight_init})
            if i == 0 and use_p6 and not p6_conv:
                # method 2 : use max pool (Detectron use this)
                y_p6 = mx.sym.Pooling(out, pool_type='max', kernel=(1, 1), pad=(0, 0),
                                      stride=(2, 2), name="P{}_pre".format(num_stages + 1))
            if norm_layer is not None:
                if norm_layer is SyncBatchNorm:
                    norm_kwargs['key'] = "P{}_bn".format(num_stages - i)
                    norm_kwargs['name'] = "P{}_bn".format(num_stages - i)
                out = norm_layer(out, **norm_kwargs)
            tmp_outputs.append(out)
        if use_p6:
            outputs = tmp_outputs[::-1] + [y_p6]  # [P2, P3, P4, P5] + [P6]
        else:
            outputs = tmp_outputs[::-1]  # [P2, P3, P4, P5]

        super(FPNFeatureExpander, self).__init__(outputs, inputs, params)
