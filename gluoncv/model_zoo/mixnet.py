# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable=redefined-variable-type,simplifiable-if-expression,inconsistent-return-statements,unused-argument,arguments-differ
"""MixNet, implemented in Gluon."""

from __future__ import division

import numpy as np
# import mxnet as mx
# from mxnet import nd
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm
from mxnet.gluon.block import HybridBlock
from mxnet.context import cpu
from ..nn import ReLU6, HardSigmoid, HardSwish


__all__ = [
    'MixNet',
    'mixnet_s',
    'mixnet_m',
    'mixnet_l',
    'get_mixnet']


# Helpers
def _conv1x1(in_channels, out_channels):
    return nn.Conv2D(out_channels, kernel_size=1, strides=1, padding=0, \
                     use_bias=False, in_channels=in_channels)

def _conv3x3(in_channels, out_channels, stride):
    return nn.Conv2D(out_channels, kernel_size=3, strides=stride, padding=1, \
                     use_bias=False, in_channels=in_channels)


def make_divisible(x, divisor=8):
    return int(np.ceil(x * 1. / divisor) * divisor)


def _round_filters(filters, divisor=8, min_depth=None):
    """Round number of filters based on depth multiplier."""
    if min_depth is None:
        min_depth = divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return new_filters


def _split_channels(total_filters, num_groups):
    """Get groups list."""
    split_channels = [total_filters // num_groups for _ in range(num_groups)]
    split_channels[0] += total_filters - sum(split_channels)
    return split_channels


def _group_split(x, split_groups, axis=1):
    """Split a tensor into arbitrary contiguous groups
       along the specified dimension.
    """
    x_splits = []
    begin = 0
    end = split_groups[0]
    # x_splits.append(nd.slice_axis(x, begin=begin, end=end, axis=axis))
    x_splits.append(x.slice_axis(begin=begin, end=end, axis=axis))
    for i in range(1, len(split_groups)):
        begin += split_groups[i - 1]
        end += split_groups[i]
        # x_splits.append(nd.slice_axis(x, begin=begin, end=end, axis=axis))
        x_splits.append(x.slice_axis(begin=begin, end=end, axis=axis))
    return x_splits


class Activation(HybridBlock):
    """Activation function."""
    def __init__(self, act_func, **kwargs):
        super(Activation, self).__init__(**kwargs)
        if act_func == 'relu':
            self.act = nn.Activation('relu')
        elif act_func == 'relu6':
            self.act = ReLU6()
        elif act_func == 'hard_sigmoid':
            self.act = HardSigmoid()
        elif act_func == 'swish':
            self.act = nn.Swish()
        elif act_func == 'hard_swish':
            self.act = HardSwish()
        elif act_func == 'leaky':
            self.act = nn.LeakyReLU(alpha=0.375)
        else:
            raise NotImplementedError

    def hybrid_forward(self, F, x):
        return self.act(x)


class _SE(HybridBlock):
    def __init__(self, num_out, ratio=4, \
                 act_func=("relu", "hard_sigmoid"), \
                 use_bn=False, prefix='', **kwargs):
        super(_SE, self).__init__(**kwargs)
        self.use_bn = use_bn
        num_mid = make_divisible(num_out // ratio)
        self.pool = nn.GlobalAvgPool2D()
        self.conv1 = nn.Conv2D(channels=num_mid, \
                               kernel_size=1, use_bias=True)
        self.act1 = Activation(act_func[0])
        self.conv2 = nn.Conv2D(channels=num_out, \
                               kernel_size=1, use_bias=True)
        self.act2 = Activation(act_func[1])

    def hybrid_forward(self, F, x):
        out = self.pool(x)
        out = self.conv1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        return F.broadcast_mul(x, out)


class MDConv(HybridBlock):
    r"""MDConv from the
    `"MixConv: Mixed Depthwise Convolutional Kernels"
    <https://arxiv.org/abs/1907.09595>`_ paper.

    Parameters
    ----------
    channels : int
        Number of input and output channels of MDConv.
    kernel_size: list
        The size of filters of each channel group.
    stride: int
        The stride of filters of each channel group.
    """
    def __init__(self, channels, kernel_size, stride, **kwargs):
        super(MDConv, self).__init__(**kwargs)

        self.num_groups = len(kernel_size)
        self.split_channels = _split_channels(channels, self.num_groups)

        self.mix_dw_conv = nn.HybridSequential()
        with self.mix_dw_conv.name_scope():
             for i in range(self.num_groups):
                self.mix_dw_conv.add(nn.Conv2D(channels=self.split_channels[i], \
                                                kernel_size=kernel_size[i], \
                                                strides=stride, \
                                                padding=kernel_size[i]//2, \
                                                groups=self.split_channels[i], \
                                                use_bias=False))

    def hybrid_forward(self, F, x):
        """Mixed Depthwise Convolution."""
        if self.num_groups == 1:
            return self.mix_dw_conv[0](x)
        # For unequal arbitrary contiguous groups.
        x_splits = _group_split(x, self.split_channels, axis=1)

        # For equal contiguous groups.
        # But the MDConv convolution param # is 0, debugging.
        # x_splits = nd.split(x, num_outputs=self.num_groups, axis=1) # for NDArray data
        # x_splits = x.split(num_outputs=self.num_groups, axis=1) # for Symbol data

        x = [conv(t) for conv, t in zip(self.mix_dw_conv, x_splits)]
        x = F.concat(*x, dim=1)
        return x


class MixNetBlock(HybridBlock):
    r"""MixNetBlock from the
    `"MixConv: Mixed Depthwise Convolutional Kernels"
    <https://arxiv.org/abs/1907.09595>`_ paper.

    Parameters
    ----------
    in_channels : int
        Number of input channels of convolution layer.
    out_channels : int
        Number of final output channels of the MixNetBlock.
    kernel_size: list
        The size of filters of each channel group.
    stride: int
        The stride of filters of each channel group.
    expand_ratio: int
        The expand ratio of MixConv channels.
    act_type: string, default relu
        The type of activation function.
    se_ratio: int, default 0
        The ratio of squeeze and excite, default 0 means that se operation is not used.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    use_global_stats: bool, default False
        Whether use global moving statistics instead of local batch-norm.
        This will force change batch-norm into a scale shift operator.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, \
                 expand_ratio, act_type='relu', se_ratio=0, \
                 norm_layer=BatchNorm, norm_kwargs=None, \
                 use_global_stats=False, **kwargs):
        super(MixNetBlock, self).__init__(**kwargs)

        self.norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
        if use_global_stats:
            self.norm_kwargs['use_global_stats'] = True

        expand = (expand_ratio != 1)
        expand_channels = in_channels * expand_ratio
        se = (se_ratio != 0)
        self.residual_connection = (stride == 1 and in_channels == out_channels)

        self.body = nn.HybridSequential(prefix='')
        if expand:
            self.body.add(_conv1x1(in_channels, expand_channels))
            self.body.add(norm_layer(in_channels=expand_channels, **(self.norm_kwargs)))
            self.body.add(Activation(act_type))

        self.body.add(MDConv(expand_channels, kernel_size, stride))
        self.body.add(norm_layer(in_channels=expand_channels, **(self.norm_kwargs)))
        self.body.add(Activation(act_type))

        if se:
            self.body.add(_SE(expand_channels, se_ratio))

        self.body.add(_conv1x1(expand_channels, out_channels))
        self.body.add(norm_layer(in_channels=out_channels, **(self.norm_kwargs)))

    def hybrid_forward(self, F, x):
        if self.residual_connection:
            return x + self.body(x)
        else:
            return self.body(x)


class MixNet(HybridBlock):
    r"""MixNet model from the
    `"MixConv: Mixed Depthwise Convolutional Kernels"
    <https://arxiv.org/abs/1907.09595>`_ paper.

    Parameters
    ----------
    net_type : string, default mixnet_s
        The name of mixnet models, mixnet_s, mixnet_m and mixnet_l are supported.
    input_size : int, default 224
        The size of input image.
    input_channels : int, default 3
        Number of channels of input image.
    stem_channels : int, default 16
        Number of output channels of the first convolution layer,
        also means the number of filters of the first convolution layer.
    feature_size: int, default 1536
        Number of final channels of the last convolution layer before a classifier.
    num_classes: int, default 1000
        Number of classification classes.
    depth_multiplier: float, default 1.0
        Update block input and output filters based on depth multiplier.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    use_global_stats: bool, default False
        Whether use global moving statistics instead of local batch-norm.
        This will force change batch-norm into a scale shift operator.
    """

    # [in_channels, out_channels, kernel_size, stride, expand_ratio, act_type, se_ratio]
    mixnet_s = [(16, 16, [3], 1, 1, 'relu', 0),
                (16, 24, [3], 2, 6, 'relu', 0),
                (24, 24, [3], 1, 3, 'relu', 0),
                (24, 40, [3, 5, 7], 2, 6, 'swish', 2),
                (40, 40, [3, 5], 1, 6, 'swish', 2),
                (40, 40, [3, 5], 1, 6, 'swish', 2),
                (40, 40, [3, 5], 1, 6, 'swish', 2),
                (40, 80, [3, 5, 7], 2, 6, 'swish', 4),
                (80, 80, [3, 5], 1, 6, 'swish', 4),
                (80, 80, [3, 5], 1, 6, 'swish', 4),
                (80, 120, [3, 5, 7], 1, 6, 'swish', 2),
                (120, 120, [3, 5, 7, 9], 1, 3, 'swish', 2),
                (120, 120, [3, 5, 7, 9], 1, 3, 'swish', 2),
                (120, 200, [3, 5, 7, 9, 11], 2, 6, 'swish', 2),
                (200, 200, [3, 5, 7, 9], 1, 6, 'swish', 2),
                (200, 200, [3, 5, 7, 9], 1, 6, 'swish', 2)]

    mixnet_m = [(24, 24, [3], 1, 1, 'relu', 0),
                (24, 32, [3, 5, 7], 2, 6, 'relu', 0),
                (32, 32, [3], 1, 3, 'relu', 0),
                (32, 40, [3, 5, 7, 9], 2, 6, 'swish', 2),
                (40, 40, [3, 5], 1, 6, 'swish', 2),
                (40, 40, [3, 5], 1, 6, 'swish', 2),
                (40, 40, [3, 5], 1, 6, 'swish', 2),
                (40, 80, [3, 5, 7], 2, 6, 'swish', 4),
                (80, 80, [3, 5, 7, 9], 1, 6, 'swish', 4),
                (80, 80, [3, 5, 7, 9], 1, 6, 'swish', 4),
                (80, 80, [3, 5, 7, 9], 1, 6, 'swish', 4),
                (80, 120, [3], 1, 6, 'swish', 2),
                (120, 120, [3, 5, 7, 9], 1, 3, 'swish', 2),
                (120, 120, [3, 5, 7, 9], 1, 3, 'swish', 2),
                (120, 120, [3, 5, 7, 9], 1, 3, 'swish', 2),
                (120, 200, [3, 5, 7, 9], 2, 6, 'swish', 2),
                (200, 200, [3, 5, 7, 9], 1, 6, 'swish', 2),
                (200, 200, [3, 5, 7, 9], 1, 6, 'swish', 2),
                (200, 200, [3, 5, 7, 9], 1, 6, 'swish', 2)]

    def __init__(self, net_type='mixnet_s', input_size=224, input_channels=3, \
                 stem_channels=16, feature_size=1536, num_classes=1000, \
                 depth_multiplier=1.0, norm_layer=BatchNorm, \
                 norm_kwargs=None, use_global_stats=False, **kwargs):
        super(MixNet, self).__init__(**kwargs)

        self.norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
        if use_global_stats:
            self.norm_kwargs['use_global_stats'] = True

        # net type
        if net_type == 'mixnet_s':
            config = self.mixnet_s
            stem_channels = 16
            dropout_rate = 0.2
        elif net_type == 'mixnet_m':
            config = self.mixnet_m
            stem_channels = 24
            dropout_rate = 0.25
        elif net_type == 'mixnet_l':
            config = self.mixnet_m
            stem_channels = 24
            depth_multiplier *= 1.3
            dropout_rate = 0.25
        else:
            raise TypeError('Unsupported MixNet type')

        assert input_size % 32 == 0

        # depth multiplier
        if depth_multiplier != 1.0:
            stem_channels = _round_filters(stem_channels * depth_multiplier)

            for i, conf in enumerate(config):
                conf_ls = list(conf)
                conf_ls[0] = _round_filters(conf_ls[0] * depth_multiplier)
                conf_ls[1] = _round_filters(conf_ls[1] * depth_multiplier)
                config[i] = tuple(conf_ls)

        # stem convolution
        self.stem_conv = nn.HybridSequential(prefix='')
        self.stem_conv.add(_conv3x3(input_channels, stem_channels, stride=2))
        self.stem_conv.add(norm_layer(in_channels=stem_channels, **(self.norm_kwargs)))
        self.stem_conv.add(Activation('relu'))

        # building MixNet blocks
        self.mix_layers = nn.HybridSequential(prefix='')
        for in_chs, out_chs, k_size, s, exp_ratio, act_type, se_ratio in config:
            self.mix_layers.add(MixNetBlock(in_chs, out_chs, k_size, s, \
                                            exp_ratio, act_type, se_ratio))

        # head layers
        self.head_layers = nn.HybridSequential(prefix='')
        self.head_layers.add(_conv1x1(config[-1][1], feature_size))
        self.head_layers.add(norm_layer(in_channels=feature_size, **(self.norm_kwargs)))
        self.head_layers.add(Activation('relu'))
        self.head_layers.add(nn.GlobalAvgPool2D())
        if dropout_rate > 0:
            self.head_layers.add(nn.Dropout(dropout_rate))

        # output layers
        self.output = nn.HybridSequential(prefix='output_')
        self.output.add(
            nn.Conv2D(in_channels=feature_size, channels=num_classes, \
                      kernel_size=1, prefix='fc_'),
            nn.Flatten())

    def hybrid_forward(self, F, x):
        x = self.stem_conv(x)
        x = self.mix_layers(x)
        x = self.head_layers(x)
        x = self.output(x)
        return x


def get_mixnet(net_type, pretrained=False, ctx=cpu(), root='~/.mxnet/models', \
               norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
    r"""MixNet model from the
    `"MixConv: Mixed Depthwise Convolutional Kernels"
    <https://arxiv.org/abs/1907.09595>`_ paper.

    Parameters
    ----------
    net_type : string
        The name of mixnet models, mixnet_s, mixnet_m and mixnet_l are supported.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    net = MixNet(net_type, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        net.load_parameters(get_model_file('mixnet_%s' % net_type.split('_')[1], \
                                           tag=pretrained, root=root), ctx=ctx)
        from ..data import ImageNet1kAttr
        attrib = ImageNet1kAttr()
        net.synset = attrib.synset
        net.classes = attrib.classes
        net.classes_long = attrib.classes_long
    return net


def mixnet_s(**kwargs):
    r"""MixNet model from the
    `"MixConv: Mixed Depthwise Convolutional Kernels"
    <https://arxiv.org/abs/1907.09595>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    return get_mixnet("mixnet_s", **kwargs)


def mixnet_m(**kwargs):
    r"""MixNet model from the
    `"MixConv: Mixed Depthwise Convolutional Kernels"
    <https://arxiv.org/abs/1907.09595>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    return get_mixnet("mixnet_m", **kwargs)


def mixnet_l(**kwargs):
    r"""MixNet model from the
    `"MixConv: Mixed Depthwise Convolutional Kernels"
    <https://arxiv.org/abs/1907.09595>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    return get_mixnet("mixnet_l", **kwargs)
