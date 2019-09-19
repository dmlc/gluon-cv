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
# pylint: disable= arguments-differ,unused-argument,missing-docstring

import math
import collections
import re
import mxnet as mx
from mxnet.gluon.block import Block
from mxnet.gluon import nn
# Parameters for the entire model (stem, all blocks, and head)

__all__ = ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
           'efficientnet_b3', 'efficientnet_b4', 'efficientnet_b5',
           'efficientnet_b6', 'efficientnet_b7']


# Parameters for an individual model block
BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'stride', 'se_ratio'])


def round_repeats(repeats, depth_coefficient=None):
    """ Round number of filters based on depth multiplier. """
    multiplier = depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def round_filters(filters, width_coefficient=None, depth_divisor=None, min_depth=None):
    """ Calculate and round number of filters based on depth multiplier. """
    multiplier = width_coefficient
    if not multiplier:
        return filters
    divisor = depth_divisor
    min_depth = min_depth
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(
        min_depth, int(
            filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


class BlockDecoder(object):
    """ Block Decoder for readability, straight from the official TensorFlow repository """

    @staticmethod
    def _decode_block_string(block_string):
        """ Gets a block through a string notation of arguments. """
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # Check stride
        assert (('s' in options and len(options['s']) == 1) or
                (len(options['s']) == 2 and
                 options['s'][0] == options['s'][1]))

        return BlockArgs(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in block_string),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=int(options['s'][0]))

    @staticmethod
    def _encode_block_string(block):
        """Encodes a block to a string."""
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters
        ]
        if 0 < block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    @staticmethod
    def decode(string_list):
        """
        Decodes a list of string notations to specify blocks inside the network.
        :param string_list: a list of strings, each string is a notation of block
        :return: a list of BlockArgs namedtuples of block args
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args

    @staticmethod
    def encode(blocks_args):
        """
        Encodes a list of BlockArgs to a list of strings.
        :param blocks_args: a list of BlockArgs namedtuples of block args
        :return: a list of strings, each string is a notation of block
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(BlockDecoder._encode_block_string(block))
        return block_strings


def efficientnet_param():
    """ Creates a efficientnet model. """
    blocks_args = [
        'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
    ]
    blocks_args = BlockDecoder.decode(blocks_args)
    return blocks_args


class SamePadding(Block):
    def __init__(self, kernel_size, stride, dilation, **kwargs):
        super(SamePadding, self).__init__(**kwargs)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 2
        if isinstance(stride, int):
            stride = (stride,) * 2
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

    def forward(self, F, x):
        ih, iw = x.shape[-2:]
        kh, kw = self.kernel_size
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] +
                    (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] +
                    (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, mode='constant', pad_width=(0, 0, 0, 0, pad_w//2, pad_w -pad_w//2,
                                                     pad_h//2, pad_h - pad_h//2))
            return x
        return x


def _add_conv(out, channels=1, kernel=1, stride=1, pad=0,
              num_group=1, active=True, batchnorm=True):
    out.add(SamePadding(kernel, stride, dilation=(1, 1)))
    out.add(nn.Conv2D(channels, kernel, stride, pad, groups=num_group, use_bias=False))
    if batchnorm:
        out.add(nn.BatchNorm(scale=True, momentum=0.99, epsilon=1e-3))
    if active:
        out.add(nn.Swish())


class MBConv(nn.Block):
    def __init__(self, in_channels, channels, t, kernel, stride, se_ratio=0,
                 drop_connect_rate=0, **kwargs):

        r"""
            Parameters
            ----------
            int_channels: int, input channels.
            channels: int, output channels.
            t: int, the expand ratio used for increasing channels.
            kernel: int, filter size.
            stride: int, stride of the convolution.
            se_ratio:int, ratio of the squeeze layer and excitation layer.
            drop_connect_rate: int, drop rate of drop out.
        """
        super(MBConv, self).__init__(**kwargs)
        self.use_shortcut = stride == 1 and in_channels == channels
        self.se_ratio = se_ratio
        self.drop_connect_rate = drop_connect_rate
        with self.name_scope():
            self.out = nn.Sequential(prefix="out_")
            with self.out.name_scope():
                if t != 1:
                    _add_conv(
                        self.out,
                        in_channels * t,
                        active=True,
                        batchnorm=True)
                _add_conv(
                    self.out,
                    in_channels * t,
                    kernel=kernel,
                    stride=stride,
                    num_group=in_channels * t,
                    active=True,
                    batchnorm=True)
            if se_ratio:
                num_squeezed_channels = max(1, int(in_channels * se_ratio))
                self._se_reduce = nn.Sequential(prefix="se_reduce_")
                self._se_expand = nn.Sequential(prefix="se_expand_")
                with self._se_reduce.name_scope():
                    _add_conv(
                        self._se_reduce,
                        num_squeezed_channels,
                        active=False,
                        batchnorm=False)
                with self._se_expand.name_scope():
                    _add_conv(
                        self._se_expand,
                        in_channels * t,
                        active=False,
                        batchnorm=False)
            self.project_layer = nn.Sequential(prefix="project_layer_")
            with self.project_layer.name_scope():
                _add_conv(
                    self.project_layer,
                    channels,
                    active=False,
                    batchnorm=True)
            if drop_connect_rate:
                self.drop_out = nn.Dropout(drop_connect_rate)

    def forward(self, F, inputs):
        x = inputs
        x = self.out(x)
        if self.se_ratio:
            out = mx.nd.contrib.AdaptiveAvgPooling2D(x, 1)
            out = self._se_expand(self._se_reduce(out))
            out = mx.ndarray.sigmoid(out) * x
        out = self.project_layer(out)
        if self.use_shortcut:
            if self.drop_connect_rate:
                out = self.drop_out(out)
            out = F.elemwise_add(out, inputs)
        return out


class EfficientNet(nn.Block):

    def __init__(self, blocks_args=None,
                 dropout_rate=None,
                 num_classes=None,
                 width_coefficient=None,
                 depth_cofficient=None,
                 depth_divisor=None,
                 min_depth=None,
                 drop_connect_rate=None,
                 **kwargs):

        r"""EfficientNet model from the
            `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
            <https://arxiv.org/abs/1905.11946>`_ paper.

            Parameters
            ----------
            blocks_args: nametuple, it concludes the hyperparameters of the MBConv block.
            dropout_rate: float, rate of hidden units to drop.
            num_classes: int, number of output classes.
            width_coefficient:float, coefficient of the filters used for
            expanding or reducing the channels.
            depth_coefficient:float, it is used for repeat the EfficientNet Blocks.
            depth_divisor:int , it is used for reducing the number of filters.
            min_depth: int, used for deciding the minimum depth of the filters.
            drop_connect_rate: used for dropout.

            """
        super(EfficientNet, self).__init__(**kwargs)
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._blocks_args = blocks_args
        self.input_size = None
        with self.name_scope():
            self.features = nn.Sequential(prefix='features_')
            with self.features.name_scope():
                # stem conv
                out_channels = round_filters(32,
                                             width_coefficient,
                                             depth_divisor,
                                             min_depth)
                _add_conv(
                    self.features,
                    out_channels,
                    kernel=3,
                    stride=2,
                    active=True,
                    batchnorm=True)
            self._blocks = nn.Sequential(prefix='blocks_')
            with self._blocks.name_scope():
                for block_arg in self._blocks_args:
                    # Update block input and output filters based on depth
                    # multiplier.
                    block_arg = block_arg._replace(
                        input_filters=round_filters(
                            block_arg.input_filters,
                            width_coefficient,
                            depth_divisor,
                            min_depth),
                        output_filters=round_filters(
                            block_arg.output_filters,
                            width_coefficient,
                            depth_divisor,
                            min_depth),
                        num_repeat=round_repeats(
                            block_arg.num_repeat, depth_cofficient))
                    self._blocks.add(MBConv(block_arg.input_filters,
                                            block_arg.output_filters,
                                            block_arg.expand_ratio,
                                            block_arg.kernel_size,
                                            block_arg.stride,
                                            block_arg.se_ratio,
                                            drop_connect_rate)
                                     )
                    if block_arg.num_repeat > 1:
                        block_arg = block_arg._replace(
                            input_filters=block_arg.output_filters, stride=1)
                    for _ in range(block_arg.num_repeat - 1):
                        self._blocks.add(
                            MBConv(
                                block_arg.input_filters,
                                block_arg.output_filters,
                                block_arg.expand_ratio,
                                block_arg.kernel_size,
                                block_arg.stride,
                                block_arg.se_ratio,
                                drop_connect_rate))



            # Head
            out_channels = round_filters(1280, width_coefficient,
                                         depth_divisor, min_depth)
            self._conv_head = nn.Sequential(prefix='conv_head_')
            with self._conv_head.name_scope():
                _add_conv(
                    self._conv_head,
                    out_channels,
                    active=True,
                    batchnorm=True)
            # Final linear layer
            self._dropout = dropout_rate
            self._fc = nn.Dense(num_classes, use_bias=False)

    def forward(self, F, x):
        x = self.features(x)
        for block in self._blocks:
            x = block(x)
        x = self._conv_head(x)
        x = F.squeeze(F.squeeze(mx.nd.contrib.AdaptiveAvgPooling2D(x, 1), axis=-1), axis=-1)
        if self._dropout:
            x = F.Dropout(x, self._dropout)
        x = self._fc(x)
        return x


def efficientnet(dropout_rate=None,
                 num_classes=None,
                 width_coefficient=None,
                 depth_coefficient=None,
                 depth_divisor=None,
                 min_depth=None,
                 drop_connect_rate=None):

    blocks_args = efficientnet_param()
    model = EfficientNet(blocks_args,
                         dropout_rate,
                         num_classes,
                         width_coefficient,
                         depth_coefficient,
                         depth_divisor, min_depth,
                         drop_connect_rate)
    return model


def efficientnet_b0(pretrained=False,
                    dropout_rate=0.2,
                    classes=1000,
                    width_coefficient=1.0,
                    depth_coefficient=1.0,
                    depth_divisor=8,
                    min_depth=None,
                    drop_connect_rate=0.2,
                    ctx=mx.cpu()
                    ):
    r"""EfficientNet model from the
        `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
        <https://arxiv.org/abs/1905.11946>`_ paper.

        Parameters
        ----------
        pretrained : bool or str
            Boolean value controls whether to load the default pretrained weights for model.
            String value represents the hashtag for a certain version of pretrained weights.
        dropout_rate : float
            Rate of hidden units to drop.
        classes : int, number of output classes.
        width_coefficient : float
            Coefficient of the filters.
            Used for expanding or reducing the channels.
        depth_coefficient : float
            It is used for repeat the EfficientNet Blocks.
        depth_divisor:int
            It is used for reducing the number of filters.
        min_depth : int
            Used for deciding the minimum depth of the filters.
        drop_connect_rate : float
            Used for dropout.

        """
    if pretrained:
        pass
    model = efficientnet(dropout_rate,
                         classes,
                         width_coefficient,
                         depth_coefficient,
                         depth_divisor,
                         min_depth,
                         drop_connect_rate)
    model.collect_params().initialize(ctx=ctx)
    return model


def efficientnet_b1(pretrained=False,
                    dropout_rate=0.2,
                    classes=1000,
                    width_coefficient=1.0,
                    depth_coefficient=1.1,
                    depth_divisor=8,
                    min_depth=None,
                    drop_connect_rate=0.2,
                    ctx=mx.cpu(),
                    ):
    r"""EfficientNet model from the
            `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
            <https://arxiv.org/abs/1905.11946>`_ paper.

            Parameters
            ----------
            pretrained : bool or str
                Boolean value controls whether to load the default pretrained weights for model.
                String value represents the hashtag for a certain version of pretrained weights.
            dropout_rate : float
                Rate of hidden units to drop.
            classes : int, number of output classes.
            width_coefficient : float
                Coefficient of the filters.
                Used for expanding or reducing the channels.
            depth_coefficient : float
                It is used for repeat the EfficientNet Blocks.
            depth_divisor:int
                It is used for reducing the number of filters.
            min_depth : int
                Used for deciding the minimum depth of the filters.
            drop_connect_rate : float
                Used for dropout.

            """
    if pretrained:
        pass
    model = efficientnet(dropout_rate,
                         classes,
                         width_coefficient,
                         depth_coefficient,
                         depth_divisor,
                         min_depth,
                         drop_connect_rate)
    model.collect_params().initialize(ctx=ctx)
    return model


def efficientnet_b2(pretrained=False,
                    dropout_rate=0.3,
                    classes=1000,
                    width_coefficient=1.1,
                    depth_coefficient=1.2,
                    depth_divisor=8,
                    min_depth=None,
                    drop_connect_rate=0.2,
                    ctx=mx.cpu()
                    ):
    r"""EfficientNet model from the
            `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
            <https://arxiv.org/abs/1905.11946>`_ paper.

            Parameters
            ----------
            pretrained : bool or str
                Boolean value controls whether to load the default pretrained weights for model.
                String value represents the hashtag for a certain version of pretrained weights.
            dropout_rate : float
                Rate of hidden units to drop.
            classes : int, number of output classes.
            width_coefficient : float
                Coefficient of the filters.
                Used for expanding or reducing the channels.
            depth_coefficient : float
                It is used for repeat the EfficientNet Blocks.
            depth_divisor:int
                It is used for reducing the number of filters.
            min_depth : int
                Used for deciding the minimum depth of the filters.
            drop_connect_rate : float
                Used for dropout.

            """
    if pretrained:
        pass
    model = efficientnet(dropout_rate,
                         classes,
                         width_coefficient,
                         depth_coefficient,
                         depth_divisor,
                         min_depth,
                         drop_connect_rate)
    model.collect_params().initialize(ctx=ctx)
    return model


def efficientnet_b3(pretrained=False,
                    dropout_rate=0.3,
                    classes=1000,
                    width_coefficient=1.2,
                    depth_coefficient=1.4,
                    depth_divisor=8,
                    min_depth=None,
                    drop_connect_rate=0.2,
                    ctx=mx.cpu()
                    ):
    r"""EfficientNet model from the
            `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
            <https://arxiv.org/abs/1905.11946>`_ paper.

            Parameters
            ----------
            pretrained : bool or str
                Boolean value controls whether to load the default pretrained weights for model.
                String value represents the hashtag for a certain version of pretrained weights.
            dropout_rate : float
                Rate of hidden units to drop.
            classes : int, number of output classes.
            width_coefficient : float
                Coefficient of the filters.
                Used for expanding or reducing the channels.
            depth_coefficient : float
                It is used for repeat the EfficientNet Blocks.
            depth_divisor:int
                It is used for reducing the number of filters.
            min_depth : int
                Used for deciding the minimum depth of the filters.
            drop_connect_rate : float
                Used for dropout.

            """
    if pretrained:
        pass
    model = efficientnet(dropout_rate,
                         classes,
                         width_coefficient,
                         depth_coefficient,
                         depth_divisor,
                         min_depth,
                         drop_connect_rate)
    model.collect_params().initialize(ctx=ctx)
    return model


def efficientnet_b4(pretrained=False,
                    dropout_rate=0.4,
                    classes=1000,
                    width_coefficient=1.4,
                    depth_coefficient=1.8,
                    depth_divisor=8,
                    min_depth=None,
                    drop_connect_rate=0.2,
                    ctx=mx.cpu()
                    ):
    r"""EfficientNet model from the
            `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
            <https://arxiv.org/abs/1905.11946>`_ paper.

            Parameters
            ----------
            pretrained : bool or str
                Boolean value controls whether to load the default pretrained weights for model.
                String value represents the hashtag for a certain version of pretrained weights.
            dropout_rate : float
                Rate of hidden units to drop.
            classes : int, number of output classes.
            width_coefficient : float
                Coefficient of the filters.
                Used for expanding or reducing the channels.
            depth_coefficient : float
                It is used for repeat the EfficientNet Blocks.
            depth_divisor:int
                It is used for reducing the number of filters.
            min_depth : int
                Used for deciding the minimum depth of the filters.
            drop_connect_rate : float
                Used for dropout.

            """
    if pretrained:
        pass
    model = efficientnet(dropout_rate,
                         classes,
                         width_coefficient,
                         depth_coefficient,
                         depth_divisor,
                         min_depth,
                         drop_connect_rate,
                         )
    model.collect_params().initialize(ctx=ctx)
    return model


def efficientnet_b5(pretrained=False,
                    dropout_rate=0.4,
                    classes=1000,
                    width_coefficient=1.6,
                    depth_coefficient=2.2,
                    depth_divisor=8,
                    min_depth=None,
                    drop_connect_rate=0.2,
                    ctx=mx.cpu(),
                    ):
    r"""EfficientNet model from the
            `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
            <https://arxiv.org/abs/1905.11946>`_ paper.

            Parameters
            ----------
            pretrained : bool or str
                Boolean value controls whether to load the default pretrained weights for model.
                String value represents the hashtag for a certain version of pretrained weights.
            dropout_rate : float
                Rate of hidden units to drop.
            classes : int, number of output classes.
            width_coefficient : float
                Coefficient of the filters.
                Used for expanding or reducing the channels.
            depth_coefficient : float
                It is used for repeat the EfficientNet Blocks.
            depth_divisor:int
                It is used for reducing the number of filters.
            min_depth : int
                Used for deciding the minimum depth of the filters.
            drop_connect_rate : float
                Used for dropout.

            """
    if pretrained:
        pass
    model = efficientnet(dropout_rate,
                         classes,
                         width_coefficient,
                         depth_coefficient,
                         depth_divisor,
                         min_depth,
                         drop_connect_rate)
    model.collect_params().initialize(ctx=ctx)
    return model


def efficientnet_b6(pretrained=False,
                    dropout_rate=0.5,
                    classes=1000,
                    width_coefficient=1.8,
                    depth_coefficient=2.6,
                    depth_divisor=8,
                    min_depth=None,
                    drop_connect_rate=0.2,
                    ctx=mx.cpu()
                    ):
    r"""EfficientNet model from the
            `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
            <https://arxiv.org/abs/1905.11946>`_ paper.

            Parameters
            ----------
            pretrained : bool or str
                Boolean value controls whether to load the default pretrained weights for model.
                String value represents the hashtag for a certain version of pretrained weights.
            dropout_rate : float
                Rate of hidden units to drop.
            classes : int, number of output classes.
            width_coefficient : float
                Coefficient of the filters.
                Used for expanding or reducing the channels.
            depth_coefficient : float
                It is used for repeat the EfficientNet Blocks.
            depth_divisor:int
                It is used for reducing the number of filters.
            min_depth : int
                Used for deciding the minimum depth of the filters.
            drop_connect_rate : float
                Used for dropout.

            """
    if pretrained:
        pass
    model = efficientnet(dropout_rate,
                         classes,
                         width_coefficient,
                         depth_coefficient,
                         depth_divisor,
                         min_depth,
                         drop_connect_rate)
    model.collect_params().initialize(ctx=ctx)
    return model


def efficientnet_b7(pretrained=False,
                    dropout_rate=0.5,
                    classes=1000,
                    width_coefficient=2.0,
                    depth_coefficient=3.1,
                    depth_divisor=8,
                    min_depth=None,
                    drop_connect_rate=0.2,
                    ctx=mx.cpu()
                    ):
    r"""EfficientNet model from the
            `"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
            <https://arxiv.org/abs/1905.11946>`_ paper.

            Parameters
            ----------
            pretrained : bool or str
                Boolean value controls whether to load the default pretrained weights for model.
                String value represents the hashtag for a certain version of pretrained weights.
            dropout_rate : float
                Rate of hidden units to drop.
            classes : int, number of output classes.
            width_coefficient : float
                Coefficient of the filters.
                Used for expanding or reducing the channels.
            depth_coefficient : float
                It is used for repeat the EfficientNet Blocks.
            depth_divisor:int
                It is used for reducing the number of filters.
            min_depth : int
                Used for deciding the minimum depth of the filters.
            drop_connect_rate : float
                Used for dropout.

            """
    if pretrained:
        pass
    model = efficientnet(dropout_rate,
                         classes,
                         width_coefficient,
                         depth_coefficient,
                         depth_divisor,
                         min_depth,
                         drop_connect_rate,
                        )
    model.collect_params().initialize(ctx=ctx)
    return model
