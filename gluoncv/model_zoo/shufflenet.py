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
# pylint: disable= line-too-long,arguments-differ,unused-argument,missing-docstring,too-many-function-args
# pylint: disable= line-too-long
"""ShuffleNetV1 and ShuffleNetV2, implemented in Gluon."""
from mxnet.context import cpu
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm
from mxnet.gluon.block import HybridBlock

__all__ = [
    'ShuffleNetV1',
    'shufflenet_v1',
    'get_shufflenet_v1',
    'ShuffleNetV2',
    'shufflenet_v2',
    'get_shufflenet_v2']

def _conv2d(channel, kernel=1, padding=0, stride=1, num_group=1, use_act=True, use_bias=True, norm_layer=BatchNorm, norm_kwargs=None):
    cell = nn.HybridSequential(prefix='')
    cell.add(nn.Conv2D(channel, kernel_size=kernel, strides=stride, padding=padding, groups=num_group, use_bias=use_bias))
    cell.add(norm_layer(epsilon=1e-5, momentum=0.9, **({} if norm_kwargs is None else norm_kwargs)))
    if use_act:
        cell.add(nn.Activation('relu'))
    return cell


class shuffleUnit(HybridBlock):
    def __init__(self, in_channels, out_channels, combine_type, groups=3, grouped_conv=True,
                 norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(shuffleUnit, self).__init__(**kwargs)
        if combine_type == 'add':
            self.DWConv_stride = 1
        elif combine_type == 'concat':
            self.DWConv_stride = 2
            out_channels -= in_channels
        self.first_groups = groups if grouped_conv else 1
        self.bottleneck_channels = out_channels // 4
        self.grouped_conv = grouped_conv
        self.output_channel = out_channels
        self.groups = groups
        self.combine_type = combine_type
        with self.name_scope():
            self.conv_beforshuffle = nn.HybridSequential()
            self.conv_beforshuffle.add(_conv2d(channel=self.bottleneck_channels, kernel=1, stride=1, num_group=self.first_groups))
            self.conv_aftershuffle = nn.HybridSequential()
            self.conv_aftershuffle.add(_conv2d(channel=self.bottleneck_channels, kernel=3, padding=1, stride=self.DWConv_stride, num_group=self.bottleneck_channels, use_act=False))
            self.conv_aftershuffle.add(_conv2d(channel=self.output_channel, kernel=1, stride=1, num_group=groups, use_act=False))

    def combine(self, F, branch1, branch2, combine):
        if combine == 'add':
            data = branch1 + branch2
            data = F.Activation(data, act_type='relu')
        elif combine == 'concat':
            data = F.concat(branch1, branch2, dim=1)
            data = F.Activation(data, act_type='relu')
        return data

    def channel_shuffle(self, F, data, groups):
        data = F.reshape(data, shape=(0, -4, groups, -1, -2))
        data = F.swapaxes(data, 1, 2)
        data = F.reshape(data, shape=(0, -3, -2))
        return data

    def hybrid_forward(self, F, x):
        res = x
        x = self.conv_beforshuffle(x)
        if self.grouped_conv:
            x = self.channel_shuffle(F, x, groups=self.groups)
        x = self.conv_aftershuffle(x)
        if self.combine_type == 'concat':
            res = F.Pooling(data=res, kernel=(3, 3), pool_type='avg', stride=(2, 2), pad=(1, 1))
        x = self.combine(F, res, x, combine=self.combine_type)
        return x


class ShuffleNetV1(HybridBlock):
    def __init__(self, groups=3, classes=1000, norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(ShuffleNetV1, self).__init__(**kwargs)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(nn.Conv2D(24, kernel_size=3, strides=2, padding=1))
            self.features.add(nn.MaxPool2D(pool_size=3, strides=2, padding=1))
            self.features.add(self.make_stage(2))
            self.features.add(self.make_stage(3))
            self.features.add(self.make_stage(4))
            self.features.add(nn.GlobalAvgPool2D())
            self.output = nn.Dense(classes)

    def make_stage(self, stage, groups=3):
        stage_repeats = [3, 7, 3]
        grouped_conv = stage > 2
        if groups == 1:
            out_channels = [-1, 24, 144, 288, 567]
        elif groups == 2:
            out_channels = [-1, 24, 200, 400, 800]
        elif groups == 3:
            out_channels = [-1, 24, 240, 480, 960]
        elif groups == 4:
            out_channels = [-1, 24, 272, 544, 1088]
        elif groups == 8:
            out_channels = [-1, 24, 384, 768, 1536]
        body = nn.HybridSequential()
        body.add(shuffleUnit(out_channels[stage - 1], out_channels[stage], 'concat', groups, grouped_conv))
        for i in range(stage_repeats[stage - 2]):
            body.add(shuffleUnit(out_channels[stage], out_channels[stage], 'add', groups, True))
        return body

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


class shuffleUnitV2(HybridBlock):
    def __init__(self, in_channels, out_channels, split, norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(shuffleUnitV2, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.equal_channels = out_channels // 2
        self.split = split
        if split:
            self.DWConv_stride = 1
        else:
            self.DWConv_stride = 2
            with self.name_scope():
                self.branch1_conv = nn.HybridSequential()
                self.branch1_conv.add(_conv2d(channel=self.in_channels, kernel=3, padding=1, stride=self.DWConv_stride, num_group=self.in_channels, use_act=False, use_bias=False))
                self.branch1_conv.add(_conv2d(channel=self.equal_channels, kernel=1, stride=1, use_act=True, use_bias=False))

        with self.name_scope():
            self.branch2_conv = nn.HybridSequential()
            self.branch2_conv.add(_conv2d(channel=self.equal_channels, kernel=1, stride=1, use_act=True, use_bias=False))
            self.branch2_conv.add(_conv2d(channel=self.equal_channels, kernel=3, padding=1, stride=self.DWConv_stride, num_group=self.equal_channels, use_act=False, use_bias=False))
            self.branch2_conv.add(_conv2d(channel=self.equal_channels, kernel=1, stride=1, use_act=True, use_bias=False))

    def channel_shuffle(self, F, data, groups):
        data = F.reshape(data, shape=(0, -4, groups, -1, -2))
        data = F.swapaxes(data, 1, 2)
        data = F.reshape(data, shape=(0, -3, -2))
        return data

    def hybrid_forward(self, F, x):
        if self.split:
            branch1 = F.slice_axis(x, axis=1, begin=0, end=self.in_channels // 2)
            branch2 = F.slice_axis(x, axis=1, begin=self.in_channels // 2, end=self.in_channels)
        else:
            branch1 = x
            branch2 = x
            branch1 = self.branch1_conv(branch1)
        branch2 = self.branch2_conv(branch2)
        x = F.concat(branch1, branch2, dim=1)
        x = self.channel_shuffle(F, data=x, groups=2)
        return x

class ShuffleNetV2(HybridBlock):
    def __init__(self, classes=1000, norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(ShuffleNetV2, self).__init__(**kwargs)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(_conv2d(channel=24, kernel=3, stride=2, padding=1, use_act=True, use_bias=False))
            self.features.add(nn.MaxPool2D(pool_size=3, strides=2, padding=1))
            self.features.add(self.make_stage(2))
            self.features.add(self.make_stage(3))
            self.features.add(self.make_stage(4))
            self.features.add(_conv2d(channel=1024, kernel=1, stride=1, use_act=True, use_bias=False))
            self.features.add(nn.GlobalAvgPool2D())
            self.output = nn.Dense(classes)

    def make_stage(self, stage, multiplier=1):
        stage_repeats = [3, 7, 3]
        if multiplier == 0.5:
            out_channels = [-1, 24, 48, 96, 192]
        elif multiplier == 1:
            out_channels = [-1, 24, 116, 232, 464]
        elif multiplier == 1.5:
            out_channels = [-1, 24, 176, 352, 704]
        elif multiplier == 2:
            out_channels = [-1, 24, 244, 488, 976]
        body = nn.HybridSequential()
        body.add(shuffleUnitV2(out_channels[stage - 1], out_channels[stage], split=False))
        for i in range(stage_repeats[stage - 2]):
            body.add(shuffleUnitV2(out_channels[stage], out_channels[stage], split=True))
        return body

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


def get_shufflenet_v1(pretrained=False, root='~/.mxnet/models', ctx=cpu(), norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
    net = ShuffleNetV1(norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)
    from ..data import ImageNet1kAttr
    attrib = ImageNet1kAttr()
    net.synset = attrib.synset
    net.classes = attrib.classes
    net.classes_long = attrib.classes_long
    return net

def shufflenet_v1(**kwargs):
    return get_shufflenet_v1(**kwargs)

def get_shufflenet_v2(pretrained=False, root='~/.mxnet/models', ctx=cpu(), norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
    net = ShuffleNetV2(norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)
    from ..data import ImageNet1kAttr
    attrib = ImageNet1kAttr()
    net.synset = attrib.synset
    net.classes = attrib.classes
    net.classes_long = attrib.classes_long
    return net

def shufflenet_v2(**kwargs):
    return get_shufflenet_v2(**kwargs)
