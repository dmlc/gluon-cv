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
# pylint: disable= arguments-differ,unused-argument
"""ResNets, implemented in Gluon."""
from __future__ import division

__all__ = ['get_cifar_wide_resnet', 'cifar_wideresnet16_10',
           'cifar_wideresnet28_10', 'cifar_wideresnet40_8']

import os
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet import cpu

# Helpers
def _conv3x3(channels, stride, in_channels):
    return nn.Conv2D(channels, kernel_size=3, strides=stride, padding=1,
                     use_bias=False, in_channels=in_channels)

class CIFARBasicBlockV2(HybridBlock):
    r"""BasicBlock V2 from
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.
    This is used for ResNet V2 for 18, 34 layers.

    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    """
    def __init__(self, channels, stride, downsample=False, drop_rate=0.0, in_channels=0, **kwargs):
        super(CIFARBasicBlockV2, self).__init__(**kwargs)
        self.bn1 = nn.BatchNorm()
        self.conv1 = _conv3x3(channels, stride, in_channels)
        self.bn2 = nn.BatchNorm()
        self.conv2 = _conv3x3(channels, 1, channels)
        self.droprate = drop_rate
        if downsample:
            self.downsample = nn.Conv2D(channels, 1, stride, use_bias=False,
                                        in_channels=in_channels)
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        """Hybrid forward"""
        residual = x
        x = self.bn1(x)
        x = F.Activation(x, act_type='relu')
        if self.downsample:
            residual = self.downsample(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = F.Activation(x, act_type='relu')
        if self.droprate > 0:
            x = F.Dropout(x, self.droprate)
        x = self.conv2(x)

        return x + residual


class CIFARWideResNet(HybridBlock):
    r"""ResNet V2 model from
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    block : HybridBlock
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    channels : list of int
        Numbers of channels in each block. Length should be one larger than layers list.
    classes : int, default 10
        Number of classification classes.
    """
    def __init__(self, block, layers, channels, drop_rate, classes=10, **kwargs):
        super(CIFARWideResNet, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(nn.BatchNorm(scale=False, center=False))

            self.features.add(nn.Conv2D(channels[0], 3, 1, 1, use_bias=False))
            self.features.add(nn.BatchNorm())

            in_channels = channels[0]
            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                self.features.add(self._make_layer(block, num_layer, channels[i+1], drop_rate,
                                                   stride, i+1, in_channels=in_channels))
                in_channels = channels[i+1]
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))
            self.features.add(nn.GlobalAvgPool2D())
            self.features.add(nn.Flatten())
            self.output = nn.Dense(classes)

    def _make_layer(self, block, layers, channels, drop_rate, stride, stage_index, in_channels=0):
        layer = nn.HybridSequential(prefix='stage%d_'%stage_index)
        with layer.name_scope():
            layer.add(block(channels, stride, channels != in_channels, drop_rate,
                            in_channels=in_channels, prefix=''))
            for _ in range(layers-1):
                layer.add(block(channels, 1, False, drop_rate, in_channels=channels, prefix=''))
        return layer

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x

# Constructor
def get_cifar_wide_resnet(num_layers, width_factor=1, drop_rate=0.0,
                          pretrained=False, ctx=cpu(),
                          root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    r"""ResNet V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    ResNet V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    num_layers : int
        Numbers of layers. Needs to be an integer in the form of 6*n+2, e.g. 20, 56, 110, 164.
    width_factor: int
        The width factor to apply to the number of channels from the original resnet.
    drop_rate: float
        The rate of dropout.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    assert (num_layers - 4) % 6 == 0

    n = (num_layers - 4) // 6
    layers = [n] * 3
    channels = [16, 16*width_factor, 32*width_factor, 64*width_factor]

    net = CIFARWideResNet(CIFARBasicBlockV2, layers, channels, drop_rate, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        net.load_params(get_model_file('cifar_wideresnet%d_%d'%(num_layers, width_factor),
                                       root=root), ctx=ctx)
    return net

def cifar_wideresnet16_10(**kwargs):
    r"""WideResNet-16-10 model for CIFAR10 from `"Wide Residual Networks"
    <https://arxiv.org/abs/1605.07146>`_ paper.

    Parameters
    ----------
    drop_rate: float
        The rate of dropout.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_cifar_wide_resnet(16, 10, **kwargs)

def cifar_wideresnet28_10(**kwargs):
    r"""WideResNet-28-10 model for CIFAR10 from `"Wide Residual Networks"
    <https://arxiv.org/abs/1605.07146>`_ paper.

    Parameters
    ----------
    drop_rate: float
        The rate of dropout.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_cifar_wide_resnet(28, 10, **kwargs)

def cifar_wideresnet40_8(**kwargs):
    r"""WideResNet-40-8 model for CIFAR10 from `"Wide Residual Networks"
    <https://arxiv.org/abs/1605.07146>`_ paper.

    Parameters
    ----------
    drop_rate: float
        The rate of dropout.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_cifar_wide_resnet(40, 8, **kwargs)
