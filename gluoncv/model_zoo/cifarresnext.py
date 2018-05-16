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

__all__ = ['get_cifar_resnext', 'cifar_resnext29_32x4d', 'cifar_resnext29_16x64d']

import os
import math
from mxnet import cpu
from mxnet.gluon import nn
from mxnet.gluon.block import HybridBlock


class CIFARBlock(HybridBlock):
    r"""Bottleneck Block from `"Aggregated Residual Transformations for Deep Neural Networks"
    <http://arxiv.org/abs/1611.05431>`_ paper.

    Parameters
    ----------
    cardinality: int
        Number of groups
    bottleneck_width: int
        Width of bottleneck block
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    """
    def __init__(self, channels, cardinality, bottleneck_width,
                 stride, downsample=False, **kwargs):
        super(CIFARBlock, self).__init__(**kwargs)
        D = int(math.floor(channels * (bottleneck_width / 64)))
        group_width = cardinality * D

        self.body = nn.HybridSequential(prefix='')
        self.body.add(nn.Conv2D(group_width, kernel_size=1, use_bias=False))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Activation('relu'))
        self.body.add(nn.Conv2D(group_width, kernel_size=3, strides=stride, padding=1,
                                groups=cardinality, use_bias=False))
        self.body.add(nn.BatchNorm())
        self.body.add(nn.Activation('relu'))
        self.body.add(nn.Conv2D(channels * 4, kernel_size=1, use_bias=False))
        self.body.add(nn.BatchNorm())

        if downsample:
            self.downsample = nn.HybridSequential(prefix='')
            self.downsample.add(nn.Conv2D(channels * 4, kernel_size=1, strides=stride,
                                          use_bias=False))
            self.downsample.add(nn.BatchNorm())
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        """Hybrid forward"""
        residual = x

        x = self.body(x)

        if self.downsample:
            residual = self.downsample(residual)

        x = F.Activation(residual+x, act_type='relu')

        return x

# Nets
class CIFARResNext(HybridBlock):
    r"""ResNext model from `"Aggregated Residual Transformations for Deep Neural Networks"
    <http://arxiv.org/abs/1611.05431>`_ paper.

    Parameters
    ----------
    layers : list of int
        Numbers of layers in each block
    cardinality: int
        Number of groups
    bottleneck_width: int
        Width of bottleneck block
    classes : int, default 10
        Number of classification classes.
    """
    def __init__(self, layers, cardinality, bottleneck_width, classes=10, **kwargs):
        super(CIFARResNext, self).__init__(**kwargs)
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        channels = 64

        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(nn.Conv2D(channels, 3, 1, 1, use_bias=False))
            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))

            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                self.features.add(self._make_layer(channels, num_layer, stride, i+1))
                channels *= 2
            self.features.add(nn.AvgPool2D(8))

            self.output = nn.Dense(classes)

    def _make_layer(self, channels, num_layer, stride, stage_index):
        layer = nn.HybridSequential(prefix='stage%d_'%stage_index)
        with layer.name_scope():
            layer.add(CIFARBlock(channels, self.cardinality, self.bottleneck_width,
                                 stride, True, prefix=''))
            for _ in range(num_layer-1):
                layer.add(CIFARBlock(channels, self.cardinality, self.bottleneck_width,
                                     1, False, prefix=''))
        return layer

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)

        return x


# Constructor
def get_cifar_resnext(num_layers, cardinality=16, bottleneck_width=64,
                      pretrained=False, ctx=cpu(),
                      root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    r"""ResNext model from `"Aggregated Residual Transformations for Deep Neural Networks"
    <http://arxiv.org/abs/1611.05431>`_ paper.

    Parameters
    ----------
    num_layers : int
        Numbers of layers. Needs to be an integer in the form of 9*n+2, e.g. 29
    cardinality: int
        Number of groups
    bottleneck_width: int
        Width of bottleneck block
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    assert (num_layers - 2) % 9 == 0
    layer = (num_layers - 2) // 9
    layers = [layer] * 3
    net = CIFARResNext(layers, cardinality, bottleneck_width, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        net.load_params(get_model_file('cifar_resnext%d_%dx%dd'%(num_layers, cardinality,
                                                                 bottleneck_width),
                                       root=root), ctx=ctx)
    return net

def cifar_resnext29_32x4d(**kwargs):
    r"""ResNext-29 32x4d model from `"Aggregated Residual Transformations for Deep Neural Networks"
    <http://arxiv.org/abs/1611.05431>`_ paper.

    Parameters
    ----------
    num_layers : int
        Numbers of layers. Needs to be an integer in the form of 9*n+2, e.g. 29
    cardinality: int
        Number of groups
    bottleneck_width: int
        Width of bottleneck block
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_cifar_resnext(29, 32, 4, **kwargs)

def cifar_resnext29_16x64d(**kwargs):
    r"""ResNext-29 16x64d model from `"Aggregated Residual Transformations for Deep Neural Networks"
    <http://arxiv.org/abs/1611.05431>`_ paper.

    Parameters
    ----------
    cardinality: int
        Number of groups
    bottleneck_width: int
        Width of bottleneck block
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """

    return get_cifar_resnext(29, 16, 64, **kwargs)
