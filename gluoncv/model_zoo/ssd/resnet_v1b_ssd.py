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
"""ResNet V1b series modified for SSD detection"""
from __future__ import absolute_import

import mxnet as mx
from mxnet.gluon import HybridBlock
from mxnet.gluon.nn import HybridSequential, Conv2D, Activation, BatchNorm
from ..model_zoo import get_model


class ResNetV1bSSD(HybridBlock):
    """Single-shot Object Detection Network: https://arxiv.org/abs/1512.02325.
       with resnetv1b base model.

    Parameters
    ----------
    network : string or None
        Name of the base network, must be end with v1b.
    add_filters : list of int
        Number of channels for the appended layers, ignored if `network`is `None`.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
        This will only apply to base networks that has `norm_layer` specified, will ignore if the
        base network (e.g. VGG) don't accept this argument.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    use_bn : bool
        Whether to use BatchNorm layer after each attached convolutional layer.
    reduce_ratio : float
        Channel reduce ratio (0, 1) of the transition layer.
    min_depth : int
        Minimum channels for the transition layers.
    """
    def __init__(self, network, add_filters,
                 norm_layer=BatchNorm, norm_kwargs=None,
                 use_bn=False, reduce_ratio=1.0, min_depth=128, **kwargs):
        super(ResNetV1bSSD, self).__init__()
        assert network.endswith('v1b')
        if norm_kwargs is None:
            norm_kwargs = {}
        res = get_model(network, **kwargs)
        weight_init = mx.init.Xavier(rnd_type='gaussian', factor_type='out', magnitude=2)
        with self.name_scope():
            self.stage1 = HybridSequential('stage1')
            for l in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2']:
                self.stage1.add(getattr(res, l))
            self.stage2 = HybridSequential('stage2')
            self.stage2.add(res.layer3)
            # set stride from (2, 2) -> (1, 1) in first conv of layer3
            self.stage2[0][0].conv1._kwargs['stride'] = (1, 1)
            # also the residuel path
            self.stage2[0][0].downsample[0]._kwargs['stride'] = (1, 1)
            self.stage2.add(res.layer4)
            self.more_stages = HybridSequential('more_stages')
            for i, num_filter in enumerate(add_filters):
                stage = HybridSequential('more_stages_' + str(i))
                num_trans = max(min_depth, int(round(num_filter * reduce_ratio)))
                stage.add(Conv2D(channels=num_trans, kernel_size=1, use_bias=not use_bn,
                                 weight_initializer=weight_init))
                if use_bn:
                    stage.add(norm_layer(**norm_kwargs))
                stage.add(Activation('relu'))
                padding = 0 if i == len(add_filters) - 1 else 1
                stage.add(Conv2D(channels=num_filter, kernel_size=3,
                                 strides=2, padding=padding, use_bias=not use_bn,
                                 weight_initializer=weight_init))
                if use_bn:
                    stage.add(norm_layer(**norm_kwargs))
                stage.add(Activation('relu'))
                self.more_stages.add(stage)

    def hybrid_forward(self, F, x):
        y1 = self.stage1(x)
        y2 = self.stage2(y1)
        more_out = [y1, y2]
        out = y2
        for stage in self.more_stages:
            out = stage(out)
            more_out.append(out)
        return more_out

def resnet34_v1b_ssd(**kwargs):
    return ResNetV1bSSD(network='resnet34_v1b', add_filters=[256, 256, 128, 128], **kwargs)
