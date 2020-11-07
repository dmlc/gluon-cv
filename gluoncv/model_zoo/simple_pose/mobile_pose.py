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
"""Pose Estimation for Mobile Device, implemented in Gluon."""

from __future__ import division

from mxnet import initializer
from mxnet.gluon import nn
from mxnet.gluon.block import HybridBlock
from mxnet.context import cpu

from ...nn.block import DUC

class MobilePose(HybridBlock):
    """Pose Estimation for Mobile Device"""
    def __init__(self, base_name, base_attrs=('features',), num_joints=17,
                 pretrained_base=False, pretrained_ctx=cpu(), **kwargs):
        super(MobilePose, self).__init__(**kwargs)

        with self.name_scope():
            from ..model_zoo import get_model
            base_model = get_model(base_name, pretrained=pretrained_base,
                                   ctx=pretrained_ctx)
            self.features = nn.HybridSequential()
            if base_name.startswith('mobilenetv2'):
                self.features.add(base_model.features[:-1])
            elif base_name.startswith('mobilenetv3'):
                self.features.add(base_model.features[:-4])
            elif base_name.startswith('mobilenet'):
                self.features.add(base_model.features[:-2])
            else:
                for layer in base_attrs:
                    self.features.add(getattr(base_model, layer))

            self.upsampling = nn.HybridSequential()
            self.upsampling.add(
                nn.Conv2D(256, 1, 1, 0, use_bias=False),
                DUC(512, 2),
                DUC(256, 2),
                DUC(128, 2),
                nn.Conv2D(num_joints, 1, use_bias=False,
                          weight_initializer=initializer.Normal(0.001)),
            )

    def hybrid_forward(self, F, x):
        x = self.features(x)
        hm = self.upsampling(x)

        return hm

def get_mobile_pose(base_name, ctx=cpu(), pretrained=False,
                    root='~/.mxnet/models', **kwargs):
    net = MobilePose(base_name, **kwargs)

    if pretrained:
        from ..model_store import get_model_file
        net.load_parameters(get_model_file('mobile_pose_%s'%(base_name),
                                           tag=pretrained, root=root), ctx=ctx)

    return net

def mobile_pose_resnet18_v1b(**kwargs):
    return get_mobile_pose('resnet18_v1b', base_attrs=['conv1', 'bn1', 'relu', 'maxpool',
                                                       'layer1', 'layer2', 'layer3', 'layer4'],
                           **kwargs)

def mobile_pose_resnet50_v1b(**kwargs):
    return get_mobile_pose('resnet50_v1b', base_attrs=['conv1', 'bn1', 'relu', 'maxpool',
                                                       'layer1', 'layer2', 'layer3', 'layer4'],
                           **kwargs)

def mobile_pose_mobilenet1_0(**kwargs):
    return get_mobile_pose('mobilenet1.0', base_attrs=['features'], **kwargs)

def mobile_pose_mobilenetv2_1_0(**kwargs):
    return get_mobile_pose('mobilenetv2_1.0', base_attrs=['features'], **kwargs)

def mobile_pose_mobilenetv3_small(**kwargs):
    return get_mobile_pose('mobilenetv3_small', base_attrs=['features'], **kwargs)

def mobile_pose_mobilenetv3_large(**kwargs):
    return get_mobile_pose('mobilenetv3_large', base_attrs=['features'], **kwargs)
