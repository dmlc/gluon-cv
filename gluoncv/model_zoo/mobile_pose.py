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

import numpy as np
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm
from mxnet.gluon.block import HybridBlock
from mxnet.context import cpu
from ..nn import ReLU6, HardSigmoid, HardSwish

from .mobilenetv3 import _ResUnit
from .model_zoo import get_model

class MobilePose(HybridBlock):
    """Pose Estimation for Mobile Device"""
    def __init__(self, num_joints, base_name, num_stages=6, use_pretrained_base=False, **kwargs):
        super(MobilePose, self).__init__(**kwargs)
        self.num_stages = num_stages
        assert self.num_stages >= 1

        with self.name_scope():
            base_model = get_model(baes_name, pretrained=use_pretrained_base)
            self.base_model1 = base_model.features[:4]
            self.base_model_pool1 = nn.MaxPool2D(4, 4)
            self.base_model2 = base_model.features[4:6]
            self.base_model_pool2 = nn.MaxPool2D(2, 2)
            self.base_model3 = base_model.features[6:9]
            self.base_model4 = base_model.features[9:15]
            self.base_model5 = base_model.features[15:21]

            self.branch = nn.HybridSequential(prefix='')
            num_in = 128
            for _ in range(self.num_stages):
                if _ == 0:
                    self.branch.add(_ResUnit(num_in, num_in*4, num_joints,
                                             kernel_size=3, strides=4))
                else:
                    self.branch.add(_ResUnit(num_in, num_in*4, num_joints,
                                             kernel_size=7, strides=4))

    def hybrid_forward(self, F, x):
        x1 = self.base_model1(x)
        x2 = self.base_model2(x1)
        x3 = self.base_model3(x2)
        x4 = self.base_model4(x3)
        x5 = self.base_model5(x4)

        x_concat = F.concat(self.base_model_pool1(x1),
                            self.base_model_pool2(x2),
                            x3,
                            F.contrib.BilinearResize2D(x4, scale_height=2, scale_width=2),
                            F.contrib.BilinearResize2D(x5, scale_height=4, scale_width=4),
                            dim=1)

        res = []
        for i, cell in enumerate(self.branch._children.values()):
            x_out = cell(x_concat)
            x_out = F.contrib.BilinearResize2D(x_out, scale_height=4, scale_width=4)
            x_concat = F.concat(x_concat,
                                x_out,
                                dim=1)
            res.append(x_out)

        return res

def get_mobilepose(num_joints, base_name, ctx=cpu(),
                       root='~/.mxnet/models', **kwargs):
    net = MobilePose(num_joints, base_name, **kwargs)

    if pretrained:
        from .model_store import get_model_file
        net.load_parameters(get_model_file('mobilepose_%s'%(base_name),
                                           tag=pretrained, root=root), ctx=ctx)

    return net

def mobilepose(**kwargs):
    return get_mobilepose(17, 'mobilenetv3_large', **kwargs)
