# coding: utf-8

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

from __future__ import print_function

import warnings
import logging

import torch
import torch.nn as nn

from gluoncv.torch.model_zoo import get_model, get_model_list
from gluoncv.torch.engine.config import get_cfg_defaults


def test_get_all_models():
    cfg = get_cfg_defaults()
    names = get_model_list()
    for name in names:
        cfg.CONFIG.MODEL.NAME = name
        net = get_model(cfg)
        assert isinstance(net, nn.Module), '{}'.format(name)


def _test_model_list(model_list, use_cuda, x, pretrained, num_classes, **kwargs):
    cfg = get_cfg_defaults()
    for model in model_list:
        cfg.CONFIG.MODEL.NAME = model
        cfg.CONFIG.MODEL.PRETRAINED = pretrained
        cfg.CONFIG.DATA.NUM_CLASSES = num_classes
        net = get_model(cfg)

        if use_cuda:
            net.cuda()
            x = x.cuda()

        net(x)


def test_action_recognition_resnet_models():
    use_cuda = True
    if torch.cuda.device_count() == 0:
        use_cuda = False

    models = ['resnet18_v1b_kinetics400', 'resnet34_v1b_kinetics400', 'resnet50_v1b_kinetics400',
              'resnet101_v1b_kinetics400', 'resnet152_v1b_kinetics400']

    # single video frame with 224x224
    x = torch.rand(1, 3, 1, 224, 224)
    _test_model_list(models, use_cuda, x, pretrained=False, num_classes=400)
    _test_model_list(models, use_cuda, x, pretrained=True, num_classes=400)

    # multiple video frames with 224x224
    x = torch.rand(5, 3, 5, 224, 224)
    _test_model_list(models, use_cuda, x, pretrained=False, num_classes=400)
    _test_model_list(models, use_cuda, x, pretrained=True, num_classes=400)

    models = ['resnet50_v1b_sthsthv2']

    # single video frame with 224x224
    x = torch.rand(1, 3, 1, 224, 224)
    _test_model_list(models, use_cuda, x, pretrained=False, num_classes=174)
    _test_model_list(models, use_cuda, x, pretrained=True, num_classes=174)

    # multiple video frames with 224x224
    x = torch.rand(5, 3, 5, 224, 224)
    _test_model_list(models, use_cuda, x, pretrained=False, num_classes=174)
    _test_model_list(models, use_cuda, x, pretrained=True, num_classes=174)


def test_action_recognition_r2plus1d_models():
    use_cuda = True
    if torch.cuda.device_count() == 0:
        use_cuda = False

    models = ['r2plus1d_v1_resnet18_kinetics400', 'r2plus1d_v1_resnet34_kinetics400',
              'r2plus1d_v1_resnet50_kinetics400', 'r2plus1d_v2_resnet152_kinetics400',
              'r2plus1d_v1_resnet101_kinetics400', 'r2plus1d_v1_resnet152_kinetics400']

    x = torch.rand(2, 3, 16, 112, 112)
    _test_model_list(models, use_cuda, x, pretrained=False, num_classes=400)


    models = ['r2plus1d_v1_resnet18_kinetics400', 'r2plus1d_v1_resnet34_kinetics400',
              'r2plus1d_v1_resnet50_kinetics400', 'r2plus1d_v2_resnet152_kinetics400']
    _test_model_list(models, use_cuda, x, pretrained=True, num_classes=400)


def test_action_recognition_i3d_models():
    use_cuda = True
    if torch.cuda.device_count() == 0:
        use_cuda = False

    models = ['i3d_resnet50_v1_kinetics400', 'i3d_resnet101_v1_kinetics400',
              'i3d_nl5_resnet50_v1_kinetics400', 'i3d_nl10_resnet50_v1_kinetics400',
              'i3d_nl5_resnet101_v1_kinetics400', 'i3d_nl10_resnet101_v1_kinetics400']

    x = torch.rand(2, 3, 32, 224, 224)
    _test_model_list(models, use_cuda, x, pretrained=True, num_classes=400)
    _test_model_list(models, use_cuda, x, pretrained=False, num_classes=400)

    models = ['i3d_resnet50_v1_sthsthv2']

    x = torch.rand(2, 3, 8, 224, 224)
    _test_model_list(models, use_cuda, x, pretrained=True, num_classes=174)
    _test_model_list(models, use_cuda, x, pretrained=False, num_classes=174)


def test_action_recognition_slowfast_models():
    use_cuda = True
    if torch.cuda.device_count() == 0:
        use_cuda = False

    models = ['slowfast_4x16_resnet50_kinetics400', 'slowfast_4x16_resnet101_kinetics400']
    x = torch.rand(2, 3, 32, 224, 224)
    _test_model_list(models, use_cuda, x, pretrained=False, num_classes=400)

    models = ['slowfast_4x16_resnet50_kinetics400']
    _test_model_list(models, use_cuda, x, pretrained=True, num_classes=400)

    models = ['slowfast_8x8_resnet50_kinetics400', 'slowfast_8x8_resnet101_kinetics400']
    x = torch.rand(2, 3, 32, 224, 224)
    _test_model_list(models, use_cuda, x, pretrained=False, num_classes=400)
    _test_model_list(models, use_cuda, x, pretrained=True, num_classes=400)

    models = ['slowfast_16x8_resnet101_kinetics400', 'slowfast_16x8_resnet101_50_50_kinetics400']
    x = torch.rand(2, 3, 64, 224, 224)
    _test_model_list(models, use_cuda, x, pretrained=False, num_classes=400)

    models = ['slowfast_16x8_resnet50_sthsthv2']
    x = torch.rand(2, 3, 64, 224, 224)
    _test_model_list(models, use_cuda, x, pretrained=False, num_classes=174)
    _test_model_list(models, use_cuda, x, pretrained=True, num_classes=174)


def test_action_recognition_i3d_slow_models():
    use_cuda = True
    if torch.cuda.device_count() == 0:
        use_cuda = False

    models = ['i3d_slow_resnet50_f32s2_kinetics400', 'i3d_slow_resnet101_f32s2_kinetics400']
    x = torch.rand(2, 3, 32, 224, 224)
    _test_model_list(models, use_cuda, x, pretrained=False, num_classes=400)
    _test_model_list(models, use_cuda, x, pretrained=True, num_classes=400)

    models = ['i3d_slow_resnet50_f8s8_kinetics400', 'i3d_slow_resnet101_f8s8_kinetics400']
    x = torch.rand(2, 3, 8, 224, 224)
    _test_model_list(models, use_cuda, x, pretrained=False, num_classes=400)
    _test_model_list(models, use_cuda, x, pretrained=True, num_classes=400)

    models = ['i3d_slow_resnet50_f16s4_kinetics400', 'i3d_slow_resnet101_f16s4_kinetics400']
    x = torch.rand(2, 3, 16, 224, 224)
    _test_model_list(models, use_cuda, x, pretrained=False, num_classes=400)
    _test_model_list(models, use_cuda, x, pretrained=True, num_classes=400)


def test_action_recognition_tpn_models():
    use_cuda = True
    if torch.cuda.device_count() == 0:
        use_cuda = False

    models = ['tpn_resnet50_f32s2_kinetics400', 'tpn_resnet101_f32s2_kinetics400']
    x = torch.rand(2, 3, 32, 224, 224)
    _test_model_list(models, use_cuda, x, pretrained=False, num_classes=400)
    _test_model_list(models, use_cuda, x, pretrained=True, num_classes=400)

    models = ['tpn_resnet50_f8s8_kinetics400', 'tpn_resnet101_f8s8_kinetics400']
    x = torch.rand(2, 3, 8, 224, 224)
    _test_model_list(models, use_cuda, x, pretrained=False, num_classes=400)
    _test_model_list(models, use_cuda, x, pretrained=True, num_classes=400)

    models = ['tpn_resnet50_f16s4_kinetics400', 'tpn_resnet101_f16s4_kinetics400']
    x = torch.rand(2, 3, 16, 224, 224)
    _test_model_list(models, use_cuda, x, pretrained=False, num_classes=400)
    _test_model_list(models, use_cuda, x, pretrained=True, num_classes=400)


if __name__ == '__main__':
    import nose

    nose.runmodule()
