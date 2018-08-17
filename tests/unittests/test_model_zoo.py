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
import mxnet as mx
import numpy as np

import gluoncv as gcv
from common import try_gpu, with_cpu

def test_get_all_models():
    names = gcv.model_zoo.get_model_list()
    for name in names:
        kwargs = {}
        if 'custom' in name:
            kwargs['classes'] = ['a', 'b']
        net = gcv.model_zoo.get_model(name, pretrained=False, **kwargs)
        assert isinstance(net, mx.gluon.Block), '{}'.format(name)

@with_cpu(0)
def _test_model_list(model_list, ctx, x, pretrained=True, **kwargs):
    for model in model_list:
        net = gcv.model_zoo.get_model(model, **kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            net.initialize()
        net.collect_params().reset_ctx(ctx)
        net(x)
        mx.nd.waitall()

    pretrained_models = gcv.model_zoo.pretrained_model_list()
    for model in model_list:
        if model in pretrained_models:
            net = gcv.model_zoo.get_model(model, pretrained=True, **kwargs)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                net.initialize()
            net.collect_params().reset_ctx(ctx)
            net(x)
            mx.nd.waitall()

@with_cpu(0)
def _test_bn_global_stats(model_list, **kwargs):
    class _BatchNorm(mx.gluon.nn.BatchNorm):
        def __init__(self, axis=1, momentum=0.9, epsilon=1e-5, center=True, scale=True,
            use_global_stats=False, beta_initializer='zeros', gamma_initializer='ones',
            running_mean_initializer='zeros', running_variance_initializer='ones',
            in_channels=0, **kwargs):
            assert use_global_stats
            super(_BatchNorm, self).__init__(axis, momentum, epsilon, center, scale,
                 use_global_stats, beta_initializer, gamma_initializer,
                 running_mean_initializer, running_variance_initializer,
                 in_channels, **kwargs)

    for model in model_list:
        gcv.model_zoo.get_model(model, norm_layer=_BatchNorm, use_global_stats=True, **kwargs)

def test_classification_models():
    ctx = mx.context.current_context()
    x = mx.random.uniform(shape=(2, 3, 32, 32), ctx=ctx)
    cifar_models = [
        'cifar_resnet20_v1', 'cifar_resnet56_v1', 'cifar_resnet110_v1',
        'cifar_resnet20_v2', 'cifar_resnet56_v2', 'cifar_resnet110_v2',
        'cifar_wideresnet16_10', 'cifar_wideresnet28_10', 'cifar_wideresnet40_8',
        'cifar_resnext29_32x4d', 'cifar_resnext29_16x64d',
    ]
    _test_model_list(cifar_models, ctx, x)

def test_imagenet_models():
    ctx = mx.context.current_context()

    # 224x224
    x = mx.random.uniform(shape=(2, 3, 224, 224), ctx=ctx)
    models = ['resnet18_v1', 'resnet34_v1',
              'resnet50_v1', 'resnet101_v1', 'resnet152_v1',
              'resnet18_v1b', 'resnet34_v1b',
              'resnet50_v1b', 'resnet101_v1b', 'resnet152_v1b',
              'resnet50_v1c', 'resnet101_v1c', 'resnet152_v1c',
              'resnet50_v1d', 'resnet101_v1d', 'resnet152_v1d',
              'resnet50_v1e', 'resnet101_v1e', 'resnet152_v1e',
              'resnet18_v2', 'resnet34_v2',
              'resnet50_v2', 'resnet101_v2', 'resnet152_v2',
              'resnext50_32x4d', 'resnext101_32x4d', 'resnext101_64x4d',
              'se_resnext50_32x4d', 'se_resnext101_32x4d', 'se_resnext101_64x4d',
              'se_resnet18_v1', 'se_resnet34_v1', 'se_resnet50_v1',
              'se_resnet101_v1', 'se_resnet152_v1',
              'se_resnet18_v2', 'se_resnet34_v2', 'se_resnet50_v2',
              'se_resnet101_v2', 'se_resnet152_v2',
              'senet_52', 'senet_103', 'senet_154',
              'alexnet', 'densenet121', 'densenet161', 'densenet169', 'densenet201',
              'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn',
              'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']
    _test_model_list(models, ctx, x)

    # 299x299
    x = mx.random.uniform(shape=(2, 3, 299, 299), ctx=ctx)
    models = ['inceptionv3', 'nasnet_5_1538', 'nasnet_7_1920', 'nasnet_6_4032']
    _test_model_list(models, ctx, x)

    # 331x331
    x = mx.random.uniform(shape=(2, 3, 331, 331), ctx=ctx)
    models = ['nasnet_5_1538', 'nasnet_7_1920', 'nasnet_6_4032']
    _test_model_list(models, ctx, x)

def test_imagenet_models_bn_global_stats():
    models = ['resnet18_v1b', 'resnet34_v1b', 'resnet50_v1b',
              'resnet101_v1b', 'resnet152_v1b']
    _test_bn_global_stats(models)

def test_ssd_models():
    ctx = mx.context.current_context()
    x = mx.random.uniform(shape=(1, 3, 512, 544), ctx=ctx)  # allow non-squre and larger inputs
    models = ['ssd_300_vgg16_atrous_voc', 'ssd_512_vgg16_atrous_voc', 'ssd_512_resnet50_v1_voc']
    if not mx.context.current_context().device_type == 'gpu':
        models = ['ssd_512_resnet50_v1_voc']
    _test_model_list(models, ctx, x)

def test_faster_rcnn_models():
    ctx = mx.context.current_context()
    x = mx.random.uniform(shape=(1, 3, 600, 800), ctx=ctx)  # allow non-squre and larger inputs
    models = ['faster_rcnn_resnet50_v1b_voc', 'faster_rcnn_resnet50_v1b_coco']
    _test_model_list(models, ctx, x)

def test_mask_rcnn_models():
    ctx = mx.context.current_context()
    x = mx.random.uniform(shape=(1, 3, 600, 800), ctx=ctx)
    models = ['mask_rcnn_resnet50_v1b_coco']
    _test_model_list(models, ctx, x)

def test_yolo3_models():
    ctx = mx.context.current_context()
    x = mx.random.uniform(shape=(1, 3, 416, 416), ctx=ctx)  # allow non-squre and larger inputs
    models = ['yolo3_darknet53_voc']
    _test_model_list(models, ctx, x)

def test_set_nms():
    model_list = ['ssd_512_resnet50_v1_voc', 'faster_rcnn_resnet50_v1b_voc', 'yolo3_darknet53_coco']
    for model in model_list:
        net = gcv.model_zoo.get_model(model, pretrained=False, pretrained_base=False)
        net.initialize()
        net.hybridize()
        ctx = mx.context.current_context()
        x = mx.random.uniform(shape=(1, 3, 608, 768), ctx=ctx)
        net.set_nms(nms_thresh=0.45, nms_topk=400, post_nms=100)
        net(x)
        net.set_nms(nms_thresh=0.3, nms_topk=200, post_nms=50)
        net(x)

def test_segmentation_models():
    ctx = mx.context.current_context()
    x = mx.random.uniform(shape=(2, 3, 480, 480), ctx=ctx)
    models = ['fcn_resnet50_voc', 'fcn_resnet101_voc', 'fcn_resnet50_ade']
    _test_model_list(models, ctx, x, pretrained=True, pretrained_base=True)
    _test_model_list(models, ctx, x, pretrained=False, pretrained_base=False)
    _test_model_list(models, ctx, x, pretrained=False, pretrained_base=True)

if __name__ == '__main__':
    import nose
    nose.runmodule()
