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

import mxnet as mx
from mxnet.contrib.quantization import *
from ..unittests.common import try_gpu, with_cpu

import gluoncv as gcv


def test_get_all_models():
    names = gcv.model_zoo.get_model_list()
    for name in names:
        kwargs = {}
        if 'custom' in name:
            kwargs['classes'] = ['a', 'b']
        if 'dcnv2' in name and not hasattr(mx.gluon.contrib.cnn, 'ModulatedDeformableConvolution'):
            continue
        net = gcv.model_zoo.get_model(name, pretrained=False, **kwargs)
        assert isinstance(net, mx.gluon.Block), '{}'.format(name)


def _test_model_list(model_list, ctx, x, pretrained=True, **kwargs):
    pretrained_models = gcv.model_zoo.pretrained_model_list()
    for model in model_list:
        if model in pretrained_models:
            net = gcv.model_zoo.get_model(model, pretrained=True, **kwargs)
        else:
            net = gcv.model_zoo.get_model(model, **kwargs)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                net.initialize()
        net.collect_params().reset_ctx(ctx)
        net(x)
        mx.nd.waitall()

def _calib_model_list(model_list, ctx, x, pretrained=True, **kwargs):
    pretrained_models = gcv.model_zoo.pretrained_model_list()
    for model in model_list:
        if model in pretrained_models:
            net = gcv.model_zoo.get_model(model, pretrained=True, **kwargs)
        else:
            net = gcv.model_zoo.get_model(model, **kwargs)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                net.initialize()
        net.collect_params().reset_ctx(ctx)
        exclude_layers_match = ['flatten']
        if model.find('ssd') != -1 or model.find('psp') != -1 or model.find('deeplab') != -1:
            exclude_layers_match += ['concat']
        random_label = mx.random.uniform(shape=(x.shape[0],1))
        dataset = mx.gluon.data.dataset.ArrayDataset(x, random_label)
        calib_data = mx.gluon.data.DataLoader(dataset, batch_size=1)
        dummy_logger = logging.getLogger('dummy')
        dummy_logger.setLevel(logging.ERROR)
        net = quantize_net(net, quantized_dtype='auto',
                           exclude_layers=None,
                           exclude_layers_match=exclude_layers_match,
                           calib_data=calib_data, calib_mode='naive',
                           num_calib_examples=1, ctx=ctx, logger=dummy_logger)
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


@try_gpu(0)
def test_classification_models():
    ctx = mx.context.current_context()
    x = mx.random.uniform(shape=(2, 3, 32, 32), ctx=ctx)
    cifar_models = [
        'cifar_resnet20_v1', 'cifar_resnet56_v1', 'cifar_resnet110_v1',
        'cifar_resnet20_v2', 'cifar_resnet56_v2', 'cifar_resnet110_v2',
        'cifar_wideresnet16_10', 'cifar_wideresnet28_10', 'cifar_wideresnet40_8',
        'cifar_resnext29_32x4d', 'cifar_resnext29_16x64d',
        'cifar_residualattentionnet56', 'cifar_residualattentionnet92',
        'cifar_residualattentionnet452'
    ]
    _test_model_list(cifar_models, ctx, x)


@try_gpu(0)
def test_imagenet_models():
    ctx = mx.context.current_context()

    # 224x224
    x = mx.random.uniform(shape=(2, 3, 224, 224), ctx=ctx)
    models = ['resnet18_v1', 'resnet34_v1',
              'resnet50_v1', 'resnet101_v1', 'resnet152_v1',
              'resnet18_v1b', 'resnet34_v1b', 'resnet50_v1b_gn',
              'resnet50_v1b', 'resnet101_v1b', 'resnet152_v1b',
              'resnet50_v1c', 'resnet101_v1c', 'resnet152_v1c',
              'resnet50_v1d', 'resnet101_v1d', 'resnet152_v1d',
              'resnet50_v1e', 'resnet101_v1e', 'resnet152_v1e',
              'resnet18_v2', 'resnet34_v2',
              'resnet50_v2', 'resnet101_v2', 'resnet152_v2',
              'resnest50', 'resnest101', 'resnest200', 'resnest269',
              'resnext50_32x4d', 'resnext101_32x4d', 'resnext101_64x4d',
              'se_resnext50_32x4d', 'se_resnext101_32x4d', 'se_resnext101_64x4d',
              'se_resnet18_v1', 'se_resnet34_v1', 'se_resnet50_v1',
              'se_resnet101_v1', 'se_resnet152_v1',
              'se_resnet18_v2', 'se_resnet34_v2', 'se_resnet50_v2',
              'se_resnet101_v2', 'se_resnet152_v2',
              'senet_154', 'squeezenet1.0', 'squeezenet1.1',
              'mobilenet1.0', 'mobilenet0.75', 'mobilenet0.5', 'mobilenet0.25',
              'mobilenetv2_1.0', 'mobilenetv2_0.75', 'mobilenetv2_0.5', 'mobilenetv2_0.25',
              'mobilenetv3_large', 'mobilenetv3_small',
              'densenet121', 'densenet161', 'densenet169', 'densenet201',
              'darknet53', 'alexnet', 'googlenet',
              'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn',
              'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn',
              'residualattentionnet56', 'residualattentionnet92',
              'residualattentionnet128', 'residualattentionnet164',
              'residualattentionnet200', 'residualattentionnet236', 'residualattentionnet452',
              'resnet18_v1b_0.89', 'resnet50_v1d_0.86', 'resnet50_v1d_0.48',
              'resnet50_v1d_0.37', 'resnet50_v1d_0.11',
              'resnet101_v1d_0.76', 'resnet101_v1d_0.73']
    _test_model_list(models, ctx, x)

    # 299x299
    x = mx.random.uniform(shape=(2, 3, 299, 299), ctx=ctx)
    models = ['inceptionv3', 'nasnet_5_1538', 'nasnet_7_1920', 'nasnet_6_4032', 'xception']
    _test_model_list(models, ctx, x)

    # 331x331
    x = mx.random.uniform(shape=(2, 3, 331, 331), ctx=ctx)
    models = ['nasnet_5_1538', 'nasnet_7_1920', 'nasnet_6_4032']
    _test_model_list(models, ctx, x)


@try_gpu(0)
def test_simple_pose_resnet_models():
    ctx = mx.context.current_context()
    models = ['simple_pose_resnet18_v1b',
              'simple_pose_resnet50_v1b', 'simple_pose_resnet101_v1b', 'simple_pose_resnet152_v1b',
              'simple_pose_resnet50_v1d', 'simple_pose_resnet101_v1d', 'simple_pose_resnet152_v1d',
              'mobile_pose_resnet18_v1b', 'mobile_pose_resnet50_v1b',
              'mobile_pose_mobilenet1.0', 'mobile_pose_mobilenetv2_1.0',
              'mobile_pose_mobilenetv3_large', 'mobile_pose_mobilenetv3_small']

    # 192x256
    x = mx.random.uniform(shape=(2, 3, 192, 256), ctx=ctx)
    _test_model_list(models, ctx, x)

    # 256x256
    x = mx.random.uniform(shape=(2, 3, 256, 256), ctx=ctx)
    _test_model_list(models, ctx, x)

    # 288x384
    x = mx.random.uniform(shape=(2, 3, 288, 384), ctx=ctx)
    _test_model_list(models, ctx, x)

@try_gpu(0)
def test_alpha_pose_resnet_models():
    ctx = mx.context.current_context()
    models = ['alpha_pose_resnet101_v1b_coco']

    # 256x320
    x = mx.random.uniform(shape=(2, 3, 256, 320), ctx=ctx)
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

def test_center_net_models():
    ctx = mx.context.current_context()
    x = mx.random.uniform(shape=(1, 3, 512, 512), ctx=ctx)  # allow non-squre and larger inputs
    models = ['center_net_resnet18_v1b_voc', 'center_net_resnet50_v1b_coco']
    if not mx.context.current_context().device_type == 'gpu':
        models = ['center_net_resnet18_v1b_voc']
    if hasattr(mx.gluon.contrib.cnn, 'ModulatedDeformableConvolution'):
        models.append('center_net_resnet18_v1b_dcnv2_voc')
    _test_model_list(models, ctx, x)

def test_ssd_reset_class():
    ctx = mx.context.current_context()
    x = mx.random.uniform(shape=(1, 3, 512, 544), ctx=ctx)  # allow non-squre and larger inputs
    model_name = 'ssd_300_vgg16_atrous_voc'
    net = gcv.model_zoo.get_model(model_name, pretrained=True, ctx=ctx)
    net.reset_class(["bus", "car", "bird"], reuse_weights=["bus", "car", "bird"])
    net = gcv.model_zoo.get_model(model_name, pretrained=True, ctx=ctx)
    net.reset_class(["bus", "car", "bird"], reuse_weights={"bus": "bus"})
    net = gcv.model_zoo.get_model(model_name, pretrained=True, ctx=ctx)
    net.reset_class(["person", "car", "bird"], reuse_weights={"person": 14})
    net = gcv.model_zoo.get_model(model_name, pretrained=True, ctx=ctx)
    net.reset_class(["person", "car", "bird"], reuse_weights={0: 14})
    net = gcv.model_zoo.get_model(model_name, pretrained=True, ctx=ctx)
    net.reset_class(["person", "car", "bird"], reuse_weights={0: "person"})
    test_classes = ['bird', 'bicycle', 'bus', 'car', 'cat']
    test_classes_dict = dict(zip(test_classes, test_classes))
    net = gcv.model_zoo.get_model(model_name, pretrained=True, ctx=ctx)
    net.reset_class(test_classes, reuse_weights=test_classes_dict)
    net(x)


# This test is only executed when a gpu is available
def test_ssd_reset_class_on_gpu():
    ctx = mx.gpu(0)
    try:
        x = mx.random.uniform(shape=(1, 3, 512, 544), ctx=ctx)
    except Exception:
        return

    model_name = 'ssd_300_vgg16_atrous_voc'
    net = gcv.model_zoo.get_model(model_name, pretrained=True, ctx=ctx)
    net.reset_class(["bus", "car", "bird"], reuse_weights=["bus", "car", "bird"])
    net(x)


def test_yolo3_reset_class():
    test_classes = ['bird', 'bicycle', 'bus', 'car', 'cat']
    test_classes_dict = dict(zip(test_classes, test_classes))

    ctx = mx.context.current_context()
    x = mx.random.uniform(shape=(1, 3, 512, 544), ctx=ctx)  # allow non-square and larger inputs
    model_name = 'yolo3_darknet53_voc'
    net = gcv.model_zoo.get_model(model_name, pretrained=True, ctx=ctx)
    net.hybridize()
    net.reset_class(["bus", "car", "bird"], reuse_weights=["bus", "car", "bird"])
    net(x)
    mx.nd.waitall()

    net = gcv.model_zoo.get_model(model_name, pretrained=True, ctx=ctx)
    net.hybridize()
    net.reset_class(test_classes, reuse_weights=test_classes_dict)
    net(x)
    mx.nd.waitall()

    # for GPU
    ctx = mx.gpu(0)
    try:
        x = mx.random.uniform(shape=(1, 3, 512, 544), ctx=ctx)
    except Exception:
        return
    net = gcv.model_zoo.get_model(model_name, pretrained=True, ctx=ctx)
    net.hybridize()
    net.reset_class(["bus", "car", "bird"])
    net(x)
    mx.nd.waitall()

    net = gcv.model_zoo.get_model(model_name, pretrained=True, ctx=ctx)
    net.hybridize()
    net.reset_class(test_classes, reuse_weights=test_classes_dict)
    net(x)
    mx.nd.waitall()


def test_faster_rcnn_reset_class():
    ctx = mx.context.current_context()
    x = mx.random.uniform(shape=(1, 3, 512, 544), ctx=ctx)  # allow non-squre and larger inputs
    model_name = 'faster_rcnn_resnet50_v1b_coco'
    net = gcv.model_zoo.get_model(model_name, pretrained=True, ctx=ctx)
    net.reset_class(["bus", "car", "bird"], reuse_weights=["bus", "car", "bird"])
    net = gcv.model_zoo.get_model(model_name, pretrained=True, ctx=ctx)
    net.reset_class(["bus", "car", "bird"], reuse_weights={"bus": "bus"})
    net = gcv.model_zoo.get_model(model_name, pretrained=True, ctx=ctx)
    net.reset_class(["person", "car", "bird"], reuse_weights={"person": 14})
    net = gcv.model_zoo.get_model(model_name, pretrained=True, ctx=ctx)
    net.reset_class(["person", "car", "bird"], reuse_weights={0: 14})
    net = gcv.model_zoo.get_model(model_name, pretrained=True, ctx=ctx)
    net.reset_class(["person", "car", "bird"], reuse_weights={0: "person"})
    net(x)

    # for GPU
    ctx = mx.gpu(0)
    try:
        x = mx.random.uniform(shape=(1, 3, 512, 544), ctx=ctx)
    except Exception:
        return
    net = gcv.model_zoo.get_model(model_name, pretrained=True, ctx=ctx)
    net.reset_class(["bus", "car", "bird"])
    net(x)


def test_mask_rcnn_reset_class():
    ctx = mx.context.current_context()
    x = mx.random.uniform(shape=(1, 3, 512, 544), ctx=ctx)  # allow non-squre and larger inputs
    model_name = 'mask_rcnn_resnet50_v1b_coco'
    net = gcv.model_zoo.get_model(model_name, pretrained=True, ctx=ctx)
    net.reset_class(["bus", "car", "bird"], reuse_weights=["bus", "car", "bird"])
    net = gcv.model_zoo.get_model(model_name, pretrained=True, ctx=ctx)
    net.reset_class(["bus", "car", "bird"], reuse_weights={"bus": "bus"})
    net = gcv.model_zoo.get_model(model_name, pretrained=True, ctx=ctx)
    net.reset_class(["person", "car", "bird"], reuse_weights={"person": 14})
    net = gcv.model_zoo.get_model(model_name, pretrained=True, ctx=ctx)
    net.reset_class(["person", "car", "bird"], reuse_weights={0: 14})
    net = gcv.model_zoo.get_model(model_name, pretrained=True, ctx=ctx)
    net.reset_class(["person", "car", "bird"], reuse_weights={0: "person"})
    net(x)

    # for GPU
    ctx = mx.gpu(0)
    try:
        x = mx.random.uniform(shape=(1, 3, 512, 544), ctx=ctx)
    except Exception:
        return
    net = gcv.model_zoo.get_model(model_name, pretrained=True, ctx=ctx)
    net.reset_class(["bus", "car", "bird"])
    net(x)


@try_gpu(0)
def test_faster_rcnn_models():
    ctx = mx.context.current_context()
    x = mx.random.uniform(shape=(1, 3, 300, 400), ctx=ctx)  # allow non-squre and larger inputs
    models = ['faster_rcnn_resnet50_v1b_voc', 'faster_rcnn_resnet50_v1b_coco',
              'faster_rcnn_fpn_resnet50_v1b_coco']
    _test_model_list(models, ctx, x)


@try_gpu(0)
def test_mask_rcnn_models():
    ctx = mx.context.current_context()
    x = mx.random.uniform(shape=(1, 3, 300, 400), ctx=ctx)
    models = ['mask_rcnn_resnet50_v1b_coco', 'mask_rcnn_fpn_resnet50_v1b_coco',
              'mask_rcnn_resnet18_v1b_coco', 'mask_rcnn_fpn_resnet18_v1b_coco']
    _test_model_list(models, ctx, x)


@try_gpu(0)
def test_rcnn_max_dets_greater_than_nms_mask_rcnn_models():
    ctx = mx.context.current_context()
    x = mx.random.uniform(shape=(1, 3, 300, 400), ctx=ctx)
    net = gcv.model_zoo.mask_rcnn_resnet18_v1b_coco(pretrained=False, pretrained_base=True,
                                                    rcnn_max_dets=1000, rpn_test_pre_nms=100,
                                                    rpn_test_post_nms=30)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net.initialize()
    net.collect_params().reset_ctx(ctx)
    net(x)
    mx.nd.waitall()


def test_yolo3_models():
    ctx = mx.context.current_context()
    x = mx.random.uniform(shape=(1, 3, 320, 320), ctx=ctx)  # allow non-squre and larger inputs
    models = ['yolo3_darknet53_voc']
    _test_model_list(models, ctx, x)


@try_gpu(0)
def test_two_stage_ctx_loading():
    model_name = 'yolo3_darknet53_coco'
    ctx = mx.test_utils.default_context()
    if str(ctx).startswith('cpu'):
        # meaningless
        return
    net = gcv.model_zoo.get_model(model_name, pretrained=True)
    net.save_parameters(model_name + '.params')
    net = gcv.model_zoo.get_model(model_name, pretrained=False, ctx=ctx)
    net.load_parameters(model_name + '.params', ctx=ctx)


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


@try_gpu(0)
def test_segmentation_models():
    ctx = mx.context.current_context()
    x = mx.random.uniform(shape=(1, 3, 480, 480), ctx=ctx)
    models = ['fcn_resnet101_coco', 'psp_resnet101_coco', 'deeplab_resnet101_coco',
              'fcn_resnet101_voc', 'psp_resnet101_voc', 'deeplab_resnet101_voc',
              'fcn_resnet50_ade', 'psp_resnet50_ade', 'deeplab_resnet50_ade',
              'fcn_resnet101_ade', 'psp_resnet101_ade', 'deeplab_resnet101_ade',
              'deeplab_resnest200_ade',
              'psp_resnet101_citys', 'deeplab_resnet152_voc', 'deeplab_resnet152_coco',
              'deeplab_v3b_plus_wideresnet_citys', 'icnet_resnet50_citys',
              'deeplab_resnet50_citys', 'deeplab_resnet101_citys']
    _test_model_list(models, ctx, x, pretrained=True, pretrained_base=True)
    _test_model_list(models, ctx, x, pretrained=False, pretrained_base=False)
    _test_model_list(models, ctx, x, pretrained=False, pretrained_base=True)


@try_gpu(0)
def test_segmentation_models_custom_size():
    ctx = mx.context.current_context()
    num_classes = 5
    width = 96
    height = 64
    x = mx.random.uniform(shape=(1, 3, height, width), ctx=ctx)

    net = gcv.model_zoo.FCN(num_classes, backbone='resnet50', aux=False, ctx=ctx, pretrained_base=True,
                            height=height, width=width)
    result = net.forward(x)
    assert result[0].shape == (1, num_classes, height, width)
    net = gcv.model_zoo.PSPNet(num_classes, backbone='resnet50', aux=False, ctx=ctx, pretrained_base=True,
                               height=height, width=width)
    result = net.forward(x)
    assert result[0].shape == (1, num_classes, height, width)

    net = gcv.model_zoo.DeepLabV3(num_classes, backbone='resnet50', aux=False, ctx=ctx, pretrained_base=True,
                               height=height, width=width)
    result = net.forward(x)
    assert result[0].shape == (1, num_classes, height, width)

    net = gcv.model_zoo.DeepLabV3Plus(num_classes, backbone='resnet50', aux=False, ctx=ctx, pretrained_base=True,
                                  height=height, width=width)
    result = net.forward(x)
    assert result[0].shape == (1, num_classes, height, width)

    net = gcv.model_zoo.DeepLabWV3Plus(num_classes, backbone='resnet50', aux=False, ctx=ctx, pretrained_base=True,
                                  height=height, width=width)
    result = net.forward(x)
    assert result[0].shape == (1, num_classes, height, width)

@try_gpu(0)
def test_action_recognition_vgg_models():
    ctx = mx.context.current_context()
    models = ['vgg16_ucf101']

    # 224x224
    x = mx.random.uniform(shape=(2, 3, 224, 224), ctx=ctx)
    _test_model_list(models, ctx, x, pretrained=True, pretrained_base=True)
    _test_model_list(models, ctx, x, pretrained=False, pretrained_base=False)
    _test_model_list(models, ctx, x, pretrained=False, pretrained_base=True)

@try_gpu(0)
def test_action_recognition_inceptionv1_models():
    ctx = mx.context.current_context()
    models = ['inceptionv1_kinetics400']

    # 299x299
    x = mx.random.uniform(shape=(2, 3, 224, 224), ctx=ctx)
    _test_model_list(models, ctx, x, pretrained=True, pretrained_base=True)
    _test_model_list(models, ctx, x, pretrained=False, pretrained_base=False)
    _test_model_list(models, ctx, x, pretrained=False, pretrained_base=True)

@try_gpu(0)
def test_action_recognition_inceptionv3_models():
    ctx = mx.context.current_context()
    models = ['inceptionv3_ucf101', 'inceptionv3_kinetics400']

    # 299x299
    x = mx.random.uniform(shape=(2, 3, 299, 299), ctx=ctx)    # only allow 299x299 input
    _test_model_list(models, ctx, x, pretrained=True, pretrained_base=True)
    _test_model_list(models, ctx, x, pretrained=False, pretrained_base=False)
    _test_model_list(models, ctx, x, pretrained=False, pretrained_base=True)

@try_gpu(0)
def test_action_recognition_resnet_models():
    ctx = mx.context.current_context()
    models = ['resnet18_v1b_kinetics400', 'resnet34_v1b_kinetics400', 'resnet50_v1b_kinetics400',
              'resnet101_v1b_kinetics400', 'resnet152_v1b_kinetics400', 'resnet50_v1b_sthsthv2',
              'resnet50_v1b_hmdb51']

    # 224x224
    x = mx.random.uniform(shape=(2, 3, 224, 224), ctx=ctx)
    _test_model_list(models, ctx, x, pretrained=True, pretrained_base=True)
    _test_model_list(models, ctx, x, pretrained=False, pretrained_base=False)
    _test_model_list(models, ctx, x, pretrained=False, pretrained_base=True)

@try_gpu(0)
def test_action_recognition_c3d_models():
    ctx = mx.context.current_context()
    models = ['c3d_kinetics400']

    # 112x112
    x = mx.random.uniform(shape=(2, 3, 16, 112, 112), ctx=ctx)
    _test_model_list(models, ctx, x, pretrained=True, pretrained_base=True)
    _test_model_list(models, ctx, x, pretrained=False, pretrained_base=False)
    _test_model_list(models, ctx, x, pretrained=False, pretrained_base=True)

@try_gpu(0)
def test_action_recognition_p3d_models():
    ctx = mx.context.current_context()
    models = ['p3d_resnet50_kinetics400', 'p3d_resnet101_kinetics400']

    # 224x224
    x = mx.random.uniform(shape=(2, 3, 16, 224, 224), ctx=ctx)
    _test_model_list(models, ctx, x, pretrained=True, pretrained_base=True)
    _test_model_list(models, ctx, x, pretrained=False, pretrained_base=False)
    _test_model_list(models, ctx, x, pretrained=False, pretrained_base=True)

@try_gpu(0)
def test_action_recognition_r2plus1d_models():
    ctx = mx.context.current_context()
    models = ['r2plus1d_resnet18_kinetics400', 'r2plus1d_resnet34_kinetics400',
              'r2plus1d_resnet50_kinetics400']

    # 112x112
    x = mx.random.uniform(shape=(2, 3, 16, 112, 112), ctx=ctx)
    _test_model_list(models, ctx, x, pretrained=True, pretrained_base=True)
    _test_model_list(models, ctx, x, pretrained=False, pretrained_base=False)
    _test_model_list(models, ctx, x, pretrained=False, pretrained_base=True)

@try_gpu(0)
def test_action_recognition_i3d_models():
    ctx = mx.context.current_context()
    models = ['i3d_resnet50_v1_kinetics400', 'i3d_resnet101_v1_kinetics400', 'i3d_inceptionv1_kinetics400',
              'i3d_inceptionv3_kinetics400', 'i3d_resnet50_v1_sthsthv2', 'i3d_resnet50_v1_hmdb51',
              'i3d_resnet50_v1_ucf101', 'i3d_nl5_resnet50_v1_kinetics400', 'i3d_nl10_resnet50_v1_kinetics400',
              'i3d_nl5_resnet101_v1_kinetics400', 'i3d_nl10_resnet101_v1_kinetics400']

    # 224x224
    x = mx.random.uniform(shape=(2, 3, 32, 224, 224), ctx=ctx)
    _test_model_list(models, ctx, x, pretrained=True, pretrained_base=True)
    _test_model_list(models, ctx, x, pretrained=False, pretrained_base=False)
    _test_model_list(models, ctx, x, pretrained=False, pretrained_base=True)

@try_gpu(0)
def test_action_recognition_slowfast_models():
    ctx = mx.context.current_context()

    models = ['slowfast_4x16_resnet50_kinetics400']
    x = mx.random.uniform(shape=(2, 3, 36, 224, 224), ctx=ctx)
    _test_model_list(models, ctx, x, pretrained=True, pretrained_base=True)
    _test_model_list(models, ctx, x, pretrained=False, pretrained_base=False)
    _test_model_list(models, ctx, x, pretrained=False, pretrained_base=True)

    models = ['slowfast_8x8_resnet50_kinetics400']
    x = mx.random.uniform(shape=(2, 3, 40, 224, 224), ctx=ctx)
    _test_model_list(models, ctx, x, pretrained=True, pretrained_base=True)
    _test_model_list(models, ctx, x, pretrained=False, pretrained_base=False)
    _test_model_list(models, ctx, x, pretrained=False, pretrained_base=True)

    models = ['slowfast_8x8_resnet101_kinetics400']
    x = mx.random.uniform(shape=(2, 3, 40, 224, 224), ctx=ctx)
    _test_model_list(models, ctx, x, pretrained=True, pretrained_base=True)
    _test_model_list(models, ctx, x, pretrained=False, pretrained_base=False)
    _test_model_list(models, ctx, x, pretrained=False, pretrained_base=True)

@with_cpu(0)
def test_mobilenet_sync_bn():
    model_name = "mobilenet1.0"
    net = gcv.model_zoo.get_model(model_name, pretrained=True)
    net.save_parameters(model_name + '.params')
    net = gcv.model_zoo.get_model(model_name, pretrained=False,
                                  norm_layer=mx.gluon.contrib.nn.SyncBatchNorm,
                                  norm_kwargs={'num_devices': 2})
    net.load_parameters(model_name + '.params')

@with_cpu(0)
def test_quantized_imagenet_models():
    model_list = ['mobilenet1.0_int8', 'resnet50_v1_int8']
    ctx = mx.context.current_context()
    x = mx.random.uniform(shape=(1, 3, 224, 224), ctx=ctx)
    _test_model_list(model_list, ctx, x)

@with_cpu(0)
def test_quantized_ssd_models():
    model_list = ['ssd_300_vgg16_atrous_voc_int8', 'ssd_512_mobilenet1.0_voc_int8',
                  'ssd_512_resnet50_v1_voc_int8', 'ssd_512_vgg16_atrous_voc_int8']
    ctx = mx.context.current_context()
    x = mx.random.uniform(shape=(1, 3, 512, 544), ctx=ctx)
    _test_model_list(model_list, ctx, x)

@with_cpu(0)
def test_calib_models():
    model_list = ['resnet50_v1', 'resnet50_v1d_0.11',
                  'mobilenet1.0', 'mobilenetv2_1.0',
                  'squeezenet1.0', 'squeezenet1.1',
                  'vgg16']
    ctx = mx.context.current_context()
    x = mx.random.uniform(shape=(1, 3, 224, 224), ctx=ctx)
    _calib_model_list(model_list, ctx, x)

    model_list = ['inceptionv3']
    x = mx.random.uniform(shape=(1, 3, 299, 299), ctx=ctx)
    _calib_model_list(model_list, ctx, x)

    model_list = ['ssd_300_vgg16_atrous_voc', 'ssd_512_mobilenet1.0_voc',
                  'ssd_512_resnet50_v1_voc', 'ssd_512_vgg16_atrous_voc']
    ctx = mx.context.current_context()
    x = mx.random.uniform(shape=(1, 3, 512, 544), ctx=ctx)
    _calib_model_list(model_list, ctx, x)

    model_list = ['fcn_resnet101_voc', 'fcn_resnet101_coco',
                  'psp_resnet101_voc', 'psp_resnet101_coco',
                  'deeplab_resnet101_voc', 'deeplab_resnet101_coco']
    ctx = mx.context.current_context()
    x = mx.random.uniform(shape=(1, 3, 480, 480), ctx=ctx)
    _calib_model_list(model_list, ctx, x)

    model_list = ['simple_pose_resnet18_v1b',
                  'simple_pose_resnet50_v1b', 'simple_pose_resnet50_v1d',
                  'simple_pose_resnet101_v1b', 'simple_pose_resnet101_v1d']
    ctx = mx.context.current_context()
    x = mx.random.uniform(shape=(1, 3, 256, 192), ctx=ctx)
    _calib_model_list(model_list, ctx, x)

    model_list = ['vgg16_ucf101', 'resnet18_v1b_kinetics400', 'resnet50_v1b_kinetics400']
    ctx = mx.context.current_context()
    x = mx.random.uniform(shape=(1, 3, 224, 224), ctx=ctx)
    _calib_model_list(model_list, ctx, x)

    model_list = ['inceptionv3_ucf101', 'inceptionv3_kinetics400']
    x = mx.random.uniform(shape=(1, 3, 299, 299), ctx=ctx)
    _calib_model_list(model_list, ctx, x)

    model_list = ['yolo3_darknet53_voc', 'yolo3_darknet53_coco',
                  'yolo3_mobilenet1.0_voc', 'yolo3_mobilenet1.0_coco']
    x = mx.random.uniform(shape=(1, 3, 416, 416), ctx=ctx)
    _calib_model_list(model_list, ctx, x)

@with_cpu(0)
def test_quantized_segmentation_models():
    model_list = ['fcn_resnet101_voc_int8', 'fcn_resnet101_coco_int8',
                  'psp_resnet101_voc_int8', 'psp_resnet101_coco_int8',
                  'deeplab_resnet101_voc_int8', 'deeplab_resnet101_coco_int8']
    ctx = mx.context.current_context()
    x = mx.random.uniform(shape=(1, 3, 480, 480), ctx=ctx)
    _test_model_list(model_list, ctx, x)


@with_cpu(0)
def test_quantized_pose_estimation_models():
    model_list = ['simple_pose_resnet18_v1b_int8',
                  'simple_pose_resnet50_v1b_int8', 'simple_pose_resnet50_v1d_int8',
                  'simple_pose_resnet101_v1b_int8', 'simple_pose_resnet101_v1d_int8']
    ctx = mx.context.current_context()
    x = mx.random.uniform(shape=(1, 3, 256, 192), ctx=ctx)
    _test_model_list(model_list, ctx, x)


if __name__ == '__main__':
    import nose

    nose.runmodule()
