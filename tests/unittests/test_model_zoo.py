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

def test_classification_models():
    x = mx.random.uniform(shape=(2, 3, 32, 32))
    cifar_models = [
        'cifar_resnet20_v1', 'cifar_resnet56_v1', 'cifar_resnet110_v1',
        'cifar_resnet20_v2', 'cifar_resnet56_v2', 'cifar_resnet110_v2',
        'cifar_wideresnet16_10', 'cifar_wideresnet28_10', 'cifar_wideresnet40_8',
    ]
    for model in cifar_models:
        net = gcv.model_zoo.get_model(model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            net.initialize()
        net(x)

def test_ssd_models():
    x = mx.random.uniform(shape=(2, 3, 512, 768))  # allow non-squre and larger inputs
    models = ['ssd_300_vgg16_atrous_voc', 'ssd_512_vgg16_atrous_voc', 'ssd_512_resnet50_v1_voc']
    for model in models:
        net = gcv.model_zoo.get_model(model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            net.initialize()
        net(x)

def test_resnet_v1b():
    x = mx.random.uniform(shape=(2, 3, 224, 224))
    models = ['resnet18_v1b', 'resnet34_v1b', 'resnet50_v1b',
              'resnet101_v1b', 'resnet152_v1b']
    for model in models:
        net = gcv.model_zoo.get_model(model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            net.initialize()
        net(x)

def test_segmentation_models():
    x = mx.random.uniform(shape=(2, 3, 480, 480))
    models = ['fcn_resnet50_voc', 'fcn_resnet101_voc']
    for model in models:
        net = gcv.model_zoo.get_model(model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            net.initialize()
        net(x)

if __name__ == '__main__':
    import nose
    nose.runmodule()
