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
"""Inception, implemented in Gluon."""
__all__ = ['Inception3', 'inception_v3']

from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm
from mxnet.gluon.contrib.nn import HybridConcurrent

# Helpers
def _make_basic_conv(norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
    out = nn.HybridSequential(prefix='')
    out.add(nn.Conv2D(use_bias=False, **kwargs))
    out.add(norm_layer(epsilon=0.001, **({} if norm_kwargs is None else norm_kwargs)))
    out.add(nn.Activation('relu'))
    return out

def _make_branch(use_pool, norm_layer, norm_kwargs, *conv_settings):
    out = nn.HybridSequential(prefix='')
    if use_pool == 'avg':
        out.add(nn.AvgPool2D(pool_size=3, strides=1, padding=1))
    elif use_pool == 'max':
        out.add(nn.MaxPool2D(pool_size=3, strides=2))
    setting_names = ['channels', 'kernel_size', 'strides', 'padding']
    for setting in conv_settings:
        kwargs = {}
        for i, value in enumerate(setting):
            if value is not None:
                kwargs[setting_names[i]] = value
        out.add(_make_basic_conv(norm_layer, norm_kwargs, **kwargs))
    return out

def _make_A(pool_features, prefix, norm_layer, norm_kwargs):
    out = HybridConcurrent(axis=1, prefix=prefix)
    with out.name_scope():
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (64, 1, None, None)))
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (48, 1, None, None),
                             (64, 5, None, 2)))
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (64, 1, None, None),
                             (96, 3, None, 1),
                             (96, 3, None, 1)))
        out.add(_make_branch('avg', norm_layer, norm_kwargs,
                             (pool_features, 1, None, None)))
    return out

def _make_B(prefix, norm_layer, norm_kwargs):
    out = HybridConcurrent(axis=1, prefix=prefix)
    with out.name_scope():
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (384, 3, 2, None)))
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (64, 1, None, None),
                             (96, 3, None, 1),
                             (96, 3, 2, None)))
        out.add(_make_branch('max', norm_layer, norm_kwargs))
    return out

def _make_C(channels_7x7, prefix, norm_layer, norm_kwargs):
    out = HybridConcurrent(axis=1, prefix=prefix)
    with out.name_scope():
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (192, 1, None, None)))
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (channels_7x7, 1, None, None),
                             (channels_7x7, (1, 7), None, (0, 3)),
                             (192, (7, 1), None, (3, 0))))
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (channels_7x7, 1, None, None),
                             (channels_7x7, (7, 1), None, (3, 0)),
                             (channels_7x7, (1, 7), None, (0, 3)),
                             (channels_7x7, (7, 1), None, (3, 0)),
                             (192, (1, 7), None, (0, 3))))
        out.add(_make_branch('avg', norm_layer, norm_kwargs,
                             (192, 1, None, None)))
    return out

def _make_D(prefix, norm_layer, norm_kwargs):
    out = HybridConcurrent(axis=1, prefix=prefix)
    with out.name_scope():
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (192, 1, None, None),
                             (320, 3, 2, None)))
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (192, 1, None, None),
                             (192, (1, 7), None, (0, 3)),
                             (192, (7, 1), None, (3, 0)),
                             (192, 3, 2, None)))
        out.add(_make_branch('max', norm_layer, norm_kwargs))
    return out

def _make_E(prefix, norm_layer, norm_kwargs):
    out = HybridConcurrent(axis=1, prefix=prefix)
    with out.name_scope():
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (320, 1, None, None)))

        branch_3x3 = nn.HybridSequential(prefix='')
        out.add(branch_3x3)
        branch_3x3.add(_make_branch(None, norm_layer, norm_kwargs,
                                    (384, 1, None, None)))
        branch_3x3_split = HybridConcurrent(axis=1, prefix='')
        branch_3x3_split.add(_make_branch(None, norm_layer, norm_kwargs,
                                          (384, (1, 3), None, (0, 1))))
        branch_3x3_split.add(_make_branch(None, norm_layer, norm_kwargs,
                                          (384, (3, 1), None, (1, 0))))
        branch_3x3.add(branch_3x3_split)

        branch_3x3dbl = nn.HybridSequential(prefix='')
        out.add(branch_3x3dbl)
        branch_3x3dbl.add(_make_branch(None, norm_layer, norm_kwargs,
                                       (448, 1, None, None),
                                       (384, 3, None, 1)))
        branch_3x3dbl_split = HybridConcurrent(axis=1, prefix='')
        branch_3x3dbl.add(branch_3x3dbl_split)
        branch_3x3dbl_split.add(_make_branch(None, norm_layer, norm_kwargs,
                                             (384, (1, 3), None, (0, 1))))
        branch_3x3dbl_split.add(_make_branch(None, norm_layer, norm_kwargs,
                                             (384, (3, 1), None, (1, 0))))

        out.add(_make_branch('avg', norm_layer, norm_kwargs,
                             (192, 1, None, None)))
    return out

def make_aux(classes, norm_layer, norm_kwargs):
    out = nn.HybridSequential(prefix='')
    out.add(nn.AvgPool2D(pool_size=5, strides=3))
    out.add(_make_basic_conv(channels=128, kernel_size=1,
                             norm_layer=norm_layer, norm_kwargs=norm_kwargs))
    out.add(_make_basic_conv(channels=768, kernel_size=5,
                             norm_layer=norm_layer, norm_kwargs=norm_kwargs))
    out.add(nn.Flatten())
    out.add(nn.Dense(classes))
    return out

# Net
class Inception3(HybridBlock):
    r"""Inception v3 model from
    `"Rethinking the Inception Architecture for Computer Vision"
    <http://arxiv.org/abs/1512.00567>`_ paper.

    Parameters
    ----------
    classes : int, default 1000
        Number of classification classes.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    def __init__(self, classes=1000, norm_layer=BatchNorm,
                 norm_kwargs=None, partial_bn=False, **kwargs):
        super(Inception3, self).__init__(**kwargs)
        # self.use_aux_logits = use_aux_logits
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(_make_basic_conv(channels=32, kernel_size=3, strides=2,
                                               norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            if partial_bn:
                if norm_kwargs is not None:
                    norm_kwargs['use_global_stats'] = True
                else:
                    norm_kwargs = {}
                    norm_kwargs['use_global_stats'] = True

            self.features.add(_make_basic_conv(channels=32, kernel_size=3,
                                               norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            self.features.add(_make_basic_conv(channels=64, kernel_size=3, padding=1,
                                               norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            self.features.add(nn.MaxPool2D(pool_size=3, strides=2))
            self.features.add(_make_basic_conv(channels=80, kernel_size=1,
                                               norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            self.features.add(_make_basic_conv(channels=192, kernel_size=3,
                                               norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            self.features.add(nn.MaxPool2D(pool_size=3, strides=2))
            self.features.add(_make_A(32, 'A1_', norm_layer, norm_kwargs))
            self.features.add(_make_A(64, 'A2_', norm_layer, norm_kwargs))
            self.features.add(_make_A(64, 'A3_', norm_layer, norm_kwargs))
            self.features.add(_make_B('B_', norm_layer, norm_kwargs))
            self.features.add(_make_C(128, 'C1_', norm_layer, norm_kwargs))
            self.features.add(_make_C(160, 'C2_', norm_layer, norm_kwargs))
            self.features.add(_make_C(160, 'C3_', norm_layer, norm_kwargs))
            self.features.add(_make_C(192, 'C4_', norm_layer, norm_kwargs))
            self.features.add(_make_D('D_', norm_layer, norm_kwargs))
            self.features.add(_make_E('E1_', norm_layer, norm_kwargs))
            self.features.add(_make_E('E2_', norm_layer, norm_kwargs))
            self.features.add(nn.AvgPool2D(pool_size=8))
            self.features.add(nn.Dropout(0.5))

            self.output = nn.Dense(classes)

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x

# Constructor
def inception_v3(pretrained=False, ctx=cpu(),
                 root='~/.mxnet/models', partial_bn=False, **kwargs):
    r"""Inception v3 model from
    `"Rethinking the Inception Architecture for Computer Vision"
    <http://arxiv.org/abs/1512.00567>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    partial_bn : bool, default False
        Freeze all batch normalization layers during training except the first layer.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """

    net = Inception3(**kwargs)
    if pretrained:
        from .model_store import get_model_file
        net.load_parameters(get_model_file('inceptionv3',
                                           tag=pretrained, root=root), ctx=ctx)
        from ..data import ImageNet1kAttr
        attrib = ImageNet1kAttr()
        net.synset = attrib.synset
        net.classes = attrib.classes
        net.classes_long = attrib.classes_long
    return net
