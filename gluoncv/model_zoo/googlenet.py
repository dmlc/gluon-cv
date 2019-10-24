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
# pylint: disable=missing-docstring,arguments-differ,unused-argument
"""GoogleNet, implemented in Gluon."""

__all__ = ['GoogLeNet', 'googlenet']

from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm
from mxnet.gluon.contrib.nn import HybridConcurrent

def _make_basic_conv(in_channels, channels, norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
    out = nn.HybridSequential(prefix='')
    out.add(nn.Conv2D(in_channels=in_channels, channels=channels, use_bias=False, **kwargs))
    out.add(norm_layer(in_channels=channels, epsilon=0.001,
                       **({} if norm_kwargs is None else norm_kwargs)))
    out.add(nn.Activation('relu'))
    return out

def _make_branch(use_pool, norm_layer, norm_kwargs, *conv_settings):
    out = nn.HybridSequential(prefix='')
    if use_pool == 'avg':
        out.add(nn.AvgPool2D(pool_size=3, strides=1, padding=1))
    elif use_pool == 'max':
        out.add(nn.MaxPool2D(pool_size=3, strides=1, padding=1))
    setting_names = ['in_channels', 'channels', 'kernel_size', 'strides', 'padding']
    for setting in conv_settings:
        kwargs = {}
        for i, value in enumerate(setting):
            if value is not None:
                if setting_names[i] == 'in_channels':
                    in_channels = value
                elif setting_names[i] == 'channels':
                    channels = value
                else:
                    kwargs[setting_names[i]] = value
        out.add(_make_basic_conv(in_channels, channels, norm_layer, norm_kwargs, **kwargs))
    return out

def _make_Mixed_3a(in_channels, pool_features, prefix, norm_layer, norm_kwargs):
    out = HybridConcurrent(axis=1, prefix=prefix)
    with out.name_scope():
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 64, 1, None, None)))
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 96, 1, None, None),
                             (96, 128, 3, None, 1)))
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 16, 1, None, None),
                             (16, 32, 3, None, 1)))
        out.add(_make_branch('max', norm_layer, norm_kwargs,
                             (in_channels, pool_features, 1, None, None)))
    return out

def _make_Mixed_3b(in_channels, pool_features, prefix, norm_layer, norm_kwargs):
    out = HybridConcurrent(axis=1, prefix=prefix)
    with out.name_scope():
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 128, 1, None, None)))
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 128, 1, None, None),
                             (128, 192, 3, None, 1)))
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 32, 1, None, None),
                             (32, 96, 3, None, 1)))
        out.add(_make_branch('max', norm_layer, norm_kwargs,
                             (in_channels, pool_features, 1, None, None)))
    return out

def _make_Mixed_4a(in_channels, pool_features, prefix, norm_layer, norm_kwargs):
    out = HybridConcurrent(axis=1, prefix=prefix)
    with out.name_scope():
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 192, 1, None, None)))
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 96, 1, None, None),
                             (96, 208, 3, None, 1)))
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 16, 1, None, None),
                             (16, 48, 3, None, 1)))
        out.add(_make_branch('max', norm_layer, norm_kwargs,
                             (in_channels, pool_features, 1, None, None)))
    return out

def _make_Mixed_4b(in_channels, pool_features, prefix, norm_layer, norm_kwargs):
    out = HybridConcurrent(axis=1, prefix=prefix)
    with out.name_scope():
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 160, 1, None, None)))
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 112, 1, None, None),
                             (112, 224, 3, None, 1)))
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 24, 1, None, None),
                             (24, 64, 3, None, 1)))
        out.add(_make_branch('max', norm_layer, norm_kwargs,
                             (in_channels, pool_features, 1, None, None)))
    return out

def _make_Mixed_4c(in_channels, pool_features, prefix, norm_layer, norm_kwargs):
    out = HybridConcurrent(axis=1, prefix=prefix)
    with out.name_scope():
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 128, 1, None, None)))
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 128, 1, None, None),
                             (128, 256, 3, None, 1)))
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 24, 1, None, None),
                             (24, 64, 3, None, 1)))
        out.add(_make_branch('max', norm_layer, norm_kwargs,
                             (in_channels, pool_features, 1, None, None)))
    return out

def _make_Mixed_4d(in_channels, pool_features, prefix, norm_layer, norm_kwargs):
    out = HybridConcurrent(axis=1, prefix=prefix)
    with out.name_scope():
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 112, 1, None, None)))
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 144, 1, None, None),
                             (144, 288, 3, None, 1)))
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 32, 1, None, None),
                             (32, 64, 3, None, 1)))
        out.add(_make_branch('max', norm_layer, norm_kwargs,
                             (in_channels, pool_features, 1, None, None)))
    return out

def _make_Mixed_4e(in_channels, pool_features, prefix, norm_layer, norm_kwargs):
    out = HybridConcurrent(axis=1, prefix=prefix)
    with out.name_scope():
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 256, 1, None, None)))
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 160, 1, None, None),
                             (160, 320, 3, None, 1)))
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 32, 1, None, None),
                             (32, 128, 3, None, 1)))
        out.add(_make_branch('max', norm_layer, norm_kwargs,
                             (in_channels, pool_features, 1, None, None)))
    return out

def _make_Mixed_5a(in_channels, pool_features, prefix, norm_layer, norm_kwargs):
    out = HybridConcurrent(axis=1, prefix=prefix)
    with out.name_scope():
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 256, 1, None, None)))
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 160, 1, None, None),
                             (160, 320, 3, None, 1)))
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 32, 1, None, None),
                             (32, 128, 3, None, 1)))
        out.add(_make_branch('max', norm_layer, norm_kwargs,
                             (in_channels, pool_features, 1, None, None)))
    return out

def _make_Mixed_5b(in_channels, pool_features, prefix, norm_layer, norm_kwargs):
    out = HybridConcurrent(axis=1, prefix=prefix)
    with out.name_scope():
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 384, 1, None, None)))
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 192, 1, None, None),
                             (192, 384, 3, None, 1)))
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 48, 1, None, None),
                             (48, 128, 3, None, 1)))
        out.add(_make_branch('max', norm_layer, norm_kwargs,
                             (in_channels, pool_features, 1, None, None)))
    return out

def _make_aux(in_channels, classes, norm_layer, norm_kwargs):
    out = nn.HybridSequential(prefix='')
    out.add(nn.AvgPool2D(pool_size=5, strides=3))
    out.add(_make_basic_conv(in_channels=in_channels, channels=128, kernel_size=1,
                             norm_layer=norm_layer, norm_kwargs=norm_kwargs))

    out.add(nn.Flatten())
    out.add(nn.Dense(units=1024, in_units=2048))
    out.add(nn.Activation('relu'))
    out.add(nn.Dropout(0.7))
    out.add(nn.Dense(units=classes, in_units=1024))
    return out

class GoogLeNet(HybridBlock):
    r"""GoogleNet model from
    `"Going Deeper with Convolutions"
    <https://arxiv.org/abs/1409.4842>`_ paper.
    `"Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"
    <https://arxiv.org/abs/1502.03167>`_ paper.

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
    partial_bn : bool, default False
        Freeze all batch normalization layers during training except the first layer.
    """
    def __init__(self, classes=1000, norm_layer=BatchNorm, dropout_ratio=0.4, aux_logits=False,
                 norm_kwargs=None, partial_bn=False, pretrained_base=True, ctx=None, **kwargs):
        super(GoogLeNet, self).__init__(**kwargs)
        self.dropout_ratio = dropout_ratio
        self.aux_logits = aux_logits

        with self.name_scope():
            self.conv1 = _make_basic_conv(in_channels=3, channels=64, kernel_size=7,
                                          strides=2, padding=3,
                                          norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.maxpool1 = nn.MaxPool2D(pool_size=3, strides=2, ceil_mode=True)

            if partial_bn:
                if norm_kwargs is not None:
                    norm_kwargs['use_global_stats'] = True
                else:
                    norm_kwargs = {}
                    norm_kwargs['use_global_stats'] = True

            self.conv2 = _make_basic_conv(in_channels=64, channels=64, kernel_size=1,
                                          norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.conv3 = _make_basic_conv(in_channels=64, channels=192,
                                          kernel_size=3, padding=1,
                                          norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.maxpool2 = nn.MaxPool2D(pool_size=3, strides=2, ceil_mode=True)

            self.inception3a = _make_Mixed_3a(192, 32, 'Mixed_3a_', norm_layer, norm_kwargs)
            self.inception3b = _make_Mixed_3b(256, 64, 'Mixed_3b_', norm_layer, norm_kwargs)
            self.maxpool3 = nn.MaxPool2D(pool_size=3, strides=2, ceil_mode=True)

            self.inception4a = _make_Mixed_4a(480, 64, 'Mixed_4a_', norm_layer, norm_kwargs)
            self.inception4b = _make_Mixed_4b(512, 64, 'Mixed_4b_', norm_layer, norm_kwargs)
            self.inception4c = _make_Mixed_4c(512, 64, 'Mixed_4c_', norm_layer, norm_kwargs)
            self.inception4d = _make_Mixed_4d(512, 64, 'Mixed_4d_', norm_layer, norm_kwargs)
            self.inception4e = _make_Mixed_4e(528, 128, 'Mixed_4e_', norm_layer, norm_kwargs)
            self.maxpool4 = nn.MaxPool2D(pool_size=2, strides=2)

            self.inception5a = _make_Mixed_5a(832, 128, 'Mixed_5a_', norm_layer, norm_kwargs)
            self.inception5b = _make_Mixed_5b(832, 128, 'Mixed_5b_', norm_layer, norm_kwargs)

            if self.aux_logits:
                self.aux1 = _make_aux(512, classes, norm_layer, norm_kwargs)
                self.aux2 = _make_aux(528, classes, norm_layer, norm_kwargs)

            self.head = nn.HybridSequential(prefix='')
            self.avgpool = nn.AvgPool2D(pool_size=7)
            self.dropout = nn.Dropout(self.dropout_ratio)
            self.output = nn.Dense(units=classes, in_units=1024)
            self.head.add(self.avgpool)
            self.head.add(self.dropout)
            self.head.add(self.output)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        if self.aux_logits:
            aux1 = self.aux1(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        if self.aux_logits:
            aux2 = self.aux2(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.head(x)

        if self.aux_logits:
            return (x, aux2, aux1)
        return x

def googlenet(classes=1000, pretrained=False, pretrained_base=True, ctx=cpu(),
              dropout_ratio=0.4, aux_logits=False,
              root='~/.mxnet/models', partial_bn=False, **kwargs):
    r"""GoogleNet model from
    `"Going Deeper with Convolutions"
    <https://arxiv.org/abs/1409.4842>`_ paper.
    `"Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"
    <https://arxiv.org/abs/1502.03167>`_ paper.

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

    net = GoogLeNet(classes=classes, partial_bn=partial_bn, pretrained_base=pretrained_base,
                    dropout_ratio=dropout_ratio, aux_logits=aux_logits, ctx=ctx, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        net.load_parameters(get_model_file('googlenet',
                                           tag=pretrained, root=root), ctx=ctx, cast_dtype=True)
        from ..data import ImageNet1kAttr
        attrib = ImageNet1kAttr()
        net.synset = attrib.synset
        net.classes = attrib.classes
        net.classes_long = attrib.classes_long
    return net
