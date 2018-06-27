"""Darknet as YOLO backbone network."""
from __future__ import absolute_import

import os
import mxnet as mx
import numpy as np
from mxnet import gluon
from mxnet.gluon import nn

__all__ = ['DarknetV3', 'get_darknet', 'darknet53']


class DarknetBasicBlockV3(gluon.HybridBlock):
    def __init__(self, channel, **kwargs):
        super(DarknetBasicBlockV3, self).__init__(**kwargs)
        self.body = nn.HybridSequential(prefix='')
        # 1x1 reduce
        self.body.add(nn.Conv2D(channel, kernel_size=1, padding=0, use_bias=False))
        self.body.add(nn.BatchNorm(epsilon=1e-5, momentum=0.9))
        self.body.add(nn.LeakyReLU(0.1))
        # 3x3 conv expand
        self.body.add(nn.Conv2D(channel * 2, kernel_size=3, padding=1, use_bias=False))
        self.body.add(nn.BatchNorm(epsilon=1e-5, momentum=0.9))
        self.body.add(nn.LeakyReLU(0.1))

    def hybrid_forward(self, F, x):
        residual = x
        x = self.body(x)
        return x + residual


class DarknetV3(gluon.HybridBlock):
    def __init__(self, layers, channels, classes=1000, **kwargs):
        super(DarknetV3, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1, (
            "len(channels) should equal to len(layers) + 1, given {} vs {}".format(
            len(channels), len(layers)))
        with self.name_scope():
            self.features = nn.HybridSequential()
            # first 3x3 conv
            self.features.add(nn.Conv2D(channels[0], kernel_size=3, padding=1, use_bias=False))
            self.features.add(nn.BatchNorm(epsilon=1e-5, momentum=0.9))
            self.features.add(nn.LeakyReLU(0.1))
            for nlayer, channel in zip(layers, channels[1:]):
                assert channel % 2 == 0, "channel {} cannot be divided by 2".format(channel)
                # add downsample conv with stride=2
                self.features.add(nn.Conv2D(
                    channel, kernel_size=3, strides=2, padding=1, use_bias=False))
                self.features.add(nn.BatchNorm(epsilon=1e-5, momentum=0.9))
                self.features.add(nn.LeakyReLU(0.1))
                # add nlayer basic blocks
                for _ in range(nlayer):
                    self.features.add(DarknetBasicBlockV3(channel // 2))
            # output
            self.output = nn.Dense(classes)

    def hybrid_forward(self, F, x):
      x = self.features(x)
      x = F.Pooling(x, kernel=(7, 7), global_pool=True, pool_type='avg')
      return self.output(x)

darknet_spec = {53: ([1, 2, 8, 8, 4], [32, 64, 128, 256, 512, 1024]),}

def get_darknet(num_layers, pretrained=False, ctx=mx.cpu(),
                root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    assert num_layers in darknet_spec, (
        "Invalid number of layers: {}. Options are {}".format(num_layers, str(darknet_spec.keys())))
    layers, channels = darknet_spec[num_layers]
    net = DarknetV3(layers, channels, **kwargs)
    if pretrained:
        from ..model_store import get_model_file
        net.load_parameters(get_model_file('darknet%d'%(num_layers), root=root), ctx=ctx)
    return net

def darknet53(**kwargs):
    return get_darknet(53, **kwargs)
