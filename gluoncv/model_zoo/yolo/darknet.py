"""Darknet as YOLO backbone network."""
from __future__ import absolute_import

import os
import mxnet as mx
import numpy as np
from mxnet import gluon
from mxnet.gluon import nn

__all__ = ['DarknetV3', 'get_darknet', 'darknet53']

def _conv2d(channel, kernel, padding, stride):
    """A common conv-bn-leakyrelu cell"""
    cell = nn.HybridSequential(prefix='')
    cell.add(nn.Conv2D(channel, kernel_size=kernel,
                       strides=stride, padding=padding, use_bias=False))
    cell.add(nn.BatchNorm(epsilon=1e-5, momentum=0.9))
    cell.add(nn.LeakyReLU(0.1))
    return cell


class DarknetBasicBlockV3(gluon.HybridBlock):
    def __init__(self, channel, **kwargs):
        super(DarknetBasicBlockV3, self).__init__(**kwargs)
        self.body = nn.HybridSequential(prefix='')
        # 1x1 reduce
        self.body.add(_conv2d(channel, 1, 0, 1))
        # 3x3 conv expand
        self.body.add(_conv2d(channel * 2, 3, 1, 1))

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
            self.features.add(_conv2d(channels[0], 3, 1, 1))
            for nlayer, channel in zip(layers, channels[1:]):
                assert channel % 2 == 0, "channel {} cannot be divided by 2".format(channel)
                # add downsample conv with stride=2
                self.features.add(_conv2d(channel, 3, 1, 2))
                # add nlayer basic blocks
                for _ in range(nlayer):
                    self.features.add(DarknetBasicBlockV3(channel // 2))
            # output
            self.output = nn.Dense(classes)

    def hybrid_forward(self, F, x):
      x = self.features(x)
      x = F.Pooling(x, kernel=(7, 7), global_pool=True, pool_type='avg')
      return self.output(x)

darknet_versions = {'v3': DarknetV3}
darknet_spec = {
    'v3': {53: ([1, 2, 8, 8, 4], [32, 64, 128, 256, 512, 1024]),}
}

def get_darknet(darknet_version, num_layers, pretrained=False, ctx=mx.cpu(),
                root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    assert darknet_version in darknet_versions and darknet_version in darknet_spec, (
        "Invalid darknet version: {}. Options are {}".format(
            darknet_version, str(darknet_versions.keys())))
    specs = darknet_spec[darknet_version]
    assert num_layers in specs, (
        "Invalid number of layers: {}. Options are {}".format(num_layers, str(specs.keys())))
    layers, channels = specs[num_layers]
    darknet_class = darknet_versions[darknet_version]
    net = darknet_class(layers, channels, **kwargs)
    if pretrained:
        from ..model_store import get_model_file
        net.load_parameters(get_model_file(
            'darknet%d_%s'%(num_layers, darknet_version), root=root), ctx=ctx)
    return net

def darknet53(**kwargs):
    return get_darknet('v3', 53, **kwargs)
