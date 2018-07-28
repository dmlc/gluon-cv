"""Darknet as YOLO backbone network."""
# pylint: disable=arguments-differ
from __future__ import absolute_import

import os
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn

__all__ = ['DarknetV3', 'get_darknet', 'darknet53']

def _conv2d(channel, kernel, padding, stride, num_sync_bn_devices=-1):
    """A common conv-bn-leakyrelu cell"""
    cell = nn.HybridSequential(prefix='')
    cell.add(nn.Conv2D(channel, kernel_size=kernel,
                       strides=stride, padding=padding, use_bias=False))
    if num_sync_bn_devices < 1:
        cell.add(nn.BatchNorm(epsilon=1e-5, momentum=0.9))
    else:
        cell.add(gluon.contrib.nn.SyncBatchNorm(
            epsilon=1e-5, momentum=0.9, num_devices=num_sync_bn_devices))
    cell.add(nn.LeakyReLU(0.1))
    return cell


class DarknetBasicBlockV3(gluon.HybridBlock):
    """Darknet Basic Block. Which is a 1x1 reduce conv followed by 3x3 conv.

    Parameters
    ----------
    channel : int
        Convolution channels for 1x1 conv.
    num_sync_bn_devices : int, default is -1
        Number of devices for training. If `num_sync_bn_devices < 2`, SyncBatchNorm is disabled.

    """
    def __init__(self, channel, num_sync_bn_devices=-1, **kwargs):
        super(DarknetBasicBlockV3, self).__init__(**kwargs)
        self.body = nn.HybridSequential(prefix='')
        # 1x1 reduce
        self.body.add(_conv2d(channel, 1, 0, 1, num_sync_bn_devices))
        # 3x3 conv expand
        self.body.add(_conv2d(channel * 2, 3, 1, 1, num_sync_bn_devices))

    # pylint: disable=unused-argument
    def hybrid_forward(self, F, x, *args):
        residual = x
        x = self.body(x)
        return x + residual


class DarknetV3(gluon.HybridBlock):
    """Darknet v3.

    Parameters
    ----------
    layers : iterable
        Description of parameter `layers`.
    channels : iterable
        Description of parameter `channels`.
    classes : int, default is 1000
        Number of classes, which determines the dense layer output channels.
    num_sync_bn_devices : int, default is -1
        Number of devices for training. If `num_sync_bn_devices < 2`, SyncBatchNorm is disabled.

    Attributes
    ----------
    features : mxnet.gluon.nn.HybridSequential
        Feature extraction layers.
    output : mxnet.gluon.nn.Dense
        A classes(1000)-way Fully-Connected Layer.

    """
    def __init__(self, layers, channels, classes=1000, num_sync_bn_devices=-1, **kwargs):
        super(DarknetV3, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1, (
            "len(channels) should equal to len(layers) + 1, given {} vs {}".format(
                len(channels), len(layers)))
        with self.name_scope():
            self.features = nn.HybridSequential()
            # first 3x3 conv
            self.features.add(_conv2d(channels[0], 3, 1, 1, num_sync_bn_devices))
            for nlayer, channel in zip(layers, channels[1:]):
                assert channel % 2 == 0, "channel {} cannot be divided by 2".format(channel)
                # add downsample conv with stride=2
                self.features.add(_conv2d(channel, 3, 1, 2, num_sync_bn_devices))
                # add nlayer basic blocks
                for _ in range(nlayer):
                    self.features.add(DarknetBasicBlockV3(channel // 2, num_sync_bn_devices))
            # output
            self.output = nn.Dense(classes)

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = F.Pooling(x, kernel=(7, 7), global_pool=True, pool_type='avg')
        return self.output(x)

# default configurations
darknet_versions = {'v3': DarknetV3}
darknet_spec = {
    'v3': {53: ([1, 2, 8, 8, 4], [32, 64, 128, 256, 512, 1024]),}
}

def get_darknet(darknet_version, num_layers, pretrained=False, ctx=mx.cpu(),
                root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    """Get darknet by `version` and `num_layers` info.

    Parameters
    ----------
    darknet_version : str
        Darknet version, choices are ['v3'].
    num_layers : int
        Number of layers.
    pretrained : boolean
        Whether fetch and load pre-trained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Returns
    -------
    mxnet.gluon.HybridBlock
        Darknet network.

    Examples
    --------
    >>> model = get_darknet('v3', 53, pretrained=True)
    >>> print(model)

    """
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
            'darknet%d'%(num_layers), root=root), ctx=ctx)
    return net

def darknet53(**kwargs):
    """Darknet v3 53 layer network.
    Reference: https://arxiv.org/pdf/1804.02767.pdf.

    Returns
    -------
    mxnet.gluon.HybridBlock
        Darknet network.

    """
    return get_darknet('v3', 53, **kwargs)
