"""MobileNet variants with DUC upsampling layers for CenterNet object detection."""
# pylint: disable=unused-argument
from __future__ import absolute_import

import warnings

from mxnet.context import cpu
from mxnet.gluon import nn
from .. model_zoo import get_model
from ...nn.block import DUC

__all__ = ['DUCMobilenet', 'get_duc_mobilenet',
           'mobilenetv3_large_duc', 'mobilenetv3_small_duc']


class DUCMobilenet(nn.HybridBlock):
    """Mobilenet with DUC upsampling block.

    Parameters
    ----------
    base_network : str
        Name of the base feature extraction network.
    up_filters : list of int
        Number of filters for DUC layers.
    up_scales : list of int
        Upsampling scales for DUC layers.
    pretrained_base : bool
        Whether load pretrained base network.
    norm_layer : mxnet.gluon.nn.HybridBlock
        Type of Norm layers, can be BatchNorm, SyncBatchNorm, GroupNorm, etc.
    norm_kwargs : dict
        Additional kwargs for `norm_layer`.

    """
    def __init__(self, base_network='mobilenetv3_small',
                 up_filters=(512, 256, 128), up_scales=(2, 2, 2),
                 pretrained_base=True, norm_layer=nn.BatchNorm, norm_kwargs=None, **kwargs):
        super(DUCMobilenet, self).__init__(**kwargs)
        if norm_layer != nn.BatchNorm:
            raise NotImplementedError('Only standard BatchNorm layer is supported in DUC module')
        assert 'mobilenet' in base_network
        net = get_model(base_network, pretrained=pretrained_base)
        feat = net.features
        idx = [type(l) for l in feat].index(nn.conv_layers.GlobalAvgPool2D)
        with self.name_scope():
            self.feature = feat[:idx]
            self.upsampling = nn.HybridSequential(prefix='upsampling_')
            for f, s in zip(up_filters, up_scales):
                self.upsampling.add(DUC(f, s))

    def hybrid_forward(self, F, x):
        """HybridForward"""
        y = self.feature(x)
        out = self.upsampling(y)
        return out


def get_duc_mobilenet(base_network, pretrained=False, ctx=cpu(), **kwargs):
    """Get mobilenet with duc upsampling layers.

    Parameters
    ----------
    base_network : str
        Name of the base feature extraction network.
    pretrained : bool
        Whether load pretrained base network.
    ctx : mxnet.Context
        mx.cpu() or mx.gpu()
    Returns
    -------
    nn.HybridBlock
        Network instance of mobilenet with duc upsampling layers

    """
    net = DUCMobilenet(base_network=base_network, pretrained_base=pretrained, **kwargs)
    with warnings.catch_warnings(record=True) as _:
        warnings.simplefilter("always")
        net.initialize()
    net.collect_params().reset_ctx(ctx)
    return net

def mobilenetv3_large_duc(**kwargs):
    """Moiblenetv3 large model with duc layers.

    Returns
    -------
    HybridBlock
        A Moiblenetv3 large model with duc layers

    """
    return get_duc_mobilenet('mobilenetv3_large',
                             up_filters=(512, 256, 256), up_scales=(2, 2, 2), **kwargs)

def mobilenetv3_small_duc(**kwargs):
    """Moiblenetv3 small model with duc layers.

    Returns
    -------
    HybridBlock
        A Moiblenetv3 small model with duc layers

    """
    return get_duc_mobilenet('mobilenetv3_small',
                             up_filters=(512, 256, 128), up_scales=(2, 2, 2), **kwargs)
