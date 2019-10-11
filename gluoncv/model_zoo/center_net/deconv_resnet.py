"""ResNet with Deconvolution layers for CenterNet object detection."""
from __future__ import absolute_import

import os
import warnings
import math

import mxnet as mx
from mxnet.context import cpu
from mxnet import autograd
from mxnet.gluon import nn
from mxnet.gluon import contrib
from .. model_zoo import get_model

__all__ = ['DeconvResnet', 'get_deconv_resnet', 'deconv_resnet18_v1b']


class BilinearUpSample(mx.init.Initializer):
    """Initializes weights as bilinear upsampling kernel.

    Example
    -------
    >>> # Given 'module', an instance of 'mxnet.module.Module', initialize weights to bilinear upsample.
    ...
    >>> init = mx.initializer.BilinearUpSample()
    >>> module.init_params(init)
    >>> for dictionary in module.get_params():
    ...     for key in dictionary:
    ...         print(key)
    ...         print(dictionary[key].asnumpy())
    ...
    fullyconnected0_weight
    [[ 0.  0.  0.]]
    """
    def __init__(self):
        super(BilinearUpSample, self).__init__()

    def _init_weight(self, _, arr):
        f = math.ceil(arr.shape[2] / 2)
        c = (2 * f - 1 - f % 2) / (2. * f)
        for i in range(arr.shape[2]):
            for j in range(arr.shape[3]):
                arr[0, 0, i, j] = \
                    (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
        for c in range(1, arr.shape[0]):
            arr[c, 0, :, :] = arr[0, 0, :, :]

class DeconvResnet(nn.HybridBlock):
    def __init__(self, base_network='resnet18_v1b', use_dcn=False,
                 deconv_filters=(256, 128, 64), deconv_kernels=(4, 4, 4),
                 pretrained_base=True, norm_layer=nn.BatchNorm, norm_kwargs=None, **kwargs):
        super(DeconvResnet, self).__init__(**kwargs)
        assert 'resnet' in base_network
        net = get_model(base_network, pretrained=pretrained_base)
        self._norm_layer = norm_layer
        self._norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
        if 'v1b' in base_network:
            feat = nn.HybridSequential()
            feat.add(*[net.conv1,
                      net.bn1,
                      net.relu,
                      net.maxpool,
                      net.layer1,
                      net.layer2,
                      net.layer3,
                      net.layer4])
            self.base_network = feat
        else:
            raise NotImplementedError()
        with self.name_scope():
            self.deconv = self._make_deconv_layer(deconv_filters, deconv_kernels, use_dcn=use_dcn)

    def _get_deconv_cfg(self, deconv_kernel):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        else:
            raise ValueError('Unsupported deconvolution kernel: {}'.format(deconv_kernel))

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_filters, num_kernels, use_dcn=False):
        assert len(num_kernels) == len(num_filters), \
            'Deconv filters and kernels number mismatch: {} vs. {}'.format(len(num_filters), len(num_kernels))

        layers = nn.HybridSequential('deconv_')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.base_network.initialize()
        in_planes = self.base_network(mx.nd.zeros((1, 3, 256, 256))).shape[1]
        for planes, k in zip(num_filters, num_kernels):
            kernel, padding, output_padding = self._get_deconv_cfg(k)
            layers.add(contrib.cnn.ModulatedDeformableConvolution(planes,
                                                                  kernel_size=3,
                                                                  strides=1,
                                                                  padding=1,
                                                                  dilation=1,
                                                                  num_deformable_group=1,
                                                                  in_channels=in_planes))
            # layers.add(nn.Conv2DTranspose(channels=planes,
            #                               kernel_size=3,
            #                               strides=1,
            #                               padding=1,
            #                               in_channels=in_planes))
            layers.add(self._norm_layer(momentum=0.9, **self._norm_kwargs))
            layers.add(nn.Activation('relu'))
            layers.add(nn.Conv2DTranspose(channels=planes,
                                          kernel_size=kernel,
                                          strides=2,
                                          padding=padding,
                                          output_padding=output_padding,
                                          use_bias=False,
                                          in_channels=planes,
                                          weight_initializer=BilinearUpSample()))
            layers.add(self._norm_layer(momentum=0.9, **self._norm_kwargs))
            layers.add(nn.Activation('relu'))
            in_planes = planes

        return layers

    def hybrid_forward(self, F, x):
        y = self.base_network(x)
        out = self.deconv(y)
        return out

def get_deconv_resnet(base_network, pretrained=False, ctx=cpu(), **kwargs):
    net = DeconvResnet(base_network=base_network, pretrained_base=pretrained, **kwargs)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        net.initialize()
    net.collect_params().reset_ctx(ctx)
    return net

def deconv_resnet18_v1b(**kwargs):
    return get_deconv_resnet('resnet18_v1b', **kwargs)
