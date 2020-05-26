"""ResNet with Deconvolution layers for CenterNet object detection."""
# pylint: disable=unused-argument
from __future__ import absolute_import

import warnings
import math

import mxnet as mx
from mxnet.context import cpu
from mxnet.gluon import nn
from mxnet.gluon import contrib
from .. model_zoo import get_model

__all__ = ['DeconvResnet', 'get_deconv_resnet',
           'resnet18_v1b_deconv', 'resnet18_v1b_deconv_dcnv2',
           'resnet50_v1b_deconv', 'resnet50_v1b_deconv_dcnv2',
           'resnet101_v1b_deconv', 'resnet101_v1b_deconv_dcnv2']


class BilinearUpSample(mx.init.Initializer):
    """Initializes weights as bilinear upsampling kernel.

    Example
    -------
    >>> # Given 'module', an instance of 'mxnet.module.Module',
    initialize weights to bilinear upsample...
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
        mx.nd.random.normal(0, 0.01, arr.shape, out=arr)
        f = math.ceil(arr.shape[2] / 2)
        c = (2 * f - 1 - f % 2) / (2. * f)
        for i in range(arr.shape[2]):
            for j in range(arr.shape[3]):
                arr[0, 0, i, j] = \
                    (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
        for c in range(1, arr.shape[0]):
            arr[c, 0, :, :] = arr[0, 0, :, :]

class DeconvResnet(nn.HybridBlock):
    """Deconvolutional ResNet.

    Parameters
    ----------
    base_network : str
        Name of the base feature extraction network.
    deconv_filters : list of int
        Number of filters for deconv layers.
    deconv_kernels : list of int
        Kernel sizes for deconv layers.
    pretrained_base : bool
        Whether load pretrained base network.
    norm_layer : mxnet.gluon.nn.HybridBlock
        Type of Norm layers, can be BatchNorm, SyncBatchNorm, GroupNorm, etc.
    norm_kwargs : dict
        Additional kwargs for `norm_layer`.
    use_dcnv2 : bool
        If true, will use DCNv2 layers in upsampling blocks

    """
    def __init__(self, base_network='resnet18_v1b',
                 deconv_filters=(256, 128, 64), deconv_kernels=(4, 4, 4),
                 pretrained_base=True, norm_layer=nn.BatchNorm, norm_kwargs=None,
                 use_dcnv2=False, **kwargs):
        super(DeconvResnet, self).__init__(**kwargs)
        assert 'resnet' in base_network
        net = get_model(base_network, pretrained=pretrained_base)
        self._norm_layer = norm_layer
        self._norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
        self._use_dcnv2 = use_dcnv2
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
            raise NotImplementedError('Only v1 variants of resnet are supported so far.')
        with self.name_scope():
            self.deconv = self._make_deconv_layer(deconv_filters, deconv_kernels)

    def _get_deconv_cfg(self, deconv_kernel):
        """Get the deconv configs using presets"""
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

    def _make_deconv_layer(self, num_filters, num_kernels):
        # pylint: disable=unused-variable
        """Make deconv layers using the configs"""
        assert len(num_kernels) == len(num_filters), \
            'Deconv filters and kernels number mismatch: {} vs. {}'.format(
                len(num_filters), len(num_kernels))

        layers = nn.HybridSequential('deconv_')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.base_network.initialize()
        in_planes = self.base_network(mx.nd.zeros((1, 3, 256, 256))).shape[1]
        for planes, k in zip(num_filters, num_kernels):
            kernel, padding, output_padding = self._get_deconv_cfg(k)
            if self._use_dcnv2:
                assert hasattr(contrib.cnn, 'ModulatedDeformableConvolution'), \
                    "No ModulatedDeformableConvolution found in mxnet, consider upgrade..."
                layers.add(contrib.cnn.ModulatedDeformableConvolution(planes,
                                                                      kernel_size=3,
                                                                      strides=1,
                                                                      padding=1,
                                                                      dilation=1,
                                                                      num_deformable_group=1,
                                                                      in_channels=in_planes))
            else:
                layers.add(nn.Conv2D(channels=planes,
                                     kernel_size=3,
                                     strides=1,
                                     padding=1,
                                     in_channels=in_planes))
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
        # pylint: disable=arguments-differ
        """HybridForward"""
        y = self.base_network(x)
        out = self.deconv(y)
        return out

def get_deconv_resnet(base_network, pretrained=False, ctx=cpu(), use_dcnv2=False, **kwargs):
    """Get resnet with deconv layers.

    Parameters
    ----------
    base_network : str
        Name of the base feature extraction network.
    pretrained : bool
        Whether load pretrained base network.
    ctx : mxnet.Context
        mx.cpu() or mx.gpu()
    use_dcnv2 : bool
        If true, will use DCNv2 layers in upsampling blocks
    pretrained : type
        Description of parameter `pretrained`.
    Returns
    -------
    nn.HybridBlock
        Network instance of resnet with deconv layers

    """
    net = DeconvResnet(base_network=base_network, pretrained_base=pretrained,
                       use_dcnv2=use_dcnv2, **kwargs)
    with warnings.catch_warnings(record=True) as _:
        warnings.simplefilter("always")
        net.initialize()
    net.collect_params().reset_ctx(ctx)
    return net

def resnet18_v1b_deconv(**kwargs):
    """Resnet18 v1b model with deconv layers.

    Returns
    -------
    HybridBlock
        A Resnet18 v1b model with deconv layers.

    """
    kwargs['use_dcnv2'] = False
    return get_deconv_resnet('resnet18_v1b', **kwargs)

def resnet18_v1b_deconv_dcnv2(**kwargs):
    """Resnet18 v1b model with deconv layers and deformable v2 conv layers.

    Returns
    -------
    HybridBlock
        A Resnet18 v1b model with deconv layers and deformable v2 conv layers.

    """
    kwargs['use_dcnv2'] = True
    return get_deconv_resnet('resnet18_v1b', **kwargs)

def resnet50_v1b_deconv(**kwargs):
    """Resnet50 v1b model with deconv layers.

    Returns
    -------
    HybridBlock
        A Resnet50 v1b model with deconv layers.

    """
    kwargs['use_dcnv2'] = False
    return get_deconv_resnet('resnet50_v1b', **kwargs)

def resnet50_v1b_deconv_dcnv2(**kwargs):
    """Resnet50 v1b model with deconv layers and deformable v2 conv layers.

    Returns
    -------
    HybridBlock
        A Resnet50 v1b model with deconv layers and deformable v2 conv layers.

    """
    kwargs['use_dcnv2'] = True
    return get_deconv_resnet('resnet50_v1b', **kwargs)

def resnet101_v1b_deconv(**kwargs):
    """Resnet101 v1b model with deconv layers.

    Returns
    -------
    HybridBlock
        A Resnet101 v1b model with deconv layers.

    """
    kwargs['use_dcnv2'] = False
    return get_deconv_resnet('resnet101_v1b', **kwargs)

def resnet101_v1b_deconv_dcnv2(**kwargs):
    """Resnet101 v1b model with deconv layers and deformable v2 conv layers.

    Returns
    -------
    HybridBlock
        A Resnet101 v1b model with deconv layers and deformable v2 conv layers.

    """
    kwargs['use_dcnv2'] = True
    return get_deconv_resnet('resnet101_v1b', **kwargs)
