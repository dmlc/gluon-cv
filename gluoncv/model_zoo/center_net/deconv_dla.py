"""DLA network with Deconvolution layers for CenterNet object detection."""
# pylint: disable=arguments-differ,unused-argument
from __future__ import absolute_import

import warnings

import numpy as np
from mxnet.context import cpu
from mxnet.gluon import nn
from mxnet.gluon import contrib
from . deconv_resnet import BilinearUpSample
from .. model_zoo import get_model

__all__ = ['get_deconv_dla', 'dla34_deconv']

class CustomConv(nn.HybridBlock):
    """Custom Conv Block.

    Parameters
    ----------
    in_channels : int
        Input channels of conv layer.
    out_channels : int
        Output channels of conv layer.
    use_dcnv2 : bool
        Whether use modulated deformable convolution(DCN version2).
    momentum : float, default is 0.9
        Momentum for Norm layer.
    norm_layer : nn.HybridBlock, default is BatchNorm.
        The norm_layer instance.
    norm_kwargs : dict
        Extra arguments for norm layer.

    """
    def __init__(self, in_channels, out_channels,
                 use_dcnv2=False, momentum=0.9,
                 norm_layer=nn.BatchNorm, norm_kwargs=None, **kwargs):
        super(CustomConv, self).__init__(**kwargs)
        if norm_kwargs is None:
            norm_kwargs = {}
        with self.name_scope():
            self.actf = nn.HybridSequential()
            self.actf.add(*[
                norm_layer(in_channels=out_channels, momentum=momentum, **norm_kwargs),
                nn.Activation('relu')])
            if use_dcnv2:
                assert hasattr(contrib.cnn, 'ModulatedDeformableConvolution'), \
                    "No ModulatedDeformableConvolution found in mxnet, consider upgrade..."
                self.conv = contrib.cnn.ModulatedDeformableConvolution(out_channels,
                                                                       kernel_size=3,
                                                                       strides=1,
                                                                       padding=1,
                                                                       dilation=1,
                                                                       num_deformable_group=1,
                                                                       in_channels=in_channels)
            else:
                self.conv = nn.Conv2D(out_channels,
                                      kernel_size=3,
                                      strides=1,
                                      padding=1,
                                      dilation=1,
                                      in_channels=in_channels)

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.actf(x)
        return x


class IDAUp(nn.HybridBlock):
    """Iterative deep aggregation layer.

    Parameters
    ----------
    out_channels : iterable of int
        Output channels for multiple layers.
    in_channels : iterable of int
        Input channels for multiple layers.
    up_f : iterable of float
        Upsampling ratios.
    use_dcnv2 : bool
        Whether use modulated deformable convolution(DCN version2).
    norm_layer : nn.HybridBlock, default is BatchNorm.
        The norm_layer instance.
    norm_kwargs : dict
        Extra arguments for norm layer.

    """
    def __init__(self, out_channels, in_channels, up_f,
                 use_dcnv2=False, norm_layer=nn.BatchNorm, norm_kwargs=None, **kwargs):
        super(IDAUp, self).__init__(**kwargs)
        self.startp = 0
        self.endp = 1
        with self.name_scope():
            self.projs = nn.HybridSequential('ida_proj')
            self.ups = nn.HybridSequential('ida_ups')
            self.nodes = nn.HybridSequential('ida_nodes')
            for i in range(1, len(in_channels)):
                c = in_channels[i]
                f = int(up_f[i])
                proj = CustomConv(c, out_channels, use_dcnv2=use_dcnv2,
                                  norm_layer=norm_layer, norm_kwargs=norm_kwargs)
                node = CustomConv(out_channels, out_channels, use_dcnv2=use_dcnv2,
                                  norm_layer=norm_layer, norm_kwargs=norm_kwargs)

                up = nn.Conv2DTranspose(in_channels=out_channels, channels=out_channels,
                                        kernel_size=f * 2, strides=f,
                                        padding=f // 2, output_padding=0,
                                        groups=out_channels, use_bias=False,
                                        weight_initializer=BilinearUpSample())

                self.projs.add(proj)
                self.ups.add(up)
                self.nodes.add(node)


    def hybrid_forward(self, F, layers):
        for i in range(self.startp + 1, self.endp):
            upsample = self.ups[i - self.startp - 1]
            project = self.projs[i - self.startp - 1]
            layers[i] = upsample(project(layers[i]))
            node = self.nodes[i - self.startp - 1]
            layers[i] = node(layers[i] + layers[i - 1])
        return layers

class DLAUp(nn.HybridBlock):
    """Deep layer aggregation upsampling layer.

    Parameters
    ----------
    startp : int
        Start index.
    channels : iterable of int
        Output channels.
    scales : iterable of int
        Upsampling scales.
    in_channels : iterable of int
        Input channels.
    use_dcnv2 : bool
        Whether use ModulatedDeformableConvolution(DCN version 2).
    norm_layer : nn.HybridBlock, default is BatchNorm
        The norm layer type.
    norm_kwargs : dict
        Extra arguments to norm layer.

    """
    def __init__(self, startp, channels, scales, in_channels=None,
                 use_dcnv2=False, norm_layer=nn.BatchNorm, norm_kwargs=None, **kwargs):
        super(DLAUp, self).__init__(**kwargs)
        self.startp = startp
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        with self.name_scope():
            self.idas = nn.HybridSequential('ida')
            for i in range(len(channels) - 1):
                j = -i - 2
                self.idas.add(IDAUp(channels[j], in_channels[j:],
                                    scales[j:] // scales[j], use_dcnv2=use_dcnv2,
                                    norm_layer=norm_layer, norm_kwargs=norm_kwargs))
                scales[j + 1:] = scales[j]
                in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def hybrid_forward(self, F, layers):
        out = [layers[-1]] # start with 32
        for i in range(len(layers) - self.startp - 1):
            ida = self.idas[i]
            ida.startp = len(layers) -i - 2
            ida.endp = len(layers)
            layers = ida(layers)
            out.insert(0, layers[-1])
        return out


class DeconvDLA(nn.HybridBlock):
    """Deep layer aggregation network with deconv layers(smaller strides) which produce larger
    feature map.

    Parameters
    ----------
    base_network : str
        Name of the base feature extraction network, must be DLA networks.
    pretrained_base : bool
        Whether load pretrained base network.
    down_ratio : int
        The downsampling ratio of the network, must be one of [2, 4, 8, 16].
    last_level : int
        Index of the last output.
    out_channel : int
        The channel number of last output. If `0`, will use the channels of the first input.
    use_dcnv2 : bool
        Whether use ModulatedDeformableConvolution(DCN version 2).
    norm_layer : nn.HybridBlock, default is BatchNorm
        The norm layer type.
    norm_kwargs : dict
        Extra arguments to norm layer.

    """
    def __init__(self, base_network, pretrained_base, down_ratio,
                 last_level, out_channel=0, use_dcnv2=False,
                 norm_layer=nn.BatchNorm, norm_kwargs=None, **kwargs):
        super(DeconvDLA, self).__init__(**kwargs)
        assert down_ratio in [2, 4, 8, 16]
        self.first_level = int(np.log2(down_ratio))
        self.last_level = last_level
        self.base = get_model(base_network, pretrained=pretrained_base, use_feature=True)
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(self.first_level, channels[self.first_level:], scales,
                            use_dcnv2=use_dcnv2, norm_layer=norm_layer, norm_kwargs=norm_kwargs)

        if out_channel == 0:
            out_channel = channels[self.first_level]

        self.ida_up = IDAUp(out_channel, channels[self.first_level:self.last_level],
                            [2 ** i for i in range(self.last_level - self.first_level)],
                            use_dcnv2=use_dcnv2, norm_layer=norm_layer, norm_kwargs=norm_kwargs)

    def hybrid_forward(self, F, x):
        """Forward pass"""
        x = self.base(x)
        x = self.dla_up(x)

        y = []
        for i in range(self.last_level - self.first_level):
            y.append(x[i])
        self.ida_up.startp = 0
        self.ida_up.endp = len(y)
        self.ida_up(y)

        return y[-1]

def get_deconv_dla(base_network, pretrained=False, ctx=cpu(), scale=4.0, use_dcnv2=False, **kwargs):
    """Get resnet with deconv layers.

    Parameters
    ----------
    base_network : str
        Name of the base feature extraction network.
    pretrained : bool
        Whether load pretrained base network.
    ctx : mxnet.Context
        mx.cpu() or mx.gpu()
    scale : int or float, default is 4.0
        The downsampling ratio for the network, must in [2, 4, 8, 16]
    use_dcnv2 : bool
        If true, will use DCNv2 layers in upsampling blocks
    pretrained : type
        Description of parameter `pretrained`.
    Returns
    -------
    get_deconv_resnet(base_network, pretrained=False,
        Description of returned object.

    """
    assert int(scale) in [2, 4, 8, 16], "scale must be one of [2, 4, 8, 16]"
    net = DeconvDLA(base_network=base_network, pretrained_base=pretrained,
                    down_ratio=int(scale), last_level=5, use_dcnv2=use_dcnv2, **kwargs)
    with warnings.catch_warnings(record=True) as _:
        warnings.simplefilter("always")
        net.initialize()
    net.collect_params().reset_ctx(ctx)
    return net

def dla34_deconv(**kwargs):
    """DLA34 model with deconv layers.

    Returns
    -------
    HybridBlock
        A DLA34 model with deconv layers.

    """
    kwargs['use_dcnv2'] = False
    return get_deconv_dla('dla34', **kwargs)

def dla34_deconv_dcnv2(**kwargs):
    """DLA34 model with deconv layers and modulated deformable convolution layers.

    Returns
    -------
    HybridBlock
        A DLA34 model with deconv layers.

    """
    kwargs['use_dcnv2'] = True
    return get_deconv_dla('dla34', **kwargs)
