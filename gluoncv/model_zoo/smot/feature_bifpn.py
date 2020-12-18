"""
Network utiliy functions
"""
# pylint: disable=line-too-long,unused-argument,missing-function-docstring
from __future__ import absolute_import
import logging

import mxnet as mx
from mxnet.gluon import HybridBlock
from mxnet.gluon import nn
from mxnet.gluon.contrib import nn as contrib_nn

from .mobilenet import get_mobilenet, Mish


class FPNFeatureExpander(HybridBlock):
    """Feature extractor with additional layers to append.
    This is specified for ``Feature Pyramid Network for Object Detection``
    which implement ``Top-down pathway and lateral connections``.

    Parameters
    ----------
    network : str or HybridBlock or Symbol
        Logic chain: load from gluon.model_zoo.vision if network is string.
        Convert to Symbol if network is HybridBlock.
    outputs : str or list of str
        The name of layers to be extracted as features
    num_filters : list of int e.g. [256, 256, 256, 256]
        Number of filters to be appended.
    use_1x1 : bool
        Whether to use 1x1 convolution
    use_upsample : bool
        Whether to use upsample
    use_elewadd : float
        Whether to use element-wise add operation
    use_p6 : bool
        Whther use P6 stage, this is used for RPN experiments in ori paper
    no_bias : bool
        Whether use bias for Convolution operation.
    norm_layer : HybridBlock or SymbolBlock
        Type of normalization layer.
    norm_kwargs : dict
        Arguments for normalization layer.
    pretrained : bool
        Use pretrained parameters as in gluon.model_zoo if `True`.
    ctx : Context
        The context, e.g. mxnet.cpu(), mxnet.gpu(0).
    inputs : list of str
        Name of input variables to the network.

    """

    def __init__(self, network, outputs, ssd_filters, fpn_filters, use_1x1=True, use_upsample=True,
                 use_elewadd=True, use_p6=False, no_bias=True, pretrained=False, norm_layer='BatchNorm',
                 norm_kwargs=None, ctx=mx.cpu(), inputs=('data',), use_bn=True, reduce_ratio=1.0,
                 min_depth=128, global_pool=False, use_mish=False, **kwargs):
        super(FPNFeatureExpander, self).__init__(**kwargs)

        gluon_norm_kwargs = {}
        if norm_kwargs is not None and 'num_devices' in norm_kwargs:
            gluon_norm_kwargs = {'num_devices': norm_kwargs['num_devices']}

        if norm_layer == nn.BatchNorm:
            logging.info("use BatchNorm for the backbone model.")
            network = get_mobilenet(1.0, pretrained=pretrained, ctx=ctx, mish=use_mish)
        else:
            logging.info("use SyncBatchNorm for the backbone model.")
            network = get_mobilenet(1.0, pretrained=pretrained, ctx=ctx,
                                    norm_layer=mx.gluon.contrib.nn.SyncBatchNorm,
                                    norm_kwargs=gluon_norm_kwargs, mish=use_mish)

        self.output_num = len(fpn_filters)
        self.norm_layer = norm_layer
        net = network.features
        with self.name_scope():
            self.ssd_backbone = nn.HybridSequential()
            self.build_output(net, [0, 21])
            self.build_output(net, [21, 33])
            self.build_output(net, [33, 69])
            self.build_output(net, [69, 81])
            for num_filter in ssd_filters:
                self.ssd_backbone.add(build_down_sample(num_filter, 2, self.norm_layer, gluon_norm_kwargs, use_mish=use_mish))

            self.upper_feat = nn.HybridSequential()
            self.lower_feat = nn.HybridSequential()
            self.fuse_feat = nn.HybridSequential()
            self.deconv_feat = nn.HybridSequential()

            self.fuse_feat_late = nn.HybridSequential()
            self.fpn_conv = nn.HybridSequential()
            self.downsample_conv = nn.HybridSequential()

            for idx in range(self.output_num):
                self.upper_feat.add(convolution(fpn_filters[idx], 1, self.norm_layer, gluon_norm_kwargs, pad=0, use_mish=use_mish))
                self.deconv_feat.add(deform_convolution(fpn_filters[idx], self.norm_layer, gluon_norm_kwargs, use_mish=use_mish))
                self.fuse_feat_late.add(convolution(fpn_filters[idx], 3, self.norm_layer, gluon_norm_kwargs, use_mish=use_mish))
                if idx > 0:
                    self.lower_feat.add(convolution(fpn_filters[idx], 1, self.norm_layer, gluon_norm_kwargs, pad=0, use_mish=use_mish))
                    self.fuse_feat.add(convolution(fpn_filters[idx], 3, self.norm_layer, gluon_norm_kwargs, use_mish=use_mish))
                    self.fpn_conv.add(convolution(fpn_filters[idx], 3, self.norm_layer, gluon_norm_kwargs, use_mish=use_mish))
                    self.downsample_conv.add(convolution(fpn_filters[idx], 3, self.norm_layer,
                                                         gluon_norm_kwargs, stride=2, pad=1, use_mish=use_mish))

    def hybrid_forward(self, F, x):
        forward_feature = []
        for i in range(self.output_num):
            x = self.ssd_backbone[i](x)
            forward_feature.append(x)

        aggregate_feat = []
        trans_feat = []
        for i in range(len(forward_feature) - 1, -1, -1):
            if i == len(forward_feature) - 1:
                P_fuse = self.upper_feat[i](forward_feature[i])
                trans_feat.append(P_fuse)
                aggregate_feat.append(P_fuse)
                continue
            P_up = self.upper_feat[i](
                F.UpSampling(aggregate_feat[-1], scale=2, sample_type='nearest', workspace=512, num_args=1))
            P_low = self.lower_feat[i](forward_feature[i])
            trans_feat.append(P_low)
            P_clip = F.Crop(*[P_up, P_low])
            P_sum = F.ElementWiseSum(*[P_clip, P_low])
            P_fuse = self.fuse_feat[i](P_sum)
            aggregate_feat.append(P_fuse)

        bi_fpn_feat = []
        for i in range(len(aggregate_feat) - 1, -1, -1):
            if i == len(aggregate_feat) - 1:
                P_fuse_late = self.fuse_feat_late[i](aggregate_feat[i])
                bi_fpn_feat.append(P_fuse_late)
                continue
            f_down = self.downsample_conv[i](bi_fpn_feat[-1])
            f_up = self.fpn_conv[i](aggregate_feat[i])
            f_clip = F.Crop(*[f_down, f_up])
            f_sum_0 = F.ElementWiseSum(*[f_clip, f_up])
            if i == 0:
                f_fuse = self.fuse_feat_late[i](f_sum_0)
                bi_fpn_feat.append(f_fuse)
                continue
            f_clip_sum = F.Crop(*[f_sum_0, trans_feat[i]])
            f_sum_1 = F.ElementWiseSum(*[f_clip_sum, trans_feat[i]])
            f_fuse = self.fuse_feat_late[i](f_sum_1)
            bi_fpn_feat.append(f_fuse)

        deform_feat = []
        for i, _ in enumerate(bi_fpn_feat):
            deform_feat.append(self.deconv_feat[i](bi_fpn_feat[i]))

        return deform_feat, bi_fpn_feat

    def build_output(self, net, layer_range):
        assert len(layer_range) == 2
        branch = nn.HybridSequential()
        for idx in range(layer_range[0], layer_range[1]):
            branch.add(net[idx])
        self.ssd_backbone.add(branch)


def build_down_sample(num_filters, stride, norm_layer, norm_kwargs, use_mish=False):
    """stack two Conv-BatchNorm-Relu blocks and then a pooling layer
    to halve the feature size"""
    out = nn.HybridSequential()
    out.add(nn.Conv2D(num_filters, 1, strides=1, padding=0))

    if norm_layer == nn.BatchNorm:
        out.add(nn.BatchNorm(in_channels=num_filters))
    elif norm_layer == contrib_nn.SyncBatchNorm:
        out.add(mx.gluon.contrib.nn.SyncBatchNorm(in_channels=num_filters, num_devices=norm_kwargs['num_devices']))
    else:
        raise ValueError("Unknown norm layer type: {}".format(norm_layer))
    if use_mish:
        out.add(Mish())
    else:
        out.add(nn.LeakyReLU(alpha=0.25))
    out.add(nn.Conv2D(num_filters, 3, strides=1, padding=1))
    if norm_layer == nn.BatchNorm:
        out.add(nn.BatchNorm(in_channels=num_filters))
    elif norm_layer == contrib_nn.SyncBatchNorm:
        out.add(mx.gluon.contrib.nn.SyncBatchNorm(in_channels=num_filters, num_devices=norm_kwargs['num_devices']))
    else:
        raise ValueError("Unknown norm layer type: {}".format(norm_layer))
    if use_mish:
        out.add(Mish())
    else:
        out.add(nn.LeakyReLU(alpha=0.25))
    out.add(nn.Conv2D(num_filters, 3, strides=stride, padding=1))
    if norm_layer == nn.BatchNorm:
        out.add(nn.BatchNorm(in_channels=num_filters))
    elif norm_layer == contrib_nn.SyncBatchNorm:
        out.add(mx.gluon.contrib.nn.SyncBatchNorm(in_channels=num_filters, num_devices=norm_kwargs['num_devices']))
    else:
        raise ValueError("Unknown norm layer type: {}".format(norm_layer))
    if use_mish:
        out.add(Mish())
    else:
        out.add(nn.LeakyReLU(alpha=0.25))
    return out


def convolution(num_filters, filter_size, norm_layer, norm_kwargs, stride=1, pad=1, use_mish=False):
    out = nn.HybridSequential()
    out.add(nn.Conv2D(num_filters, filter_size, padding=pad, strides=stride))
    if norm_layer == nn.BatchNorm:
        out.add(nn.BatchNorm(in_channels=num_filters))
    elif norm_layer == contrib_nn.SyncBatchNorm:
        out.add(mx.gluon.contrib.nn.SyncBatchNorm(in_channels=num_filters, num_devices=norm_kwargs['num_devices']))
    else:
        raise ValueError("Unknown norm layer type: {}".format(norm_layer))
    if use_mish:
        out.add(Mish())
    else:
        out.add(nn.LeakyReLU(alpha=0.25))
    return out


def deform_convolution(num_filters, norm_layer, norm_kwargs, use_mish=False):
    out = nn.HybridSequential()
    out.add(mx.gluon.contrib.cnn.DeformableConvolution(num_filters, kernel_size=(3, 3),
                                                       padding=(1, 1),
                                                       num_deformable_group=4,
                                                       weight_initializer=mx.init.Xavier(rnd_type='gaussian',
                                                                                         factor_type='out',
                                                                                         magnitude=2),
                                                       offset_weight_initializer=mx.init.Xavier(rnd_type='gaussian',
                                                                                                factor_type='out',
                                                                                                magnitude=2)))

    if norm_layer == nn.BatchNorm:
        out.add(nn.BatchNorm(in_channels=num_filters))
    elif norm_layer == contrib_nn.SyncBatchNorm:
        out.add(mx.gluon.contrib.nn.SyncBatchNorm(in_channels=num_filters, num_devices=norm_kwargs['num_devices']))
    else:
        raise ValueError("Unknown norm layer type: {}".format(norm_layer))
    if use_mish:
        out.add(Mish())
    else:
        out.add(nn.LeakyReLU(alpha=0.25))
    return out
