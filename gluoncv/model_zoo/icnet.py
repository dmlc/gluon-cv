# pylint: disable=unused-argument,abstract-method,missing-docstring
"""Image Cascade Network (ICNet)"""
from __future__ import division
from mxnet.gluon import nn
from mxnet.context import cpu
from mxnet.gluon.nn import HybridBlock
from gluoncv.model_zoo.segbase import SegBaseModel
from gluoncv.model_zoo.pspnet import _PSPHead
from gluoncv.loss import SoftmaxCrossEntropyLoss

import mxnet as mx
from mxnet import nd

__all__ = ['ICNet', 'get_icnet', 'ICNetLoss']


class ICNetLoss(SoftmaxCrossEntropyLoss):
    """Weighted SoftmaxCrossEntropyLoss2D

    Parameters
    ----------
    weights : tuple, default (0.4, 0.4, 0.4)
        The weight for cascade label guidance.
    """
    def __init__(self, weights=(0.4, 0.4, 0.4), **kwargs):
        super(ICNetLoss, self).__init__(**kwargs)
        self.weights = weights

    def _weighted_forwarad(self, F, scale_pred1, scale_pred2, scale_pred3, scale_pred4, label, **kwargs):
        h, w = label.shape[1], label.shape[2]

        scale_pred = F.contrib.BilinearResize2D(scale_pred1, height=h, width=w)
        loss1 = super(ICNetLoss, self).hybrid_forward(F, scale_pred, label)

        scale_pred = F.contrib.BilinearResize2D(scale_pred2, height=h, width=w)
        loss2 = super(ICNetLoss, self).hybrid_forward(F, scale_pred, label)

        scale_pred = F.contrib.BilinearResize2D(scale_pred3, height=h, width=w)
        loss3 = super(ICNetLoss, self).hybrid_forward(F, scale_pred, label)

        scale_pred = F.contrib.BilinearResize2D(scale_pred4, height=h, width=w)
        loss4 = super(ICNetLoss, self).hybrid_forward(F, scale_pred, label)

        return loss1 + self.weights[0] * loss2 + \
               self.weights[1] * loss3 + self.weights[2] * loss4

    def hybrid_forward(self, F, *inputs, **kwargs):
        """Compute loss"""
        return self._weighted_forwarad(F, *inputs, **kwargs)


class ICNet(SegBaseModel):
    r"""Image Cascade Network (ICNet)

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    pretrained_base : bool or str
        Refers to if the FCN backbone or the encoder is pretrained or not. If `True`,
        model weights of a model that was trained on ImageNet is loaded.


    Reference:

        Hengshuang Zhao, Xiaojuan Qi, Xiaoyong Shen, Jianping Shi and Jiaya Jia. "ICNet for Real-Time
        Semantic Segmentation on High-Resolution Images." *ECCV*, 2018

    Examples
    --------
    >>> model = ICNet(nclass=19, backbone='resnet50')
    >>> print(model)
    """
    # pylint: disable=arguments-differ
    def __init__(self, nclass, backbone='resnet50', aux=False, ctx=cpu(), pretrained_base=True,
                 base_size=520, crop_size=480, **kwargs):
        super(ICNet, self).__init__(nclass, aux=aux, backbone=backbone, ctx=ctx, base_size=base_size,
                                    crop_size=crop_size, pretrained_base=pretrained_base, **kwargs)

        with self.name_scope():
            self.maxpool = nn.MaxPool2D(pool_size=3, strides=2, padding=1, ceil_mode=True)

            base_psp_head = _PSPHead(nclass, feature_map_height=int(round(self._up_kwargs['height'] / 32)),
                                     feature_map_width=int(round(self._up_kwargs['width'] / 32)), **kwargs)
            self.psp_head = nn.HybridSequential()
            with self.psp_head.name_scope():
                self.psp_head.add(
                    base_psp_head.psp,
                    base_psp_head.block[:-1]
                )
            self.psp_head.initialize(ctx=ctx)
            self.psp_head.collect_params().setattr('lr_mult', 10)

            self.head = _ICHead(nclass=nclass)
            self.head.initialize(ctx=ctx)
            self.head.collect_params().setattr('lr_mult', 10)

            self.conv_sub1 = nn.HybridSequential()
            with self.conv_sub1.name_scope():
                self.conv_sub1.add(ConvBnRelu(3, 32, 3, 2, 1),
                                   ConvBnRelu(32, 32, 3, 2, 1),
                                   ConvBnRelu(32, 64, 3, 2, 1))
            self.conv_sub1.initialize(ctx=ctx)
            self.conv_sub1.collect_params().setattr('lr_mult', 10)

            self.conv_sub4 = ConvBnRelu(512, 256, 1)
            self.conv_sub4.initialize(ctx=ctx)
            self.conv_sub4.collect_params().setattr('lr_mult', 10)

            self.conv_sub2 = ConvBnRelu(512, 256, 1)
            self.conv_sub2.initialize(ctx=ctx)
            self.conv_sub2.collect_params().setattr('lr_mult', 10)

    def hybrid_forward(self, F, x):
        # sub 1
        x_sub1_out = self.conv_sub1(x)

        # sub_2
        x_sub2 = F.contrib.BilinearResize2D(x, height=x.shape[2] // 2, width=x.shape[3] // 2)

        x = self.conv1(x_sub2)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x_sub2_out = self.layer2(x)

        # sub 4
        x_sub4 = F.contrib.BilinearResize2D(x_sub2_out, height=x_sub2_out.shape[2] // 2, width=x_sub2_out.shape[3] // 2)

        x = self.layer3(x_sub4)
        x = self.layer4(x)
        x_sub4_out = self.psp_head(x)

        x_sub4_out = self.conv_sub4(x_sub4_out)
        x_sub2_out = self.conv_sub2(x_sub2_out)

        res = self.head(x_sub1_out, x_sub2_out, x_sub4_out)

        return res


class ConvBnRelu(HybridBlock):
    def __init__(self, in_planes, out_planes, ksize, stride=1, pad=0, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm, bn_eps=1e-5,
                 has_relu=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        with self.name_scope():
            self.conv = nn.Conv2D(in_channels=in_planes, channels=out_planes, kernel_size=ksize,
                                  padding=pad, strides=stride,
                                  dilation=dilation, groups=groups, use_bias=has_bias)
            self.has_bn = has_bn
            self.has_relu = has_relu

            if self.has_bn:
                self.bn = norm_layer(in_channels=out_planes, epsilon=bn_eps)
            if self.has_relu:
                self.relu = nn.Activation('relu')

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x


class CascadeFeatureFusion(HybridBlock):
    def __init__(self, low_channels, high_channels, out_channels, nclass, norm_layer=nn.BatchNorm):
        super(CascadeFeatureFusion, self).__init__()

        with self.name_scope():
            self.conv_low = nn.HybridSequential()
            with self.conv_low.name_scope():
                self.conv_low.add(nn.Conv2D(in_channels=low_channels, channels=out_channels,
                                            kernel_size=3, padding=2, dilation=2, use_bias=False))
                self.conv_low.add(norm_layer(in_channels=out_channels))

            self.conv_hign = nn.HybridSequential()
            with self.conv_hign.name_scope():
                self.conv_hign.add(nn.Conv2D(in_channels=high_channels, channels=out_channels,
                                             kernel_size=1, use_bias=False))
                self.conv_hign.add(norm_layer(in_channels=out_channels))

            self.conv_low_cls = nn.Conv2D(in_channels=out_channels, channels=nclass,
                                          kernel_size=1, use_bias=False)

    def hybrid_forward(self, F, x_low, x_high):
        x_low = F.contrib.BilinearResize2D(x_low, height=x_high.shape[2], width=x_high.shape[3])
        x_low = self.conv_low(x_low)
        x_high = self.conv_hign(x_high)

        x = x_low + x_high
        x = F.relu(x)

        x_low_cls = self.conv_low_cls(x_low)

        return x, x_low_cls


class _ICHead(HybridBlock):
    # pylint: disable=redefined-outer-name
    def __init__(self, nclass, norm_layer=nn.BatchNorm):
        super(_ICHead, self).__init__()
        self.cff_12 = CascadeFeatureFusion(128, 64, 128, nclass, norm_layer)
        self.cff_24 = CascadeFeatureFusion(256, 256, 128, nclass, norm_layer)

        with self.name_scope():
            self.conv_cls = nn.Conv2D(in_channels=128, channels=nclass,
                                      kernel_size=1, use_bias=False)

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, x_sub1, x_sub2, x_sub4):
        outputs = []

        x_cff_24, x_24_cls = self.cff_24(x_sub4, x_sub2)
        outputs.append(x_24_cls)
        x_cff_12, x_12_cls = self.cff_12(x_cff_24, x_sub1)
        outputs.append(x_12_cls)

        up_x2 = F.contrib.BilinearResize2D(x_cff_12, height=2*x_cff_12.shape[2], width=2*x_cff_12.shape[3])
        up_x2 = self.conv_cls(up_x2)
        outputs.append(up_x2)
        up_x8 = F.contrib.BilinearResize2D(up_x2, height=4*up_x2.shape[2], width=4*up_x2.shape[3])
        outputs.append(up_x8)
        # 1 -> 1/4 -> 1/8 -> 1/16
        outputs.reverse()
        return tuple(outputs)


def get_icnet(nclass, backbone='resnet50',
              ctx=cpu(0), pretrained_base=True, **kwargs):
    model = ICNet(nclass=nclass, backbone=backbone,
                  pretrained_base=pretrained_base, ctx=ctx, **kwargs)
    model.classes = nclass

    return model


if __name__ == '__main__':
    ctx = mx.cpu()
    # ctx = mx.gpu(0)

    crop_size = 1024
    x = nd.random.uniform(shape=(1, 3, 1024, 2048)).as_in_context(ctx)
    # print(x)
    model = get_icnet(nclass=19, pretrained_base=False, crop_size=crop_size, height=1024, width=2048)
    model.initialize(force_reinit=True, ctx=ctx)
    # print(model)
    import time

    t_cpu = 0.
    num = 100
    res = None
    for i in range(num):
        tic = time.time()
        res = model(x)
        t_cpu += time.time() - tic

    print('compuational time: %0.2f ms' % (t_cpu*1000/num))
    print("ICnet output length: ", len(res))
    for i in res:
        print(i.shape)
