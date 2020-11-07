# pylint: disable=unused-argument,abstract-method,missing-docstring,arguments-differ
"""Image Cascade Network (ICNet)
ICNet for Real-Time Semantic Segmentation on High-Resolution Images, ECCV 2018
https://hszhao.github.io/projects/icnet/
Code partially borrowed from https://github.com/lxtGH/Fast_Seg/blob/master/libs/models/ICNet.py.
"""
from __future__ import division
from mxnet.gluon import nn
from mxnet.context import cpu
from mxnet.gluon.nn import HybridBlock
from gluoncv.model_zoo.segbase import SegBaseModel
from gluoncv.model_zoo.pspnet import _PSPHead

__all__ = ['ICNet', 'get_icnet', 'get_icnet_resnet50_citys', 'get_icnet_resnet50_mhpv1']

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
        for Synchronized Cross-GPU BachNormalization).
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    pretrained_base : bool or str
        Refers to if the backbone is pretrained or not. If `True`,
        model weights of a model that was trained on ImageNet is loaded.

    Reference:

        Hengshuang Zhao, Xiaojuan Qi, Xiaoyong Shen, Jianping Shi and Jiaya Jia.
        "ICNet for Real-Time Semantic Segmentation on High-Resolution Images." *ECCV*, 2018

    Examples
    --------
    >>> model = ICNet(nclass=19, backbone='resnet50')
    >>> print(model)
    """

    def __init__(self, nclass, backbone='resnet50', aux=False, ctx=cpu(), pretrained_base=True,
                 height=None, width=None, base_size=520, crop_size=480, lr_mult=10, **kwargs):
        super(ICNet, self).__init__(nclass, aux=aux, backbone=backbone, ctx=ctx,
                                    base_size=base_size, crop_size=crop_size,
                                    pretrained_base=pretrained_base, **kwargs)

        height = height if height is not None else crop_size
        width = width if width is not None else crop_size
        self._up_kwargs = {'height': height, 'width': width}
        self.base_size = base_size
        self.crop_size = crop_size

        with self.name_scope():
            # large resolution branch
            self.conv_sub1 = nn.HybridSequential()
            with self.conv_sub1.name_scope():
                self.conv_sub1.add(ConvBnRelu(3, 32, 3, 2, 1, **kwargs),
                                   ConvBnRelu(32, 32, 3, 2, 1, **kwargs),
                                   ConvBnRelu(32, 64, 3, 2, 1, **kwargs))
            self.conv_sub1.initialize(ctx=ctx)
            self.conv_sub1.collect_params().setattr('lr_mult', lr_mult)

            # small and medium resolution branches, backbone comes from segbase.py
            self.psp_head = _PSPHead(nclass,
                                     feature_map_height=self._up_kwargs['height'] // 32,
                                     feature_map_width=self._up_kwargs['width'] // 32,
                                     **kwargs)
            self.psp_head.block = self.psp_head.block[:-1]
            self.psp_head.initialize(ctx=ctx)
            self.psp_head.collect_params().setattr('lr_mult', lr_mult)

            # ICNet head
            self.head = _ICHead(nclass=nclass,
                                height=self._up_kwargs['height'],
                                width=self._up_kwargs['width'],
                                **kwargs)
            self.head.initialize(ctx=ctx)
            self.head.collect_params().setattr('lr_mult', lr_mult)

            # reduce conv
            self.conv_sub4 = ConvBnRelu(512, 256, 1, **kwargs)
            self.conv_sub4.initialize(ctx=ctx)
            self.conv_sub4.collect_params().setattr('lr_mult', lr_mult)

            self.conv_sub2 = ConvBnRelu(512, 256, 1, **kwargs)
            self.conv_sub2.initialize(ctx=ctx)
            self.conv_sub2.collect_params().setattr('lr_mult', lr_mult)

    def hybrid_forward(self, F, x):
        # large resolution branch
        x_sub1_out = self.conv_sub1(x)

        # medium resolution branch
        x_sub2 = F.contrib.BilinearResize2D(x,
                                            height=self._up_kwargs['height'] // 2,
                                            width=self._up_kwargs['width'] // 2)
        x = self.conv1(x_sub2)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x_sub2_out = self.layer2(x)

        # small resolution branch
        x_sub4 = F.contrib.BilinearResize2D(x_sub2_out,
                                            height=self._up_kwargs['height'] // 32,
                                            width=self._up_kwargs['width'] // 32)
        x = self.layer3(x_sub4)
        x = self.layer4(x)
        x_sub4_out = self.psp_head(x)

        # reduce conv
        x_sub4_out = self.conv_sub4(x_sub4_out)
        x_sub2_out = self.conv_sub2(x_sub2_out)

        # ICNet head
        res = self.head(x_sub1_out, x_sub2_out, x_sub4_out)
        return res

    def demo(self, x):
        return self.predict(x)

    def predict(self, x):
        h, w = x.shape[2:]
        self._up_kwargs['height'] = h
        self._up_kwargs['width'] = w

        import mxnet.ndarray as F
        x_sub1_out = self.conv_sub1(x)

        x_sub2 = F.contrib.BilinearResize2D(x,
                                            height=self._up_kwargs['height'] // 2,
                                            width=self._up_kwargs['width'] // 2)
        x = self.conv1(x_sub2)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x_sub2_out = self.layer2(x)

        x_sub4 = F.contrib.BilinearResize2D(x_sub2_out,
                                            height=self._up_kwargs['height'] // 32,
                                            width=self._up_kwargs['width'] // 32)
        x = self.layer3(x_sub4)
        x = self.layer4(x)
        x_sub4_out = self.psp_head.demo(x)

        x_sub4_out = self.conv_sub4(x_sub4_out)
        x_sub2_out = self.conv_sub2(x_sub2_out)
        res = self.head.demo(x_sub1_out, x_sub2_out, x_sub4_out)
        return res[0]

class _ICHead(HybridBlock):
    # pylint: disable=redefined-outer-name
    def __init__(self, nclass, height=None, width=None,
                 norm_layer=nn.BatchNorm, **kwargs):
        super(_ICHead, self).__init__()

        self._up_kwargs = {'height': height, 'width': width}
        self.cff_12 = CascadeFeatureFusion(low_channels=128,
                                           high_channels=64,
                                           out_channels=128,
                                           nclass=nclass,
                                           height=height // 8,
                                           width=width // 8,
                                           norm_layer=norm_layer,
                                           **kwargs)
        self.cff_24 = CascadeFeatureFusion(low_channels=256,
                                           high_channels=256,
                                           out_channels=128,
                                           nclass=nclass,
                                           height=height // 16,
                                           width=width // 16,
                                           norm_layer=norm_layer,
                                           **kwargs)

        with self.name_scope():
            self.conv_cls = nn.Conv2D(in_channels=128, channels=nclass,
                                      kernel_size=1, use_bias=False)

    def hybrid_forward(self, F, x_sub1, x_sub2, x_sub4):
        outputs = []

        x_cff_24, x_24_cls = self.cff_24(x_sub4, x_sub2)
        outputs.append(x_24_cls)
        x_cff_12, x_12_cls = self.cff_12(x_cff_24, x_sub1)
        outputs.append(x_12_cls)

        up_x2 = F.contrib.BilinearResize2D(x_cff_12,
                                           height=self._up_kwargs['height'] // 4,
                                           width=self._up_kwargs['width'] // 4)
        up_x2 = self.conv_cls(up_x2)
        outputs.append(up_x2)

        up_x8 = F.contrib.BilinearResize2D(up_x2,
                                           height=self._up_kwargs['height'],
                                           width=self._up_kwargs['width'])
        outputs.append(up_x8)

        # 1 -> 1/4 -> 1/8 -> 1/16
        outputs.reverse()
        return tuple(outputs)

    def demo(self, x_sub1, x_sub2, x_sub4):
        outputs = []

        x_cff_24, x_24_cls = self.cff_24.demo(x_sub4, x_sub2)
        outputs.append(x_24_cls)
        x_cff_12, x_12_cls = self.cff_12.demo(x_cff_24, x_sub1)
        outputs.append(x_12_cls)

        import mxnet.ndarray as F
        up_x2 = F.contrib.BilinearResize2D(x_cff_12,
                                           height=x_cff_12.shape[2] * 2,
                                           width=x_cff_12.shape[3] * 2)
        up_x2 = self.conv_cls(up_x2)
        outputs.append(up_x2)

        up_x8 = F.contrib.BilinearResize2D(up_x2,
                                           height=up_x2.shape[2] * 4,
                                           width=up_x2.shape[3] * 4)
        outputs.append(up_x8)

        # 1 -> 1/4 -> 1/8 -> 1/16
        outputs.reverse()
        return tuple(outputs)

class CascadeFeatureFusion(HybridBlock):
    def __init__(self, low_channels, high_channels, out_channels,
                 nclass, height=None, width=None,
                 norm_layer=nn.BatchNorm, **kwargs):
        super(CascadeFeatureFusion, self).__init__()
        self._up_kwargs = {'height': height, 'width': width}

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
        x_low = F.contrib.BilinearResize2D(x_low,
                                           height=self._up_kwargs['height'],
                                           width=self._up_kwargs['width'])
        x_low = self.conv_low(x_low)
        x_high = self.conv_hign(x_high)

        x = x_low + x_high
        x = F.relu(x)

        x_low_cls = self.conv_low_cls(x_low)
        return x, x_low_cls

    def demo(self, x_low, x_high):
        self._up_kwargs['height'] = x_high.shape[2]
        self._up_kwargs['width'] = x_high.shape[3]

        import mxnet.ndarray as F
        x_low = F.contrib.BilinearResize2D(x_low,
                                           height=self._up_kwargs['height'],
                                           width=self._up_kwargs['width'])
        x_low = self.conv_low(x_low)
        x_high = self.conv_hign(x_high)

        x = x_low + x_high
        x = F.relu(x)

        x_low_cls = self.conv_low_cls(x_low)
        return x, x_low_cls

class ConvBnRelu(HybridBlock):
    def __init__(self, in_planes, out_planes, ksize, stride=1, pad=0, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm, bn_eps=1e-5,
                 has_relu=True, has_bias=False, **kwargs):
        super(ConvBnRelu, self).__init__()
        with self.name_scope():
            self.conv = nn.Conv2D(in_channels=in_planes, channels=out_planes,
                                  kernel_size=ksize, padding=pad, strides=stride,
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

    def demo(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)
        return x

def get_icnet(dataset='citys', backbone='resnet50', pretrained=False,
              root='~/.mxnet/models', pretrained_base=True, ctx=cpu(0), **kwargs):
    r"""Image Cascade Network

    Parameters
    ----------
    dataset : str, default citys
        The dataset that model pretrained on. (default: cityscapes)
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    pretrained_base : bool or str, default True
        This will load pretrained backbone network, that was trained on ImageNet.

    """
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_aug': 'voc',
        'ade20k': 'ade',
        'coco': 'coco',
        'citys': 'citys',
        'mhpv1': 'mhpv1',
    }
    from ..data import datasets
    # infer number of classes
    model = ICNet(datasets[dataset].NUM_CLASS, backbone=backbone,
                  pretrained_base=pretrained_base, ctx=ctx, **kwargs)
    model.classes = datasets[dataset].CLASSES

    if pretrained:
        from .model_store import get_model_file
        model.load_parameters(get_model_file('icnet_%s_%s' % (backbone, acronyms[dataset]),
                                             tag=pretrained, root=root), ctx=ctx)
    return model


def get_icnet_resnet50_citys(**kwargs):
    r"""Image Cascade Network

    Parameters
    ----------
    dataset : str, default citys
        The dataset that model pretrained on. (default: cityscapes)
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50').

    """
    return get_icnet(dataset='citys', backbone='resnet50', **kwargs)


def get_icnet_resnet50_mhpv1(**kwargs):
    r"""Image Cascade Network

    Parameters
    ----------
    dataset : str, default citys
        The dataset that model pretrained on. (default: cityscapes)
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50').

    """
    return get_icnet(dataset='mhpv1', backbone='resnet50', **kwargs)
