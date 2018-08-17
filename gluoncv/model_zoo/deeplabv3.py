# pylint: disable=unused-argument
"""Pyramid Scene Parsing Network"""
from mxnet.gluon import nn
from mxnet.context import cpu
from mxnet.gluon.nn import HybridBlock
from mxnet import gluon
from .segbase import SegBaseModel
from .fcn import _FCNHead
# pylint: disable-all

__all__ = ['DeepLabV3', 'get_deeplabv3', 'get_deeplabv3_ade_resnet50']

class DeepLabV3(SegBaseModel):
    r"""DeepLabV3

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
    aux : bool
        Auxilary loss.


    Reference:

        Chen, Liang-Chieh, et al. "Rethinking atrous convolution for semantic image segmentation."
        arXiv preprint arXiv:1706.05587 (2017).

    """
    def __init__(self, nclass, backbone='resnet50', aux=True, ctx=cpu(), pretrained_base=True,
                 **kwargs):
        super(DeepLabV3, self).__init__(nclass, aux, backbone, ctx=ctx, pretrained_base=True,
                                        **kwargs)
        with self.name_scope():
            self.head = _DeepLabHead(nclass, **kwargs)
            self.head.initialize(ctx=ctx)
            self.head.collect_params().setattr('lr_mult', 10)
            if self.aux:
                self.auxlayer = _FCNHead(1024, nclass, **kwargs)
                self.auxlayer.initialize(ctx=ctx)
                self.auxlayer.collect_params().setattr('lr_mult', 10)

    def hybrid_forward(self, F, x):
        c3, c4 = self.base_forward(x)
        outputs = []
        x = self.head(c4)
        x = F.contrib.BilinearResize2D(x, **self._up_kwargs)
        outputs.append(x)

        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.contrib.BilinearResize2D(auxout, **self._up_kwargs)
            outputs.append(auxout)
        return tuple(outputs)


class _DeepLabHead(HybridBlock):
    def __init__(self, nclass, norm_layer, norm_kwargs={}, **kwargs):
        super(_DeepLabHead, self).__init__()
        self.aspp = _ASPP(2048, [12, 24, 36], norm_layer=norm_layer,
                               norm_kwargs=norm_kwargs, **kwargs)
        with self.name_scope():
            self.block = nn.HybridSequential()
            self.block.add(nn.Conv2D(in_channels=256, channels=256,
                                     kernel_size=3, padding=1, use_bias=False))
            self.block.add(norm_layer(in_channels=256, **norm_kwargs))
            self.block.add(nn.Activation('relu'))
            self.block.add(nn.Dropout(0.1))
            self.block.add(nn.Conv2D(in_channels=256, channels=nclass,
                                     kernel_size=1))

    def hybrid_forward(self, F, x):
        x = self.aspp(x)
        return self.block(x)


def _ASPPConv(in_channels, out_channels, atrous_rate, norm_layer, norm_kwargs):
    block = nn.HybridSequential()
    with block.name_scope():
        block.add(nn.Conv2D(in_channels=in_channels, channels=out_channels,
                            kernel_size=3, padding=atrous_rate,
                            dilation=atrous_rate, use_bias=False))
        block.add(norm_layer(in_channels=out_channels, **norm_kwargs))
        block.add(nn.Activation('relu'))
    return block

class _AsppPooling(nn.HybridBlock):
    def __init__(self, in_channels, out_channels, norm_layer, norm_kwargs):
        super(_AsppPooling, self).__init__()
        self.gap = nn.HybridSequential()
        with self.gap.name_scope():
            self.gap.add(nn.GlobalAvgPool2D())
            self.gap.add(nn.Conv2D(in_channels=in_channels, channels=out_channels,
                                   kernel_size=1, use_bias=False))
            self.gap.add(norm_layer(in_channels=out_channels, **norm_kwargs))
            self.gap.add(nn.Activation("relu"))

    def hybrid_forward(self, F, x):
        *_, h, w = x.shape
        pool = self.gap(x)
        return F.contrib.BilinearResize2D(pool, height=h, width=w)

class _ASPP(nn.HybridBlock):
    def __init__(self, in_channels, atrous_rates, norm_layer, norm_kwargs):
        super(_ASPP, self).__init__()

        out_channels = 256
        b0 = nn.HybridSequential()
        with b0.name_scope():
            b0.add(nn.Conv2D(in_channels=in_channels, channels=out_channels,
                             kernel_size=1, use_bias=False))
            b0.add(norm_layer(in_channels=out_channels, **norm_kwargs))
            b0.add(nn.Activation("relu"))

        rate1, rate2, rate3 = tuple(atrous_rates)

        b1 = _ASPPConv(in_channels, out_channels, rate1, norm_layer, norm_kwargs)
        b2 = _ASPPConv(in_channels, out_channels, rate2, norm_layer, norm_kwargs)
        b3 = _ASPPConv(in_channels, out_channels, rate3, norm_layer, norm_kwargs)
        b4 = _AsppPooling(in_channels, out_channels, norm_layer=norm_layer,
                          norm_kwargs=norm_kwargs)

        self.concurent = gluon.contrib.nn.HybridConcurrent(axis=1)
        self.concurent.add(b0)
        self.concurent.add(b1)
        self.concurent.add(b2)
        self.concurent.add(b3)
        self.concurent.add(b4)

        self.project = nn.HybridSequential()
        self.project.add(nn.Conv2D(in_channels=5*out_channels, channels=out_channels,
                                   kernel_size=1, use_bias=False))
        self.project.add(norm_layer(in_channels=out_channels, **norm_kwargs))
        self.project.add(nn.Activation("relu"))
        self.project.add(nn.Dropout(0.5))

    def hybrid_forward(self, F, x):
        return self.project(self.concurent(x))


def get_deeplabv3(dataset='pascal_voc', backbone='resnet50', pretrained=False,
            root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    r"""DeepLabV3
    Parameters
    ----------
    dataset : str, default pascal_voc
        The dataset that model pretrained on. (pascal_voc, ade20k)
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_fcn(dataset='pascal_voc', backbone='resnet50', pretrained=False)
    >>> print(model)
    """
    from ..data.pascal_voc.segmentation import VOCSegmentation
    from ..data.pascal_aug.segmentation import VOCAugSegmentation
    from ..data.ade20k.segmentation import ADE20KSegmentation
    from ..data.mscoco.segmentation import COCOSegmentation
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_aug': 'voc',
        'ade20k': 'ade',
        'coco': 'coco',
    }
    datasets = {
        'pascal_voc': VOCSegmentation,
        'pascal_aug': VOCAugSegmentation,
        'ade20k': ADE20KSegmentation,
        'coco': COCOSegmentation,
    }
    # infer number of classes
    model = DeepLabV3(datasets[dataset].NUM_CLASS, backbone=backbone, ctx=ctx, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_parameters(get_model_file('deeplabv3_%s_%s'%(backbone, acronyms[dataset]),
                                         root=root), ctx=ctx)
    return model


def get_deeplabv3_ade_resnet50(**kwargs):
    r"""DeepLabV3
    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_fcn_ade_resnet50(pretrained=True)
    >>> print(model)
    """
    return get_deeplabv3('ade20k', 'resnet50', **kwargs)
