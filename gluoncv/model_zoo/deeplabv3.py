# pylint: disable=unused-argument
"""Pyramid Scene Parsing Network"""
from mxnet.gluon import nn
from mxnet.context import cpu
from mxnet.gluon.nn import HybridBlock
from mxnet import gluon
from .segbase import SegBaseModel
from .fcn import _FCNHead
# pylint: disable-all

__all__ = ['DeepLabV3', 'get_deeplab', 'get_deeplab_resnet101_coco',
    'get_deeplab_resnet101_voc', 'get_deeplab_resnet50_ade', 'get_deeplab_resnet101_ade',
    'get_deeplab_resnet152_coco', 'get_deeplab_resnet152_voc']

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
        Auxiliary loss.


    Reference:

        Chen, Liang-Chieh, et al. "Rethinking atrous convolution for semantic image segmentation."
        arXiv preprint arXiv:1706.05587 (2017).

    """
    def __init__(self, nclass, backbone='resnet50', aux=True, ctx=cpu(), pretrained_base=True,
                 base_size=520, crop_size=480, **kwargs):
        super(DeepLabV3, self).__init__(nclass, aux, backbone, ctx=ctx, base_size=base_size,
                                     crop_size=crop_size, pretrained_base=pretrained_base, **kwargs)
        with self.name_scope():
            self.head = _DeepLabHead(nclass, height=self._up_kwargs['height']//8,
                                     width=self._up_kwargs['width']//8, **kwargs)
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

    def demo(self, x):
        h, w = x.shape[2:]
        self._up_kwargs['height'] = h
        self._up_kwargs['width'] = w
        c3, c4 = self.base_forward(x)
        x = self.head.demo(c4)
        import mxnet.ndarray as F
        pred = F.contrib.BilinearResize2D(x, **self._up_kwargs)
        return pred


class _DeepLabHead(HybridBlock):
    def __init__(self, nclass, norm_layer=nn.BatchNorm, norm_kwargs=None, **kwargs):
        super(_DeepLabHead, self).__init__()
        with self.name_scope():
            self.aspp = _ASPP(2048, [12, 24, 36], norm_layer=norm_layer,
                              norm_kwargs=norm_kwargs, **kwargs)
            self.block = nn.HybridSequential()
            self.block.add(nn.Conv2D(in_channels=256, channels=256,
                                     kernel_size=3, padding=1, use_bias=False))
            self.block.add(norm_layer(in_channels=256, **({} if norm_kwargs is None else norm_kwargs)))
            self.block.add(nn.Activation('relu'))
            self.block.add(nn.Dropout(0.1))
            self.block.add(nn.Conv2D(in_channels=256, channels=nclass,
                                     kernel_size=1))

    def hybrid_forward(self, F, x):
        x = self.aspp(x)
        return self.block(x)

    def demo(self, x):
        h, w = x.shape[2:]
        self.aspp.concurent[-1]._up_kwargs['height'] = h
        self.aspp.concurent[-1]._up_kwargs['width'] = w
        x = self.aspp(x)
        return self.block(x)
        

def _ASPPConv(in_channels, out_channels, atrous_rate, norm_layer, norm_kwargs):
    block = nn.HybridSequential()
    with block.name_scope():
        block.add(nn.Conv2D(in_channels=in_channels, channels=out_channels,
                            kernel_size=3, padding=atrous_rate,
                            dilation=atrous_rate, use_bias=False))
        block.add(norm_layer(in_channels=out_channels, **({} if norm_kwargs is None else norm_kwargs)))
        block.add(nn.Activation('relu'))
    return block

class _AsppPooling(nn.HybridBlock):
    def __init__(self, in_channels, out_channels, norm_layer, norm_kwargs,
                 height=60, width=60, **kwargs):
        super(_AsppPooling, self).__init__()
        self.gap = nn.HybridSequential()
        self._up_kwargs = {'height': height, 'width': width}
        with self.gap.name_scope():
            self.gap.add(nn.GlobalAvgPool2D())
            self.gap.add(nn.Conv2D(in_channels=in_channels, channels=out_channels,
                                   kernel_size=1, use_bias=False))
            self.gap.add(norm_layer(in_channels=out_channels,
                                    **({} if norm_kwargs is None else norm_kwargs)))
            self.gap.add(nn.Activation("relu"))

    def hybrid_forward(self, F, x):
        pool = self.gap(x)
        return F.contrib.BilinearResize2D(pool, **self._up_kwargs)

class _ASPP(nn.HybridBlock):
    def __init__(self, in_channels, atrous_rates, norm_layer, norm_kwargs,
                 height=60, width=60):
        super(_ASPP, self).__init__()
        out_channels = 256
        b0 = nn.HybridSequential()
        with b0.name_scope():
            b0.add(nn.Conv2D(in_channels=in_channels, channels=out_channels,
                             kernel_size=1, use_bias=False))
            b0.add(norm_layer(in_channels=out_channels, **({} if norm_kwargs is None else norm_kwargs)))
            b0.add(nn.Activation("relu"))

        rate1, rate2, rate3 = tuple(atrous_rates)
        b1 = _ASPPConv(in_channels, out_channels, rate1, norm_layer, norm_kwargs)
        b2 = _ASPPConv(in_channels, out_channels, rate2, norm_layer, norm_kwargs)
        b3 = _ASPPConv(in_channels, out_channels, rate3, norm_layer, norm_kwargs)
        b4 = _AsppPooling(in_channels, out_channels, norm_layer=norm_layer,
                          norm_kwargs=norm_kwargs, height=height, width=width)

        self.concurent = gluon.contrib.nn.HybridConcurrent(axis=1)
        with self.concurent.name_scope():
            self.concurent.add(b0)
            self.concurent.add(b1)
            self.concurent.add(b2)
            self.concurent.add(b3)
            self.concurent.add(b4)

        self.project = nn.HybridSequential()
        with self.project.name_scope():
            self.project.add(nn.Conv2D(in_channels=5*out_channels, channels=out_channels,
                                       kernel_size=1, use_bias=False))
            self.project.add(norm_layer(in_channels=out_channels,
                                        **({} if norm_kwargs is None else norm_kwargs)))
            self.project.add(nn.Activation("relu"))
            self.project.add(nn.Dropout(0.5))

    def hybrid_forward(self, F, x):
        return self.project(self.concurent(x))


def get_deeplab(dataset='pascal_voc', backbone='resnet50', pretrained=False,
            root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    r"""DeepLabV3
    Parameters
    ----------
    dataset : str, default pascal_voc
        The dataset that model pretrained on. (pascal_voc, ade20k)
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_fcn(dataset='pascal_voc', backbone='resnet50', pretrained=False)
    >>> print(model)
    """
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_aug': 'voc',
        'ade20k': 'ade',
        'coco': 'coco',
    }
    from ..data import datasets
    # infer number of classes
    model = DeepLabV3(datasets[dataset].NUM_CLASS, backbone=backbone, ctx=ctx, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_parameters(get_model_file('deeplab_%s_%s'%(backbone, acronyms[dataset]),
                                             tag=pretrained, root=root), ctx=ctx)
    return model

def get_deeplab_resnet101_coco(**kwargs):
    r"""DeepLabV3
    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_deeplab_resnet101_coco(pretrained=True)
    >>> print(model)
    """
    return get_deeplab('coco', 'resnet101', **kwargs)

def get_deeplab_resnet152_coco(**kwargs):
    r"""DeepLabV3
    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_deeplab_resnet152_coco(pretrained=True)
    >>> print(model)
    """
    return get_deeplab('coco', 'resnet152', **kwargs)

def get_deeplab_resnet101_voc(**kwargs):
    r"""DeepLabV3
    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_deeplab_resnet101_voc(pretrained=True)
    >>> print(model)
    """
    return get_deeplab('pascal_voc', 'resnet101', **kwargs)

def get_deeplab_resnet152_voc(**kwargs):
    r"""DeepLabV3
    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_deeplab_resnet152_voc(pretrained=True)
    >>> print(model)
    """
    return get_deeplab('pascal_voc', 'resnet152', **kwargs)

def get_deeplab_resnet50_ade(**kwargs):
    r"""DeepLabV3
    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_deeplab_resnet50_ade(pretrained=True)
    >>> print(model)
    """
    return get_deeplab('ade20k', 'resnet50', **kwargs)

def get_deeplab_resnet101_ade(**kwargs):
    r"""DeepLabV3
    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_deeplab_resnet101_ade(pretrained=True)
    >>> print(model)
    """
    return get_deeplab('ade20k', 'resnet101', **kwargs)
