"""Fully Convolutional Network with Strdie of 8"""
from __future__ import division
from mxnet.gluon import nn
from mxnet.context import cpu
from mxnet.gluon.nn import HybridBlock
from .segbase import SegBaseModel
# pylint: disable=unused-argument,abstract-method,missing-docstring

__all__ = ['FCN', 'get_fcn', 'get_fcn_voc_resnet50', 'get_fcn_voc_resnet101',
           'get_fcn_ade_resnet50']

class FCN(SegBaseModel):
    r"""Fully Convolutional Networks for Semantic Segmentation

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;


    Reference:

        Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks
        for semantic segmentation." *CVPR*, 2015

    Examples
    --------
    >>> model = FCN(nclass=21, backbone='resnet50')
    >>> print(model)
    """
    # pylint: disable=arguments-differ
    def __init__(self, nclass, backbone='resnet50', norm_layer=nn.BatchNorm,
                 aux=True, ctx=cpu(), **kwargs):
        super(FCN, self).__init__(nclass, aux, backbone, ctx=ctx,
                                  norm_layer=norm_layer, **kwargs)
        with self.name_scope():
            self.head = _FCNHead(2048, nclass, norm_layer=norm_layer, **kwargs)
            self.head.initialize(ctx=ctx)
            self.head.collect_params().setattr('lr_mult', 10)
            if self.aux:
                self.auxlayer = _FCNHead(1024, nclass, norm_layer=norm_layer, **kwargs)
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
        else:
            return x


class _FCNHead(HybridBlock):
    # pylint: disable=redefined-outer-name
    def __init__(self, in_channels, channels, norm_layer, **kwargs):
        super(_FCNHead, self).__init__()
        with self.name_scope():
            self.block = nn.HybridSequential()
            inter_channels = in_channels // 4
            with self.block.name_scope():
                self.block.add(nn.Conv2D(in_channels=in_channels, channels=inter_channels,
                                         kernel_size=3, padding=1))
                self.block.add(norm_layer(in_channels=inter_channels))
                self.block.add(nn.Activation('relu'))
                self.block.add(nn.Dropout(0.1))
                self.block.add(nn.Conv2D(in_channels=inter_channels, channels=channels,
                                         kernel_size=1))

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, x):
        return self.block(x)


def get_fcn(dataset='pascal_voc', backbone='resnet50', pretrained=False,
            root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    r"""FCN model from the paper `"Fully Convolutional Network for semantic segmentation"
    <https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf>`_

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
    from ..data.ade20k.segmentation import ADE20KSegmentation
    acronyms = {
        'pascal_voc': 'voc',
        'ade20k': 'ade',
    }
    datasets = {
        'pascal_voc': VOCSegmentation,
        'ade20k': ADE20KSegmentation,
    }
    # infer number of classes
    model = FCN(datasets[dataset].NUM_CLASS, backbone=backbone, ctx=ctx, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_params(get_model_file('fcn_%s_%s'%(backbone, acronyms[dataset]),
                                         root=root), ctx=ctx)
    return model

def get_fcn_voc_resnet50(**kwargs):
    r"""FCN model with base network ResNet-50 pre-trained on Pascal VOC dataset
    from the paper `"Fully Convolutional Network for semantic segmentation"
    <https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf>`_

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
    >>> model = get_fcn_voc_resnet50(pretrained=True)
    >>> print(model)
    """
    return get_fcn('pascal_voc', 'resnet50', **kwargs)

def get_fcn_voc_resnet101(**kwargs):
    r"""FCN model with base network ResNet-101 pre-trained on Pascal VOC dataset
    from the paper `"Fully Convolutional Network for semantic segmentation"
    <https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf>`_

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
    >>> model = get_fcn_voc_resnet101(pretrained=True)
    >>> print(model)
    """
    return get_fcn('pascal_voc', 'resnet101', **kwargs)

def get_fcn_ade_resnet50(**kwargs):
    r"""FCN model with base network ResNet-50 pre-trained on ADE20K dataset
    from the paper `"Fully Convolutional Network for semantic segmentation"
    <https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf>`_

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
    return get_fcn('ade20k', 'resnet50', **kwargs)
