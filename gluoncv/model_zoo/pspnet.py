# pylint: disable=unused-argument
"""Pyramid Scene Parsing Network"""
from mxnet.gluon import nn
from mxnet.context import cpu
from mxnet.gluon.nn import HybridBlock
from .segbase import SegBaseModel
from .fcn import _FCNHead
# pylint: disable-all

class PSPNet(SegBaseModel):
    r"""Pyramid Scene Parsing Network

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

        Zhao, Hengshuang, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, and Jiaya Jia.
        "Pyramid scene parsing network." *CVPR*, 2017

    """
    def __init__(self, nclass, backbone='resnet50', norm_layer=nn.BatchNorm,
                 aux=True, ctx=cpu(), **kwargs):
        super(PSPNet, self).__init__(nclass, aux, backbone, ctx=ctx,
                                     norm_layer=norm_layer, **kwargs)
        with self.name_scope():
            self.head = _PSPHead(nclass, norm_layer=norm_layer, **kwargs)
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


def _PSP1x1Conv(in_channels, out_channels, norm_layer=None, **kwargs):
    block = nn.HybridSequential(prefix='')
    with block.name_scope():
        block.add(nn.Conv2D(in_channels=in_channels,
                            channels=out_channels, kernel_size=1))
        block.add(norm_layer(in_channels=out_channels))
        block.add(nn.Activation('relu'))
    return block


class _PyramidPooling(HybridBlock):
    def __init__(self, in_channels, **kwargs):
        super(_PyramidPooling, self).__init__()
        out_channels = int(in_channels/4)
        with self.name_scope():
            self.conv1 = _PSP1x1Conv(in_channels, out_channels, **kwargs)
            self.conv2 = _PSP1x1Conv(in_channels, out_channels, **kwargs)
            self.conv3 = _PSP1x1Conv(in_channels, out_channels, **kwargs)
            self.conv4 = _PSP1x1Conv(in_channels, out_channels, **kwargs)

    def pool(self, F, x, size):
        return F.contrib.AdaptiveAvgPooling2D(x, output_size=size)

    def upsample(self, F, x, h, w):
        return F.contrib.BilinearResize2D(x, height=h, width=w)

    def hybrid_forward(self, F, x):
        _, _, h, w = x.shape
        feat1 = self.upsample(F, self.conv1(self.pool(F, x, 1)), h, w)
        feat2 = self.upsample(F, self.conv2(self.pool(F, x, 2)), h, w)
        feat3 = self.upsample(F, self.conv3(self.pool(F, x, 3)), h, w)
        feat4 = self.upsample(F, self.conv4(self.pool(F, x, 4)), h, w)
        return F.concat(x, feat1, feat2, feat3, feat4, dim=1)


class _PSPHead(HybridBlock):
    def __init__(self, nclass, norm_layer=None, **kwargs):
        super(_PSPHead, self).__init__()
        self.psp = _PyramidPooling(2048, norm_layer=norm_layer, **kwargs)
        with self.name_scope():
            self.block = nn.HybridSequential(prefix='')
            self.block.add(nn.Conv2D(in_channels=4096, channels=512,
                                     kernel_size=3, padding=1))
            self.block.add(norm_layer(in_channels=512))
            self.block.add(nn.Activation('relu'))
            self.block.add(nn.Dropout(0.1))
            self.block.add(nn.Conv2D(in_channels=512, channels=nclass,
                                     kernel_size=1))

    def hybrid_forward(self, F, x):
        x = self.psp(x)
        return self.block(x)

def get_psp(dataset='pascal_voc', backbone='resnet50', pretrained=False,
            root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    r"""Pyramid Scene Parsing Network
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
    model = PSPNet(datasets[dataset].NUM_CLASS, backbone=backbone, ctx=ctx, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_params(get_model_file('psp_%s_%s'%(backbone, acronyms[dataset]),
                                         root=root), ctx=ctx)
    return model


def get_psp_ade_resnet50(**kwargs):
    r"""Pyramid Scene Parsing Network
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
    return get_psp('ade20k', 'resnet50', **kwargs)
