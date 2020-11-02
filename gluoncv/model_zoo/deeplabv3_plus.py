"""Pyramid Scene Parsing Network"""
from mxnet.gluon import nn
from mxnet.context import cpu
from mxnet.gluon.nn import HybridBlock
from mxnet import gluon
from .fcn import _FCNHead
from .xception import get_xcetption
# pylint: disable-all

__all__ = ['DeepLabV3Plus', 'get_deeplab_plus', 'get_deeplab_plus_xception_coco']

class DeepLabV3Plus(HybridBlock):
    r"""DeepLabV3Plus

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'xception').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    aux : bool
        Auxiliary loss.


    Reference:

        Chen, Liang-Chieh, et al. "Encoder-Decoder with Atrous Separable Convolution for Semantic
        Image Segmentation."
    """
    def __init__(self, nclass, backbone='xception', aux=True, ctx=cpu(), pretrained_base=True,
                 height=None, width=None,base_size=576, crop_size=512, dilated=True, **kwargs):
        super(DeepLabV3Plus, self).__init__()
        self.aux = aux
        height = height if height is not None else crop_size
        width = width if width is not None else crop_size
        output_stride = 8 if dilated else 32
        with self.name_scope():
            pretrained = get_xcetption(pretrained=pretrained_base, output_stride=output_stride,
                                       ctx=ctx, **kwargs)
            kwargs.pop('root', None)
            # base network
            self.conv1 = pretrained.conv1
            self.bn1 = pretrained.bn1
            self.relu = pretrained.relu
            self.conv2 = pretrained.conv2
            self.bn2 = pretrained.bn2
            self.block1 = pretrained.block1
            self.block2 = pretrained.block2
            self.block3 = pretrained.block3
            # Middle flow
            self.midflow = pretrained.midflow
            # Exit flow
            self.block20 = pretrained.block20
            self.conv3 = pretrained.conv3
            self.bn3 = pretrained.bn3
            self.conv4 = pretrained.conv4
            self.bn4 = pretrained.bn4
            self.conv5 = pretrained.conv5
            self.bn5 = pretrained.bn5

            # deeplabv3 plus
            self.head = _DeepLabHead(nclass, height=height//4, width=width//4, **kwargs)
            self.head.initialize(ctx=ctx)
            self.head.collect_params().setattr('lr_mult', 10)
            if self.aux:
                self.auxlayer = _FCNHead(728, nclass, **kwargs)
                self.auxlayer.initialize(ctx=ctx)
                self.auxlayer.collect_params().setattr('lr_mult', 10)
        self._up_kwargs = {'height': height, 'width': width}
        self.base_size = base_size
        self.crop_size = crop_size

    def base_forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        # add relu here
        x = self.relu(x)
        low_level_feat = x

        x = self.block2(x)
        x = self.block3(x)

        # Middle flow
        x = self.midflow(x)
        mid_level_feat = x

        # Exit flow
        x = self.block20(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        return low_level_feat, mid_level_feat, x

    def hybrid_forward(self, F, x):
        c1, c3, c4 = self.base_forward(x)
        outputs = []
        x = self.head(c4, c1)
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
        self.head.aspp.concurent[-1]._up_kwargs['height'] = h// 8
        self.head.aspp.concurent[-1]._up_kwargs['width'] = w// 8
        pred = self.forward(x)
        if self.aux:
            pred = pred[0]
        return pred

    def evaluate(self, x):
        """evaluating network with inputs and targets"""
        return self.forward(x)[0]


class _DeepLabHead(HybridBlock):
    def __init__(self, nclass, c1_channels=128, norm_layer=nn.BatchNorm, norm_kwargs=None,
                 height=128, width=128, **kwargs):
        super(_DeepLabHead, self).__init__()
        self._up_kwargs = {'height': height, 'width': width}
        with self.name_scope():
            self.aspp = _ASPP(2048, [12, 24, 36], norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                              height=height//2, width=width//2, **kwargs)
            self.c1_block = nn.HybridSequential()
            self.c1_block.add(nn.Conv2D(in_channels=c1_channels, channels=48,
                                     kernel_size=3, padding=1, use_bias=False))
            self.c1_block.add(norm_layer(in_channels=48, **({} if norm_kwargs is None else norm_kwargs)))
            self.c1_block.add(nn.Activation('relu'))

            self.block = nn.HybridSequential()
            self.block.add(nn.Conv2D(in_channels=304, channels=256,
                                     kernel_size=3, padding=1, use_bias=False))
            self.block.add(norm_layer(in_channels=256, **({} if norm_kwargs is None else norm_kwargs)))
            self.block.add(nn.Activation('relu'))
            self.block.add(nn.Dropout(0.5))
            self.block.add(nn.Conv2D(in_channels=256, channels=256,
                                     kernel_size=3, padding=1, use_bias=False))
            self.block.add(norm_layer(in_channels=256, **({} if norm_kwargs is None else norm_kwargs)))
            self.block.add(nn.Activation('relu'))
            self.block.add(nn.Dropout(0.1))
            self.block.add(nn.Conv2D(in_channels=256, channels=nclass, kernel_size=1))

    def hybrid_forward(self, F, x, c1):
        c1 = self.c1_block(c1)
        x = self.aspp(x)
        x = F.contrib.BilinearResize2D(x, **self._up_kwargs)
        return self.block(F.concat(x, c1, dim=1))

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
                 height=64, width=64, **kwargs):
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
                 height=64, width=64):
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


def get_deeplab_plus(dataset='pascal_voc', backbone='xception', pretrained=False,
            root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    r"""DeepLabV3Plus
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
    >>> model = get_fcn(dataset='pascal_voc', backbone='xception', pretrained=False)
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
    if pretrained:
        kwargs['pretrained_base'] = False
    kwargs['root'] = root
    model = DeepLabV3Plus(datasets[dataset].NUM_CLASS, backbone=backbone, ctx=ctx, **kwargs)
    model.classes = datasets[dataset].CLASSES
    if pretrained:
        from .model_store import get_model_file
        model.load_parameters(get_model_file('deeplab_%s_%s'%(backbone, acronyms[dataset]),
                                             tag=pretrained, root=root), ctx=ctx)
    return model

def get_deeplab_plus_xception_coco(**kwargs):
    r"""DeepLabV3Plus
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
    >>> model = get_deeplab_plus_xception_coco(pretrained=True)
    >>> print(model)
    """
    return get_deeplab_plus('coco', 'xception', **kwargs)
