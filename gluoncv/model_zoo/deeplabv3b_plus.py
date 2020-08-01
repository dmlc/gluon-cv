"""DeepLabV3+ with wideresnet backbone for semantic segmentation"""
# pylint: disable=missing-docstring,arguments-differ,unused-argument
from mxnet.gluon import nn
from mxnet.context import cpu
from mxnet.gluon.nn import HybridBlock
from .wideresnet import wider_resnet38_a2

__all__ = ['DeepLabWV3Plus', 'get_deeplabv3b_plus', 'get_deeplab_v3b_plus_wideresnet_citys']

class DeepLabWV3Plus(HybridBlock):
    r"""DeepLabWV3Plus

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'wideresnet').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    aux : bool
        Auxiliary loss.

    Reference:

        Chen, Liang-Chieh, et al. "Encoder-Decoder with Atrous Separable Convolution for Semantic
        Image Segmentation.", https://arxiv.org/abs/1802.02611, ECCV 2018
    """
    def __init__(self, nclass, backbone='wideresnet', aux=False, ctx=cpu(), pretrained_base=True,
                 height=None, width=None, base_size=520, crop_size=480, dilated=True, **kwargs):
        super(DeepLabWV3Plus, self).__init__()

        height = height if height is not None else crop_size
        width = width if width is not None else crop_size
        self._up_kwargs = {'height': height, 'width': width}
        self.base_size = base_size
        self.crop_size = crop_size
        #print('self.crop_size', self.crop_size)
        kwargs.pop('root', None)

        with self.name_scope():
            pretrained = wider_resnet38_a2(classes=1000, dilation=True)
            pretrained.initialize(ctx=ctx)
            self.mod1 = pretrained.mod1
            self.mod2 = pretrained.mod2
            self.mod3 = pretrained.mod3
            self.mod4 = pretrained.mod4
            self.mod5 = pretrained.mod5
            self.mod6 = pretrained.mod6
            self.mod7 = pretrained.mod7
            self.pool2 = pretrained.pool2
            self.pool3 = pretrained.pool3
            del pretrained
            self.head = _DeepLabHead(nclass, height=height//2, width=width//2, **kwargs)
            self.head.initialize(ctx=ctx)

    def hybrid_forward(self, F, x):
        outputs = []
        x = self.mod1(x)
        m2 = self.mod2(self.pool2(x))
        x = self.mod3(self.pool3(m2))
        x = self.mod4(x)
        x = self.mod5(x)
        x = self.mod6(x)
        x = self.mod7(x)
        x = self.head(x, m2)
        x = F.contrib.BilinearResize2D(x, **self._up_kwargs)
        outputs.append(x)
        return tuple(outputs)

    def demo(self, x):
        return self.predict(x)

    def predict(self, x):
        h, w = x.shape[2:]
        self._up_kwargs['height'] = h
        self._up_kwargs['width'] = w
        x = self.mod1(x)
        m2 = self.mod2(self.pool2(x))
        x = self.mod3(self.pool3(m2))
        x = self.mod4(x)
        x = self.mod5(x)
        x = self.mod6(x)
        x = self.mod7(x)
        x = self.head.demo(x, m2)
        import mxnet.ndarray as F
        x = F.contrib.BilinearResize2D(x, **self._up_kwargs)
        return x

class _DeepLabHead(HybridBlock):
    def __init__(self, nclass, c1_channels=128, norm_layer=nn.BatchNorm, norm_kwargs=None,
                 height=240, width=240, **kwargs):
        super(_DeepLabHead, self).__init__()
        self._up_kwargs = {'height': height, 'width': width}
        with self.name_scope():
            self.aspp = _ASPP(in_channels=4096, atrous_rates=[12, 24, 36], norm_layer=norm_layer,
                              norm_kwargs=norm_kwargs, height=height//4, width=width//4, **kwargs)

            self.c1_block = nn.HybridSequential(prefix='bot_fine_')
            self.c1_block.add(nn.Conv2D(in_channels=c1_channels, channels=48,
                                        kernel_size=1, use_bias=False))

            self.block = nn.HybridSequential(prefix='final_')
            self.block.add(nn.Conv2D(in_channels=304, channels=256,
                                     kernel_size=3, padding=1, use_bias=False))
            self.block.add(norm_layer(in_channels=256,
                                      **({} if norm_kwargs is None else norm_kwargs)))
            self.block.add(nn.Activation('relu'))
            self.block.add(nn.Conv2D(in_channels=256, channels=256,
                                     kernel_size=3, padding=1, use_bias=False))
            self.block.add(norm_layer(in_channels=256,
                                      **({} if norm_kwargs is None else norm_kwargs)))
            self.block.add(nn.Activation('relu'))
            self.block.add(nn.Conv2D(in_channels=256, channels=nclass,
                                     kernel_size=1, use_bias=False))

    def hybrid_forward(self, F, x, c1):
        c1 = self.c1_block(c1)
        x = self.aspp(x)
        x = F.contrib.BilinearResize2D(x, **self._up_kwargs)
        return self.block(F.concat(c1, x, dim=1))

    def demo(self, x, c1):
        h, w = c1.shape[2:]
        self._up_kwargs['height'] = h
        self._up_kwargs['width'] = w
        c1 = self.c1_block(c1)
        x = self.aspp.demo(x)
        import mxnet.ndarray as F
        x = F.contrib.BilinearResize2D(x, **self._up_kwargs)
        return self.block(F.concat(c1, x, dim=1))

def _ASPPConv(in_channels, out_channels, atrous_rate, norm_layer, norm_kwargs):
    block = nn.HybridSequential()
    with block.name_scope():
        block.add(nn.Conv2D(in_channels=in_channels, channels=out_channels,
                            kernel_size=3, padding=atrous_rate,
                            dilation=atrous_rate, use_bias=False))
        block.add(norm_layer(in_channels=out_channels,
                             **({} if norm_kwargs is None else norm_kwargs)))
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

    def demo(self, x):
        h, w = x.shape[2:]
        self._up_kwargs['height'] = h
        self._up_kwargs['width'] = w
        pool = self.gap(x)
        import mxnet.ndarray as F
        return F.contrib.BilinearResize2D(pool, **self._up_kwargs)

class _ASPP(nn.HybridBlock):
    def __init__(self, in_channels, atrous_rates, norm_layer, norm_kwargs,
                 height=60, width=60):
        super(_ASPP, self).__init__()
        out_channels = 256
        self.b0 = nn.HybridSequential()
        self.b0.add(nn.Conv2D(in_channels=in_channels, channels=out_channels,
                              kernel_size=1, use_bias=False))
        self.b0.add(norm_layer(in_channels=out_channels,
                               **({} if norm_kwargs is None else norm_kwargs)))
        self.b0.add(nn.Activation("relu"))

        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b1 = _ASPPConv(in_channels, out_channels, rate1, norm_layer, norm_kwargs)
        self.b2 = _ASPPConv(in_channels, out_channels, rate2, norm_layer, norm_kwargs)
        self.b3 = _ASPPConv(in_channels, out_channels, rate3, norm_layer, norm_kwargs)
        self.b4 = _AsppPooling(in_channels, out_channels, norm_layer=norm_layer,
                               norm_kwargs=norm_kwargs, height=height, width=width)

        self.project = nn.HybridSequential(prefix='bot_aspp_')
        self.project.add(nn.Conv2D(in_channels=5*out_channels, channels=out_channels,
                                   kernel_size=1, use_bias=False))

    def hybrid_forward(self, F, x):
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat3 = self.b2(x)
        feat4 = self.b3(x)
        x = self.b4(x)
        x = F.concat(x, feat1, feat2, feat3, feat4, dim=1)
        return self.project(x)

    def demo(self, x):
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat3 = self.b2(x)
        feat4 = self.b3(x)
        x = self.b4.demo(x)
        import mxnet.ndarray as F
        x = F.concat(x, feat1, feat2, feat3, feat4, dim=1)
        return self.project(x)

def get_deeplabv3b_plus(dataset='citys', backbone='wideresnet', pretrained=False,
                        root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    r"""DeepLabWV3Plus
    Parameters
    ----------
    dataset : str, default pascal_voc
        The dataset that model pretrained on. (pascal_voc, ade20k, citys)
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_deeplabv3b_plus(dataset='citys', backbone='wideresnet', pretrained=False)
    >>> print(model)
    """
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_aug': 'voc',
        'ade20k': 'ade',
        'coco': 'coco',
        'citys': 'citys',
    }
    from ..data import datasets
    # infer number of classes
    if pretrained:
        kwargs['pretrained_base'] = False
    kwargs['root'] = root
    model = DeepLabWV3Plus(datasets[dataset].NUM_CLASS, backbone=backbone, ctx=ctx, **kwargs)
    model.classes = datasets[dataset].CLASSES
    if pretrained:
        from .model_store import get_model_file
        model.load_parameters(get_model_file('deeplab_v3b_plus_%s_%s'%(backbone, acronyms[dataset]),
                                             tag=pretrained, root=root), ctx=ctx)
    return model

def get_deeplab_v3b_plus_wideresnet_citys(**kwargs):
    r"""DeepLabWV3Plus
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
    >>> model = get_deeplab_v3b_plus_wideresnet_citys(pretrained=True)
    >>> print(model)
    """
    return get_deeplabv3b_plus('citys', 'wideresnet', **kwargs)
