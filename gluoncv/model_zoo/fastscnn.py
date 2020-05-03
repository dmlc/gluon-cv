# pylint: disable=unused-argument, arguments-differ, unused-variable
"""Fast-SCNN, implemented in Gluon. Code adapted from lxtGH/Fast_Seg"""
__all__ = ['FastSCNN', 'get_fastscnn', 'get_fastscnn_citys']

from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn


class FastSCNN(HybridBlock):
    r"""Fast-SCNN: Fast Semantic Segmentation Network

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`).
    aux : bool
        Auxiliary loss.


    Reference:

        Rudra P K Poudel, et al. https://bmvc2019.org/wp-content/uploads/papers/0959-paper.pdf
        "Fast-SCNN: Fast Semantic Segmentation Network." *BMVC*, 2019

    """
    def __init__(self, nclass, aux=True, ctx=cpu(), pretrained_base=False,
                 height=None, width=None, base_size=2048, crop_size=1024, **kwargs):
        super(FastSCNN, self).__init__()

        height = height if height is not None else crop_size
        width = width if width is not None else crop_size
        self._up_kwargs = {'height': height, 'width': width}
        self.base_size = base_size
        self.crop_size = crop_size

        self.aux = aux
        with self.name_scope():
            self.learning_to_downsample = LearningToDownsample(32, 48, 64, **kwargs)
            self.learning_to_downsample.initialize(ctx=ctx)
            self.global_feature_extractor = GlobalFeatureExtractor(
                64, [64, 96, 128], 128, 6, [3, 3, 3], height=height//32, width=width//32, **kwargs)
            self.global_feature_extractor.initialize(ctx=ctx)
            self.feature_fusion = FeatureFusionModule(64, 128, 128,
                                                      height=height//8, width=width//8, **kwargs)
            self.feature_fusion.initialize(ctx=ctx)
            self.classifier = Classifer(128, nclass, **kwargs)
            self.classifier.initialize(ctx=ctx)

            if self.aux:
                self.auxlayer = _auxHead(in_channels=64, channels=64, nclass=nclass, **kwargs)
                self.auxlayer.initialize(ctx=ctx)
                self.auxlayer.collect_params().setattr('lr_mult', 10)

    def hybrid_forward(self, F, x):
        """hybrid forward for Fast SCNN"""
        higher_res_features = self.learning_to_downsample(x)
        x = self.global_feature_extractor(higher_res_features)
        x = self.feature_fusion(higher_res_features, x)
        x = self.classifier(x)
        x = F.contrib.BilinearResize2D(x, **self._up_kwargs)

        outputs = []
        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(higher_res_features)
            auxout = F.contrib.BilinearResize2D(auxout, **self._up_kwargs)
            outputs.append(auxout)
        return tuple(outputs)

    def demo(self, x):
        """fastscnn demo"""
        h, w = x.shape[2:]
        self._up_kwargs['height'] = h
        self._up_kwargs['width'] = w
        self.global_feature_extractor.ppm._up_kwargs = {'height': h // 32, 'width': w // 32}
        self.feature_fusion._up_kwargs = {'height': h // 8, 'width': w // 8}

        higher_res_features = self.learning_to_downsample(x)
        x = self.global_feature_extractor(higher_res_features)
        x = self.feature_fusion(higher_res_features, x)
        x = self.classifier(x)
        import mxnet.ndarray as F
        x = F.contrib.BilinearResize2D(x, **self._up_kwargs)
        return x

    def predict(self, x):
        """fastscnn predict"""
        return self.demo(x)

    def evaluate(self, x):
        """evaluating network with inputs and targets"""
        return self.forward(x)[0]


class FeatureFusionModule(HybridBlock):
    """FastSCNN feature fusion module"""
    def __init__(self, highter_in_channels, lower_in_channels, out_channels, height, width,
                 scale_factor=4, norm_layer=nn.BatchNorm, norm_kwargs=None, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.scale_factor = scale_factor
        self._up_kwargs = {'height': height, 'width': width}

        with self.name_scope():
            self.dwconv = _DWConv(lower_in_channels, out_channels,
                                  norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.conv_lower_res = nn.Conv2D(in_channels=out_channels, channels=out_channels,
                                            kernel_size=1)
            self.conv_higher_res = nn.Conv2D(in_channels=highter_in_channels,
                                             channels=out_channels, kernel_size=1)
            self.bn = norm_layer(in_channels=out_channels)
            self.relu = nn.Activation('relu')

    def hybrid_forward(self, F, higher_res_feature, lower_res_feature):
        lower_res_feature = F.contrib.BilinearResize2D(lower_res_feature, **self._up_kwargs)

        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.bn(self.conv_lower_res(lower_res_feature))
        higher_res_feature = self.bn(self.conv_higher_res(higher_res_feature))
        out = higher_res_feature + lower_res_feature
        return self.relu(out)


class Classifer(HybridBlock):
    """FastSCNN classifier"""
    def __init__(self, dw_channels, num_classes, stride=1,
                 norm_layer=nn.BatchNorm, norm_kwargs=None, **kwargs):
        super(Classifer, self).__init__()
        with self.name_scope():
            self.dsconv1 = _DSConv(dw_channels, dw_channels, stride,
                                   norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.dsconv2 = _DSConv(dw_channels, dw_channels, stride,
                                   norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.dp = nn.Dropout(0.1)
            self.conv = nn.Conv2D(in_channels=dw_channels, channels=num_classes, kernel_size=1)

    def hybrid_forward(self, F, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.dp(x)
        x = self.conv(x)
        return x


class _auxHead(HybridBlock):
    # pylint: disable=redefined-outer-name
    def __init__(self, in_channels=64, channels=64, nclass=19,
                 norm_layer=nn.BatchNorm, norm_kwargs=None, **kwargs):
        super(_auxHead, self).__init__()
        with self.name_scope():
            self.block = nn.HybridSequential()
            with self.block.name_scope():
                self.block.add(nn.Conv2D(in_channels=in_channels, channels=channels,
                                         kernel_size=3, padding=1, use_bias=False))
                self.block.add(norm_layer(in_channels=channels,
                                          **({} if norm_kwargs is None else norm_kwargs)))
                self.block.add(nn.Activation('relu'))
                self.block.add(nn.Dropout(0.1))
                self.block.add(nn.Conv2D(in_channels=channels, channels=nclass,
                                         kernel_size=1))

    def hybrid_forward(self, F, x):
        return self.block(x)


class _ConvBNReLU(HybridBlock):
    """Conv-BN-ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 norm_layer=nn.BatchNorm, norm_kwargs=None, **kwargs):
        super(_ConvBNReLU, self).__init__()
        with self.name_scope():
            self.block = nn.HybridSequential()
            self.block.add(nn.Conv2D(in_channels=in_channels, channels=out_channels,
                                     kernel_size=kernel_size,
                                     padding=padding, strides=stride, use_bias=False))
            self.block.add(norm_layer(in_channels=out_channels,
                                      **({} if norm_kwargs is None else norm_kwargs)))
            self.block.add(nn.Activation('relu'))

    def hybrid_forward(self, F, x):
        return self.block(x)


class _DSConv(HybridBlock):
    """Depthwise Separable Convolutions"""
    def __init__(self, dw_channels, out_channels, stride=1,
                 norm_layer=nn.BatchNorm, norm_kwargs=None, **kwargs):
        super(_DSConv, self).__init__()
        with self.name_scope():
            self.conv = nn.HybridSequential()
            self.conv.add(nn.Conv2D(in_channels=dw_channels, channels=dw_channels,
                                    kernel_size=3, strides=stride,
                                    padding=1, groups=dw_channels, use_bias=False))
            self.conv.add(norm_layer(in_channels=dw_channels,
                                     **({} if norm_kwargs is None else norm_kwargs)))
            self.conv.add(nn.Activation('relu'))
            self.conv.add(nn.Conv2D(in_channels=dw_channels, channels=out_channels,
                                    kernel_size=1, use_bias=False))
            self.conv.add(norm_layer(in_channels=out_channels,
                                     **({} if norm_kwargs is None else norm_kwargs)))
            self.conv.add(nn.Activation('relu'))

    def hybrid_forward(self, F, x):
        return self.conv(x)


class LearningToDownsample(HybridBlock):
    """Learning to downsample module"""
    def __init__(self, dw_channels1=32, dw_channels2=48, out_channels=64,
                 norm_layer=nn.BatchNorm, norm_kwargs=None, **kwargs):
        super(LearningToDownsample, self).__init__()
        with self.name_scope():
            self.conv = _ConvBNReLU(3, dw_channels1, 3, 2,
                                    norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)
            self.dsconv1 = _DSConv(dw_channels1, dw_channels2, 2,
                                   norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)
            self.dsconv2 = _DSConv(dw_channels2, out_channels, 2,
                                   norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x


class GlobalFeatureExtractor(HybridBlock):
    """Global feature extractor module"""
    def __init__(self, in_channels=64, block_channels=(64, 96, 128), out_channels=128, t=6,
                 num_blocks=(3, 3, 3), height=32, width=64,
                 norm_layer=nn.BatchNorm, norm_kwargs=None, **kwargs):
        super(GlobalFeatureExtractor, self).__init__()
        self.num_blocks = num_blocks
        with self.name_scope():
            self.bottleneck1 = self._make_layer(LinearBottleneck, in_channels, block_channels[0],
                                                num_blocks[0], t, 2, norm_layer, norm_kwargs)
            self.bottleneck2 = self._make_layer(LinearBottleneck, block_channels[0],
                                                block_channels[1], num_blocks[1], t, 2,
                                                norm_layer, norm_kwargs)
            self.bottleneck3 = self._make_layer(LinearBottleneck, block_channels[1],
                                                block_channels[2], num_blocks[2], t, 1,
                                                norm_layer, norm_kwargs)
            self.ppm = _FastPyramidPooling(block_channels[2], out_channels,
                                           height=height, width=width,
                                           norm_layer=norm_layer, norm_kwargs=norm_kwargs)

    def _make_layer(self, block, inplanes, planes, blocks, t=6, stride=1,
                    norm_layer=nn.BatchNorm, norm_kwargs=None):
        layers = nn.HybridSequential()
        with layers.name_scope():
            layers.add(block(inplanes, planes, t, stride, norm_layer, norm_kwargs))
            for i in range(1, blocks):
                layers.add(block(planes, planes, t, 1, norm_layer, norm_kwargs))
        return layers

    def hybrid_forward(self, F, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.ppm(x)
        return x


class LinearBottleneck(HybridBlock):
    """LinearBottleneck used in MobileNetV2"""
    def __init__(self, in_channels, out_channels, t=6, stride=2,
                 norm_layer=nn.BatchNorm, norm_kwargs=None):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        with self.name_scope():
            self.block = nn.HybridSequential()
            self.block.add(_ConvBNReLU(in_channels, in_channels * t, 1,
                                       norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            self.block.add(_DWConv(in_channels * t, in_channels * t, stride,
                                   norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            self.block.add(nn.Conv2D(in_channels=in_channels * t, channels=out_channels,
                                     kernel_size=1, use_bias=False))
            self.block.add(norm_layer(in_channels=out_channels,
                                      **({} if norm_kwargs is None else norm_kwargs)))

    def hybrid_forward(self, F, x):
        out = self.block(x)
        if self.use_shortcut:
            out = x + out
        return out


class _DWConv(HybridBlock):
    def __init__(self, dw_channels, out_channels, stride=1,
                 norm_layer=nn.BatchNorm, norm_kwargs=None):
        super(_DWConv, self).__init__()
        with self.name_scope():
            self.conv = nn.HybridSequential()
            self.conv.add(nn.Conv2D(in_channels=dw_channels, channels=out_channels, kernel_size=3,
                                    strides=stride, padding=1, groups=dw_channels, use_bias=False))
            self.conv.add(norm_layer(in_channels=out_channels,
                                     **({} if norm_kwargs is None else norm_kwargs)))
            self.conv.add(nn.Activation('relu'))

    def hybrid_forward(self, F, x):
        return self.conv(x)


def _PSP1x1Conv(in_channels, out_channels, norm_layer, norm_kwargs):
    block = nn.HybridSequential()
    with block.name_scope():
        block.add(nn.Conv2D(in_channels=in_channels, channels=out_channels,
                            kernel_size=1, use_bias=False))
        block.add(norm_layer(in_channels=out_channels,
                             **({} if norm_kwargs is None else norm_kwargs)))
        block.add(nn.Activation('relu'))
    return block


class _FastPyramidPooling(HybridBlock):
    def __init__(self, in_channels, ppm_out_channels, height=32, width=64,
                 norm_layer=nn.BatchNorm, norm_kwargs=None):
        super(_FastPyramidPooling, self).__init__()
        out_channels = int(in_channels/4)
        self._up_kwargs = {'height': height, 'width': width}
        with self.name_scope():
            self.conv1 = _PSP1x1Conv(in_channels, out_channels, norm_layer, norm_kwargs)
            self.conv2 = _PSP1x1Conv(in_channels, out_channels, norm_layer, norm_kwargs)
            self.conv3 = _PSP1x1Conv(in_channels, out_channels, norm_layer, norm_kwargs)
            self.conv4 = _PSP1x1Conv(in_channels, out_channels, norm_layer, norm_kwargs)
            self.out = _ConvBNReLU(in_channels * 2, ppm_out_channels, 1,
                                   norm_layer=norm_layer, norm_kwargs=norm_kwargs)

    def pool(self, F, x, size):
        return F.contrib.AdaptiveAvgPooling2D(x, output_size=size)

    def upsample(self, F, x):
        return F.contrib.BilinearResize2D(x, **self._up_kwargs)

    def hybrid_forward(self, F, x):
        feat1 = self.upsample(F, self.conv1(self.pool(F, x, 1)))
        feat2 = self.upsample(F, self.conv2(self.pool(F, x, 2)))
        feat3 = self.upsample(F, self.conv3(self.pool(F, x, 3)))
        feat4 = self.upsample(F, self.conv4(self.pool(F, x, 6)))
        x = F.concat(x, feat1, feat2, feat3, feat4, dim=1)
        x = self.out(x)
        return x

    def demo(self, x):
        """PyramidPooling for Fast SCNN """
        self._up_kwargs['height'] = x.shape[2]
        self._up_kwargs['width'] = x.shape[3]
        import mxnet.ndarray as F
        feat1 = self.upsample(F, self.conv1(self.pool(F, x, 1)))
        feat2 = self.upsample(F, self.conv2(self.pool(F, x, 2)))
        feat3 = self.upsample(F, self.conv3(self.pool(F, x, 3)))
        feat4 = self.upsample(F, self.conv4(self.pool(F, x, 6)))
        x = F.concat(x, feat1, feat2, feat3, feat4, dim=1)
        x = self.out(x)
        return x


def get_fastscnn(dataset='citys', ctx=cpu(0), pretrained=False,
                 root='~/.mxnet/models', **kwargs):
    r"""Fast-SCNN: Fast Semantic Segmentation Network
    Parameters
    ----------
    dataset : str, default cityscapes
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_fastscnn(dataset='citys')
    >>> print(model)
    """
    acronyms = {
        'citys': 'citys',
    }
    from ..data import datasets

    model = FastSCNN(datasets[dataset].NUM_CLASS, ctx=ctx, **kwargs)
    model.classes = datasets[dataset].classes

    if pretrained:
        from .model_store import get_model_file
        model.load_parameters(get_model_file('fastscnn_%s' % (acronyms[dataset]),
                                             tag=pretrained, root=root), ctx=ctx)
    return model


def get_fastscnn_citys(**kwargs):
    r"""Fast-SCNN: Fast Semantic Segmentation Network
        Parameters
        ----------
        dataset : str, default cityscapes
        ctx : Context, default CPU
            The context in which to load the pretrained weights.

        Examples
        --------
        >>> model = get_fastscnn_citys()
        >>> print(model)
        """
    return get_fastscnn('citys', **kwargs)
