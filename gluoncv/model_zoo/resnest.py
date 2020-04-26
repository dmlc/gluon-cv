"""ResNeSt implemented in Gluon."""
# pylint: disable=arguments-differ,unused-argument,missing-docstring,line-too-long
from __future__ import division

import math

from mxnet.context import cpu
from mxnet.gluon import nn
from mxnet.gluon.block import HybridBlock
from mxnet.gluon.nn import BatchNorm

from ..nn.dropblock import DropBlock
from ..nn.splat import SplitAttentionConv

__all__ = ['ResNeSt', 'Bottleneck', 'resnest14', 'resnest26', 'resnest50', 'resnest101',
           'resnest200', 'resnest269']


def _update_input_size(input_size, stride):
    sh, sw = (stride, stride) if isinstance(stride, int) else stride
    ih, iw = (input_size, input_size) if isinstance(input_size, int) else input_size
    oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
    input_size = (oh, ow)
    return input_size


class Bottleneck(HybridBlock):
    """ResNeSt Bottleneck
    """
    # pylint: disable=unused-argument
    expansion = 4

    def __init__(self, channels, cardinality=1, bottleneck_width=64, strides=1, dilation=1,
                 downsample=None, previous_dilation=1, norm_layer=None,
                 norm_kwargs=None, last_gamma=False,
                 dropblock_prob=0, input_size=None, use_splat=False,
                 radix=2, avd=False, avd_first=False, in_channels=None,
                 split_drop_ratio=0, **kwargs):
        super(Bottleneck, self).__init__()
        group_width = int(channels * (bottleneck_width / 64.)) * cardinality
        norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
        self.dropblock_prob = dropblock_prob
        self.use_splat = use_splat
        self.avd = avd and (strides > 1 or previous_dilation != dilation)
        self.avd_first = avd_first
        if self.dropblock_prob > 0:
            self.dropblock1 = DropBlock(dropblock_prob, 3, group_width, *input_size)
            if self.avd:
                if avd_first:
                    input_size = _update_input_size(input_size, strides)
                self.dropblock2 = DropBlock(dropblock_prob, 3, group_width, *input_size)
                if not avd_first:
                    input_size = _update_input_size(input_size, strides)
            else:
                input_size = _update_input_size(input_size, strides)
                self.dropblock2 = DropBlock(dropblock_prob, 3, group_width, *input_size)
            self.dropblock3 = DropBlock(dropblock_prob, 3, channels * 4, *input_size)
        self.conv1 = nn.Conv2D(channels=group_width, kernel_size=1,
                               use_bias=False, in_channels=in_channels)
        self.bn1 = norm_layer(in_channels=group_width, **norm_kwargs)
        self.relu1 = nn.Activation('relu')
        if self.use_splat:
            self.conv2 = SplitAttentionConv(channels=group_width, kernel_size=3,
                                            strides=1 if self.avd else strides,
                                            padding=dilation, dilation=dilation, groups=cardinality,
                                            use_bias=False, in_channels=group_width,
                                            norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                                            radix=radix, drop_ratio=split_drop_ratio,
                                            **kwargs)
        else:
            self.conv2 = nn.Conv2D(channels=group_width, kernel_size=3,
                                   strides=1 if self.avd else strides,
                                   padding=dilation, dilation=dilation, groups=cardinality,
                                   use_bias=False, in_channels=group_width, **kwargs)
            self.bn2 = norm_layer(in_channels=group_width, **norm_kwargs)
            self.relu2 = nn.Activation('relu')
        self.conv3 = nn.Conv2D(channels=channels * 4, kernel_size=1, use_bias=False,
                               in_channels=group_width)
        if not last_gamma:
            self.bn3 = norm_layer(in_channels=channels * 4, **norm_kwargs)
        else:
            self.bn3 = norm_layer(in_channels=channels * 4, gamma_initializer='zeros',
                                  **norm_kwargs)
        if self.avd:
            self.avd_layer = nn.AvgPool2D(3, strides, padding=1)
        self.relu3 = nn.Activation('relu')
        self.downsample = downsample
        self.dilation = dilation
        self.strides = strides

    def hybrid_forward(self, F, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        if self.dropblock_prob > 0:
            out = self.dropblock1(out)
        out = self.relu1(out)

        if self.avd and self.avd_first:
            out = self.avd_layer(out)

        if self.use_splat:
            out = self.conv2(out)
            if self.dropblock_prob > 0:
                out = self.dropblock2(out)
        else:
            out = self.conv2(out)
            out = self.bn2(out)
            if self.dropblock_prob > 0:
                out = self.dropblock2(out)
            out = self.relu2(out)

        if self.avd and not self.avd_first:
            out = self.avd_layer(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.dropblock_prob > 0:
            out = self.dropblock3(out)

        out = out + residual
        out = self.relu3(out)

        return out


class ResNeSt(HybridBlock):
    """ ResNeSt Model
    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    classes : int, default 1000
        Number of classification classes.
    dilated : bool, default False
        Applying dilation strategy to pretrained ResNet yielding a stride-8 model,
        typically used in Semantic Segmentation.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    last_gamma : bool, default False
        Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
    deep_stem : bool, default False
        Whether to replace the 7x7 conv1 with 3 3x3 convolution layers.
    avg_down : bool, default False
        Whether to use average pooling for projection skip connection between stages/downsample.
    final_drop : float, default 0.0
        Dropout ratio before the final classification layer.
    use_global_stats : bool, default False
        Whether forcing BatchNorm to use global statistics instead of minibatch statistics;
        optionally set to True if finetuning using ImageNet classification pretrained models.
    Reference:
        - He, Kaiming, et al. "Deep residual learning for image recognition."
        Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """

    # pylint: disable=unused-variable
    def __init__(self, block, layers, cardinality=1, bottleneck_width=64,
                 classes=1000, dilated=False, dilation=1, norm_layer=BatchNorm,
                 norm_kwargs=None, last_gamma=False, deep_stem=False, stem_width=32,
                 avg_down=False, final_drop=0.0, use_global_stats=False,
                 name_prefix='', dropblock_prob=0, input_size=224,
                 use_splat=False, radix=2, avd=False, avd_first=False, split_drop_ratio=0):
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.inplanes = stem_width * 2 if deep_stem else 64
        self.radix = radix
        self.split_drop_ratio = split_drop_ratio
        self.avd_first = avd_first
        super(ResNeSt, self).__init__(prefix=name_prefix)
        norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
        if use_global_stats:
            norm_kwargs['use_global_stats'] = True
        self.norm_kwargs = norm_kwargs
        with self.name_scope():
            if not deep_stem:
                self.conv1 = nn.Conv2D(channels=64, kernel_size=7, strides=2,
                                       padding=3, use_bias=False, in_channels=3)
            else:
                self.conv1 = nn.HybridSequential(prefix='conv1')
                self.conv1.add(nn.Conv2D(channels=stem_width, kernel_size=3, strides=2,
                                         padding=1, use_bias=False, in_channels=3))
                self.conv1.add(norm_layer(in_channels=stem_width, **norm_kwargs))
                self.conv1.add(nn.Activation('relu'))
                self.conv1.add(nn.Conv2D(channels=stem_width, kernel_size=3, strides=1,
                                         padding=1, use_bias=False, in_channels=stem_width))
                self.conv1.add(norm_layer(in_channels=stem_width, **norm_kwargs))
                self.conv1.add(nn.Activation('relu'))
                self.conv1.add(nn.Conv2D(channels=stem_width * 2, kernel_size=3, strides=1,
                                         padding=1, use_bias=False, in_channels=stem_width))
            input_size = _update_input_size(input_size, 2)
            self.bn1 = norm_layer(in_channels=64 if not deep_stem else stem_width * 2,
                                  **norm_kwargs)
            self.relu = nn.Activation('relu')
            self.maxpool = nn.MaxPool2D(pool_size=3, strides=2, padding=1)
            input_size = _update_input_size(input_size, 2)
            self.layer1 = self._make_layer(1, block, 64, layers[0], avg_down=avg_down,
                                           norm_layer=norm_layer, last_gamma=last_gamma,
                                           use_splat=use_splat, avd=avd)
            self.layer2 = self._make_layer(2, block, 128, layers[1], strides=2, avg_down=avg_down,
                                           norm_layer=norm_layer, last_gamma=last_gamma,
                                           use_splat=use_splat, avd=avd)
            input_size = _update_input_size(input_size, 2)
            if dilated or dilation == 4:
                self.layer3 = self._make_layer(3, block, 256, layers[2], strides=1, dilation=2,
                                               avg_down=avg_down, norm_layer=norm_layer,
                                               last_gamma=last_gamma, dropblock_prob=dropblock_prob,
                                               input_size=input_size, use_splat=use_splat, avd=avd)
                self.layer4 = self._make_layer(4, block, 512, layers[3], strides=1, dilation=4,
                                               pre_dilation=2,
                                               avg_down=avg_down, norm_layer=norm_layer,
                                               last_gamma=last_gamma, dropblock_prob=dropblock_prob,
                                               input_size=input_size, use_splat=use_splat, avd=avd)
            elif dilation == 3:
                # special
                self.layer3 = self._make_layer(3, block, 256, layers[2], strides=1, dilation=2,
                                               avg_down=avg_down, norm_layer=norm_layer,
                                               last_gamma=last_gamma, dropblock_prob=dropblock_prob,
                                               input_size=input_size, use_splat=use_splat, avd=avd)
                self.layer4 = self._make_layer(4, block, 512, layers[3], strides=2, dilation=2,
                                               pre_dilation=2,
                                               avg_down=avg_down, norm_layer=norm_layer,
                                               last_gamma=last_gamma, dropblock_prob=dropblock_prob,
                                               input_size=input_size, use_splat=use_splat, avd=avd)
            elif dilation == 2:
                self.layer3 = self._make_layer(3, block, 256, layers[2], strides=2,
                                               avg_down=avg_down, norm_layer=norm_layer,
                                               last_gamma=last_gamma, dropblock_prob=dropblock_prob,
                                               input_size=input_size, use_splat=use_splat, avd=avd)
                self.layer4 = self._make_layer(4, block, 512, layers[3], strides=1, dilation=2,
                                               avg_down=avg_down, norm_layer=norm_layer,
                                               last_gamma=last_gamma, dropblock_prob=dropblock_prob,
                                               input_size=input_size, use_splat=use_splat, avd=avd)
            else:
                self.layer3 = self._make_layer(3, block, 256, layers[2], strides=2,
                                               avg_down=avg_down, norm_layer=norm_layer,
                                               last_gamma=last_gamma, dropblock_prob=dropblock_prob,
                                               input_size=input_size, use_splat=use_splat, avd=avd)
                input_size = _update_input_size(input_size, 2)
                self.layer4 = self._make_layer(4, block, 512, layers[3], strides=2,
                                               avg_down=avg_down, norm_layer=norm_layer,
                                               last_gamma=last_gamma, dropblock_prob=dropblock_prob,
                                               input_size=input_size, use_splat=use_splat, avd=avd)
                input_size = _update_input_size(input_size, 2)
            self.avgpool = nn.GlobalAvgPool2D()
            self.flat = nn.Flatten()
            self.drop = None
            if final_drop > 0.0:
                self.drop = nn.Dropout(final_drop)
            self.fc = nn.Dense(in_units=512 * block.expansion, units=classes)

    def _make_layer(self, stage_index, block, planes, blocks, strides=1, dilation=1, pre_dilation=1,
                    avg_down=False, norm_layer=None, last_gamma=False, dropblock_prob=0,
                    input_size=224, use_splat=False, avd=False):
        downsample = None
        if strides != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.HybridSequential(prefix='down%d_' % stage_index)
            with downsample.name_scope():
                if avg_down:
                    if pre_dilation == 1:
                        downsample.add(nn.AvgPool2D(pool_size=strides, strides=strides,
                                                    ceil_mode=True, count_include_pad=False))
                    elif strides == 1:
                        downsample.add(nn.AvgPool2D(pool_size=1, strides=1,
                                                    ceil_mode=True, count_include_pad=False))
                    else:
                        downsample.add(
                            nn.AvgPool2D(pool_size=pre_dilation * strides, strides=strides,
                                         padding=1, ceil_mode=True, count_include_pad=False))
                    downsample.add(nn.Conv2D(channels=planes * block.expansion, kernel_size=1,
                                             strides=1, use_bias=False, in_channels=self.inplanes))
                    downsample.add(norm_layer(in_channels=planes * block.expansion,
                                              **self.norm_kwargs))
                else:
                    downsample.add(nn.Conv2D(channels=planes * block.expansion,
                                             kernel_size=1, strides=strides, use_bias=False,
                                             in_channels=self.inplanes))
                    downsample.add(norm_layer(in_channels=planes * block.expansion,
                                              **self.norm_kwargs))

        layers = nn.HybridSequential(prefix='layers%d_' % stage_index)
        with layers.name_scope():
            if dilation in (1, 2):
                layers.add(block(planes, cardinality=self.cardinality,
                                 bottleneck_width=self.bottleneck_width,
                                 strides=strides, dilation=pre_dilation,
                                 downsample=downsample, previous_dilation=dilation,
                                 norm_layer=norm_layer, norm_kwargs=self.norm_kwargs,
                                 last_gamma=last_gamma, dropblock_prob=dropblock_prob,
                                 input_size=input_size, use_splat=use_splat, avd=avd,
                                 avd_first=self.avd_first, radix=self.radix,
                                 in_channels=self.inplanes, split_drop_ratio=self.split_drop_ratio))
            elif dilation == 4:
                layers.add(block(planes, cardinality=self.cardinality,
                                 bottleneck_width=self.bottleneck_width,
                                 strides=strides, dilation=pre_dilation,
                                 downsample=downsample, previous_dilation=dilation,
                                 norm_layer=norm_layer, norm_kwargs=self.norm_kwargs,
                                 last_gamma=last_gamma, dropblock_prob=dropblock_prob,
                                 input_size=input_size, use_splat=use_splat, avd=avd,
                                 avd_first=self.avd_first, radix=self.radix,
                                 in_channels=self.inplanes, split_drop_ratio=self.split_drop_ratio))
            else:
                raise RuntimeError("=> unknown dilation size: {}".format(dilation))

            input_size = _update_input_size(input_size, strides)
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.add(block(planes, cardinality=self.cardinality,
                                 bottleneck_width=self.bottleneck_width, dilation=dilation,
                                 previous_dilation=dilation, norm_layer=norm_layer,
                                 norm_kwargs=self.norm_kwargs, last_gamma=last_gamma,
                                 dropblock_prob=dropblock_prob, input_size=input_size,
                                 use_splat=use_splat, avd=avd, avd_first=self.avd_first,
                                 radix=self.radix, in_channels=self.inplanes,
                                 split_drop_ratio=self.split_drop_ratio))

        return layers

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flat(x)
        if self.drop is not None:
            x = self.drop(x)
        x = self.fc(x)

        return x


def resnest14(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    """Constructs a ResNeSt-14 model.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    dilated: bool, default False
        Whether to apply dilation strategy to ResNeSt, yielding a stride 8 model.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`).
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    model = ResNeSt(Bottleneck, [1, 1, 1, 1],
                    radix=2, cardinality=1, bottleneck_width=64,
                    deep_stem=True, avg_down=True,
                    avd=True, avd_first=False,
                    use_splat=True, dropblock_prob=0.0,
                    name_prefix='resnest_', **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_parameters(get_model_file('resnest14', root=root), ctx=ctx)
        from ..data import ImageNet1kAttr
        attrib = ImageNet1kAttr()
        model.synset = attrib.synset
        model.classes = attrib.classes
        model.classes_long = attrib.classes_long
    return model


def resnest26(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    """Constructs a ResNeSt-26 model.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    dilated: bool, default False
        Whether to apply dilation strategy to ResNeSt, yielding a stride 8 model.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`).
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    model = ResNeSt(Bottleneck, [2, 2, 2, 2],
                    radix=2, cardinality=1, bottleneck_width=64,
                    deep_stem=True, avg_down=True,
                    avd=True, avd_first=False,
                    use_splat=True, dropblock_prob=0.1,
                    name_prefix='resnest_', **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_parameters(get_model_file('resnest26', root=root), ctx=ctx)
        from ..data import ImageNet1kAttr
        attrib = ImageNet1kAttr()
        model.synset = attrib.synset
        model.classes = attrib.classes
        model.classes_long = attrib.classes_long
    return model


def resnest50(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    """Constructs a ResNeSt-50 model.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    dilated: bool, default False
        Whether to apply dilation strategy to ResNeSt, yielding a stride 8 model.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`).
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    model = ResNeSt(Bottleneck, [3, 4, 6, 3],
                    radix=2, cardinality=1, bottleneck_width=64,
                    deep_stem=True, avg_down=True,
                    avd=True, avd_first=False,
                    use_splat=True, dropblock_prob=0.1,
                    name_prefix='resnest_', **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_parameters(get_model_file('resnest50', root=root), ctx=ctx)
        from ..data import ImageNet1kAttr
        attrib = ImageNet1kAttr()
        model.synset = attrib.synset
        model.classes = attrib.classes
        model.classes_long = attrib.classes_long
    return model


def resnest101(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    """Constructs a ResNeSt-101 model.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    dilated: bool, default False
        Whether to apply dilation strategy to ResNeSt, yielding a stride 8 model.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`).
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    model = ResNeSt(Bottleneck, [3, 4, 23, 3],
                    radix=2, cardinality=1, bottleneck_width=64,
                    deep_stem=True, avg_down=True, stem_width=64,
                    avd=True, avd_first=False, use_splat=True, dropblock_prob=0.1,
                    name_prefix='resnest_', **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_parameters(get_model_file('resnest101', root=root), ctx=ctx)
        from ..data import ImageNet1kAttr
        attrib = ImageNet1kAttr()
        model.synset = attrib.synset
        model.classes = attrib.classes
        model.classes_long = attrib.classes_long
    return model


def resnest200(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    """Constructs a ResNeSt-200 model.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    dilated: bool, default False
        Whether to apply dilation strategy to ResNeSt, yielding a stride 8 model.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`).
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    model = ResNeSt(Bottleneck, [3, 24, 36, 3], deep_stem=True, avg_down=True, stem_width=64,
                    avd=True, use_splat=True, dropblock_prob=0.1, final_drop=0.2,
                    name_prefix='resnest_', **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_parameters(get_model_file('resnest200', root=root), ctx=ctx)
        from ..data import ImageNet1kAttr
        attrib = ImageNet1kAttr()
        model.synset = attrib.synset
        model.classes = attrib.classes
        model.classes_long = attrib.classes_long
    return model


def resnest269(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    """Constructs a ResNeSt-269 model.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    dilated: bool, default False
        Whether to apply dilation strategy to ResNeSt, yielding a stride 8 model.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`).
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    model = ResNeSt(Bottleneck, [3, 30, 48, 8], deep_stem=True, avg_down=True, stem_width=64,
                    avd=True, use_splat=True, dropblock_prob=0.1, final_drop=0.2,
                    name_prefix='resnest_', **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_parameters(get_model_file('resnest269', root=root), ctx=ctx)
        from ..data import ImageNet1kAttr
        attrib = ImageNet1kAttr()
        model.synset = attrib.synset
        model.classes = attrib.classes
        model.classes_long = attrib.classes_long
    return model
