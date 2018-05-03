"""ResNetV1bs, implemented in Gluon."""
# pylint: disable=arguments-differ,unused-argument,missing-docstring
from __future__ import division

from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm

__all__ = ['ResNetV1b', 'resnet18_v1b', 'resnet34_v1b',
           'resnet50_v1b', 'resnet101_v1b',
           'resnet152_v1b', 'BasicBlockV1b', 'BottleneckV1b']


class BasicBlockV1b(HybridBlock):
    """ResNetV1b BasicBlockV1b
    """
    expansion = 1
    def __init__(self, inplanes, planes, strides=1, dilation=1, downsample=None,
                 previous_dilation=1, norm_layer=None, **kwargs):
        super(BasicBlockV1b, self).__init__()
        self.conv1 = nn.Conv2D(in_channels=inplanes, channels=planes,
                               kernel_size=3, strides=strides,
                               padding=dilation, dilation=dilation, use_bias=False)
        self.bn1 = nn.BatchNorm(in_channels=planes)
        self.relu = nn.Activation('relu')
        self.conv2 = nn.Conv2D(in_channels=planes, channels=planes, kernel_size=3, strides=1,
                               padding=previous_dilation, dilation=previous_dilation,
                               use_bias=False)
        self.bn2 = nn.BatchNorm(in_channels=planes)
        self.downsample = downsample
        self.strides = strides

    def hybrid_forward(self, F, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class BottleneckV1b(HybridBlock):
    """ResNetV1b BottleneckV1b
    """
    # pylint: disable=unused-argument
    expansion = 4
    def __init__(self, inplanes, planes, strides=1, dilation=1,
                 downsample=None, previous_dilation=1, norm_layer=None, **kwargs):
        super(BottleneckV1b, self).__init__()
        self.conv1 = nn.Conv2D(in_channels=inplanes, channels=planes, kernel_size=1, use_bias=False)
        self.bn1 = nn.BatchNorm(in_channels=planes)
        self.conv2 = nn.Conv2D(
            in_channels=planes, channels=planes, kernel_size=3, strides=strides,
            padding=dilation, dilation=dilation, use_bias=False)
        self.bn2 = nn.BatchNorm(in_channels=planes)
        self.conv3 = nn.Conv2D(
            in_channels=planes, channels=planes * 4, kernel_size=1, use_bias=False)
        self.bn3 = nn.BatchNorm(in_channels=planes * 4)
        self.relu = nn.Activation('relu')
        self.downsample = downsample
        self.dilation = dilation
        self.strides = strides

    def hybrid_forward(self, F, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class ResNetV1b(HybridBlock):
    """ Pre-trained ResNetV1b Model, which preduces the strides of 8
    featuremaps at conv5.

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
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).


    Reference:

        - He, Kaiming, et al. "Deep residual learning for image recognition."
        Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """
    # pylint: disable=unused-variable
    def __init__(self, block, layers, classes=1000, dilated=False, norm_layer=BatchNorm,
                 **kwargs):
        self.inplanes = 64
        super(ResNetV1b, self).__init__()
        with self.name_scope():
            self.conv1 = nn.Conv2D(in_channels=3, channels=64, kernel_size=7, strides=2, padding=3,
                                   use_bias=False)
            self.bn1 = norm_layer(in_channels=64)
            self.relu = nn.Activation('relu')
            self.maxpool = nn.MaxPool2D(pool_size=3, strides=2, padding=1)
            self.layer1 = self._make_layer(1, block, 64, layers[0], norm_layer=norm_layer)
            self.layer2 = self._make_layer(2, block, 128, layers[1], strides=2,
                                           norm_layer=norm_layer)
            if dilated:
                self.layer3 = self._make_layer(3, block, 256, layers[2], strides=1, dilation=2,
                                               norm_layer=norm_layer)
                self.layer4 = self._make_layer(4, block, 512, layers[3], strides=1, dilation=4,
                                               norm_layer=norm_layer)
            else:
                self.layer3 = self._make_layer(3, block, 256, layers[2], strides=2,
                                               norm_layer=norm_layer)
                self.layer4 = self._make_layer(4, block, 512, layers[3], strides=2,
                                               norm_layer=norm_layer)
            self.avgpool = nn.AvgPool2D(7)
            self.flat = nn.Flatten()
            self.fc = nn.Dense(in_units=512 * block.expansion, units=classes)

    def _make_layer(self, stage_index, block, planes, blocks, strides=1, dilation=1,
                    norm_layer=None):
        downsample = None
        if strides != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.HybridSequential(prefix='down%d_'%stage_index)
            with downsample.name_scope():
                downsample.add(nn.Conv2D(in_channels=self.inplanes,
                                         channels=planes * block.expansion,
                                         kernel_size=1, strides=strides, use_bias=False))
                downsample.add(norm_layer(in_channels=planes * block.expansion))

        layers = nn.HybridSequential(prefix='layers%d_'%stage_index)
        with layers.name_scope():
            if dilation == 1 or dilation == 2:
                layers.add(block(self.inplanes, planes, strides, dilation=1,
                                 downsample=downsample, previous_dilation=dilation,
                                 norm_layer=norm_layer))
            elif dilation == 4:
                layers.add(block(self.inplanes, planes, strides, dilation=2,
                                 downsample=downsample, previous_dilation=dilation,
                                 norm_layer=norm_layer))
            else:
                raise RuntimeError("=> unknown dilation size: {}".format(dilation))

            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.add(block(self.inplanes, planes, dilation=dilation,
                                 previous_dilation=dilation, norm_layer=norm_layer))

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
        x = self.fc(x)

        return x


def resnet18_v1b(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    """Constructs a ResNetV1b-18 model.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    """
    model = ResNetV1b(BasicBlockV1b, [2, 2, 2, 2], **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_params(get_model_file('resnet%d_v%d'%(18, 0),
                                         root=root), ctx=ctx)
    return model


def resnet34_v1b(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    """Constructs a ResNetV1b-34 model.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
    """
    model = ResNetV1b(BasicBlockV1b, [3, 4, 6, 3], **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_params(get_model_file('resnet%d_v%d'%(34, 0),
                                         root=root), ctx=ctx)
    return model


def resnet50_v1b(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    """Constructs a ResNetV1b-50 model.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
    """
    model = ResNetV1b(BottleneckV1b, [3, 4, 6, 3], **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_params(get_model_file('resnet%d_v%d'%(50, 0),
                                         root=root), ctx=ctx)
    return model


def resnet101_v1b(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    """Constructs a ResNetV1b-101 model.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
    """
    model = ResNetV1b(BottleneckV1b, [3, 4, 23, 3], **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_params(get_model_file('resnet%d_v%d'%(101, 0),
                                         root=root), ctx=ctx)
    return model


def resnet152_v1b(pretrained=False, root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    """Constructs a ResNetV1b-152 model.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
    """
    model = ResNetV1b(BottleneckV1b, [3, 8, 36, 3], **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_params(get_model_file('resnet%d_v%d'%(152, 0),
                                         root=root), ctx=ctx)
    return model
