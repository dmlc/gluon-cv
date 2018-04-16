"""Dilated_ResNetV1s, implemented in Gluon."""
# pylint: disable=arguments-differ,unused-argument
from __future__ import division

from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm

__all__ = ['DilatedResNetV1', 'dilated_resnet18', 'dilated_resnet34',
           'dilated_resnet50', 'dilated_resnet101',
           'dilated_resnet152', 'DilatedBasicBlockV1', 'DilatedBottleneckV1']


class DilatedBasicBlockV1(HybridBlock):
    """DilatedResNetV1 DilatedBasicBlockV1
    """
    expansion = 1
    def __init__(self, inplanes, planes, strides=1, dilation=1, downsample=None, first_dilation=1,
                 norm_layer=None, **kwargs):
        super(DilatedBasicBlockV1, self).__init__()
        self.conv1 = nn.Conv2D(in_channels=inplanes, channels=planes, kernel_size=3, strides=strides,
                               padding=dilation, dilation=dilation, use_bias=False)
        self.bn1 = nn.BatchNorm(in_channels=planes)
        self.relu = nn.Activation('relu')
        self.conv2 = nn.Conv2D(in_channels=planes, channels=planes, kernel_size=3, strides=1,
                               padding=first_dilation, dilation=first_dilation, use_bias=False)
        self.bn2 = nn.BatchNorm(in_channels=planes)
        self.downsample = downsample
        self.strides = strides

    def forward(self, x):
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


class DilatedBottleneckV1(HybridBlock):
    """DilatedResNetV1 DilatedBottleneckV1
    """
    # pylint: disable=unused-argument
    expansion = 4
    def __init__(self, inplanes, planes, strides=1, dilation=1,
                 downsample=None, first_dilation=1, norm_layer=None, **kwargs):
        super(DilatedBottleneckV1, self).__init__()
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

    def forward(self, x):
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


class DilatedResNetV1(HybridBlock):
    """Dilated Pre-trained DilatedResNetV1 Model, which preduces the strides of 8 featuremaps at conv5.

    Reference:

        - He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.

        - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
    """
    # pylint: disable=unused-variable
    def __init__(self, block, layers, num_classes=1000, norm_layer=None, **kwargs):
        self.inplanes = 64
        super(DilatedResNetV1, self).__init__()
        with self.name_scope():
            self.conv1 = nn.Conv2D(in_channels=3, channels=64, kernel_size=7, strides=2, padding=3,
                                   use_bias=False)
            self.bn1 = norm_layer(in_channels=64)
            self.relu = nn.Activation('relu')
            self.maxpool = nn.MaxPool2D(pool_size=3, strides=2, padding=1)
            self.layer1 = self._make_layer(1, block, 64, layers[0], norm_layer=norm_layer)
            self.layer2 = self._make_layer(2, block, 128, layers[1], strides=2, norm_layer=norm_layer)
            self.layer3 = self._make_layer(3, block, 256, layers[2], strides=1, dilation=2, norm_layer=norm_layer)
            self.layer4 = self._make_layer(4, block, 512, layers[3], strides=1, dilation=4, norm_layer=norm_layer)
            self.avgpool = nn.AvgPool2D(7)
            self.flat = nn.Flatten()
            self.fc = nn.Dense(in_units=512 * block.expansion, units=num_classes)

    def _make_layer(self, stage_index, block, planes, blocks, strides=1, dilation=1,
                    norm_layer=None):
        downsample = None
        if strides != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.HybridSequential(prefix='down%d_'%stage_index)
            with downsample.name_scope():
                downsample.add(nn.Conv2D(in_channels=self.inplanes, channels=planes * block.expansion,
                                         kernel_size=1, strides=strides, use_bias=False))
                downsample.add(norm_layer(in_channels=planes * block.expansion))

        layers = nn.HybridSequential(prefix='layers%d_'%stage_index)
        with layers.name_scope():
            if dilation == 1 or dilation == 2:
                layers.add(block(self.inplanes, planes, strides, dilation=1,
                                 downsample=downsample, first_dilation=dilation, norm_layer=norm_layer))
            elif dilation == 4:
                layers.add(block(self.inplanes, planes, strides, dilation=2,
                                 downsample=downsample, first_dilation=dilation, norm_layer=norm_layer))
            else:
                raise RuntimeError("=> unknown dilation size: {}".format(dilation, norm_layer=norm_layer))

            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.add(block(self.inplanes, planes, dilation=dilation, first_dilation=dilation,
                           norm_layer=norm_layer))

        return layers

    def forward(self, x):
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


def dilated_resnet18(pretrained=False, **kwargs):
    """Constructs a DilatedResNetV1-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DilatedResNetV1(DilatedBasicBlockV1, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['dilated_resnet18']))
    return model


def dilated_resnet34(pretrained=False, **kwargs):
    """Constructs a DilatedResNetV1-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DilatedResNetV1(DilatedBasicBlockV1, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['dilated_resnet34']))
    return model


def dilated_resnet50(pretrained=False, **kwargs):
    """Constructs a DilatedResNetV1-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DilatedResNetV1(DilatedBottleneckV1, [3, 4, 6, 3], **kwargs)
    if pretrained:
        print('loading pretrained weights')
        model.load_params('ResNet50.params')
    return model


def dilated_resnet101(pretrained=False, **kwargs):
    """Constructs a DilatedResNetV1-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DilatedResNetV1(DilatedBottleneckV1, [3, 4, 23, 3], **kwargs)
    if pretrained:
        print('loading pretrained weights')
        model.load_params('ResNet101.params')
    return model


def dilated_resnet152(pretrained=False, **kwargs):
    """Constructs a DilatedResNetV1-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DilatedResNetV1(DilatedBottleneckV1, [3, 8, 36, 3], **kwargs)
    if pretrained:
        print('loading pretrained weights')
        model.load_params('ResNet152.params')
    return model
