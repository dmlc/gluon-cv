# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import division

__all__ = ['get_pose_resnet']

import torch
import torch.nn as nn

from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn

def _conv3x3(out_planes, stride=1):
    return nn.Conv2D(out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(HybridBlock):
    expansion = 1

    def __init__(self, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = _conv3x3(planes, stride)
        self.bn1 = nn.BatchNorm()
        self.relu1 = nn.Activation('relu')
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm()
        self.relu2 = nn.Activation('relu')
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.relu2(out + residual)

        return out

class Bottleneck(HybridBlock):
    expansion = 4

    def __init__(self, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2D(planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm()
        self.relu1 = nn.Activation('relu')
        self.conv2 = nn.Conv2D(planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm()
        self.relu2 = nn.Activation('relu')
        self.conv3 = nn.Conv2D(planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm()
        self.relu3 = nn.Activation('relu')
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.relu3(out + residual)

        return out

class PoseResNet(HybridBlock):

    def __init__(self, block, layers,
                 num_deconv_layers, num_deconv_filters, num_deconv_kernels,
                 final_conv_kernel, deconv_with_bias, num_joints, **kwargs):
        self.inplanes = 64
        self.deconv_with_bias = deconv_with_bias

        super(PoseResNet, self).__init__()
        self.conv1 = nn.Conv2D(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm()
        self.relu = nn.Activation('relu')
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], 1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, 2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, 3)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, 4)

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            num_deconv_layers,
            num_deconv_filters,
            num_deconv_kernels,
        )

        self.final_layer = nn.Conv2D(
            out_channels=num_joints,
            kernel_size=final_conv_kernel,
            stride=1,
            padding=1 if final_conv_kernel == 3 else 0
        )

    def _make_layer(self, block, planes, blocks, stride=1, stage_index):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.HybridSequential(prefix='')
            downsample.add(nn.Conv2D(planes * block.expansion,
                                     kernel_size=1, stride=stride, bias=False))
            downsample.add(nn.BatchNorm())

        layer = nn.HybridSequential(prefix='stage%d_'%stage_index)
        with layer.name_scope():
            layers.add(block(planes, stride, downsample))
            self.inplanes = planes * block.expansion
            for i in range(1, blocks):
                layers.add(block(planes))

        return layer

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different from len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different from len(num_deconv_filters)'

        layer = nn.HybridSequential(prefix='')
        with layer.name_scope():
            for i in range(num_layers):
                kernel, padding, output_padding = \
                    self._get_deconv_cfg(num_kernels[i], i)

                planes = num_filters[i]
                layers.add(
                    nn.Conv2DTranspose(
                        out_channels=planes,
                        kernel_size=kernel,
                        stride=2,
                        padding=padding,
                        output_padding=output_padding,
                        use_bias=self.deconv_with_bias))
                layers.add(nn.BatchNorm())
                layers.add(nn.Activation('relu'))
                self.inplanes = planes

        return layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.deconv_layers(x)
        x = self.final_layer(x)

        return x

    def init_weights(self, pretrained=''):
        if os.path.isfile(pretrained):
            logger.info('=> init deconv weights from normal distribution')
            for name, m in self.deconv_layers.named_modules():
                if isinstance(m, nn.Conv2DTranspose):
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm):
                    logger.info('=> init {}.weight as 1'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            logger.info('=> init final conv weights from normal distribution')
            for m in self.final_layer.modules():
                if isinstance(m, nn.Conv2D):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    logger.info('=> init {}.weight as normal(0, 0.001)'.format(name))
                    logger.info('=> init {}.bias as 0'.format(name))
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.constant_(m.bias, 0)

            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))
            self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            logger.error('=> imagenet pretrained model dose not exist')
            logger.error('=> please download it first')
            raise ValueError('imagenet pretrained model does not exist')


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def get_pose_resnet(num_layers, pretrained=False, ctx=cpu(),
                    root='~/.mxnet/models', **kwargs):
    block_class, layers = resnet_spec[num_layers]

    net = PoseResNet(block_class, layers, **kwargs)

    if pretrained:
        from .model_store import get_model_file
        net.load_parameters(get_model_file('pose_resnet%d_v%d'%(num_layers, version),
                                           tag=pretrained, root=root), ctx=ctx)
        from ..data import ImageNet1kAttr
        attrib = ImageNet1kAttr()
        net.synset = attrib.synset
        net.classes = attrib.classes
        net.classes_long = attrib.classes_long

    return model
