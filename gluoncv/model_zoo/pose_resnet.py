# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

# coding: utf-8
# pylint: disable=missing-docstring

from __future__ import division

__all__ = ['get_pose_resnet', 'pose_resnet18', 'pose_resnet34',
           'pose_resnet50', 'pose_resnet101', 'pose_resnet152']

from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn

class BasicBlock(HybridBlock):
    expansion = 1

    def __init__(self, planes, strides=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2D(planes, kernel_size=3, strides=strides,
                               padding=1, use_bias=False)
        self.bn1 = nn.BatchNorm()
        self.relu1 = nn.Activation('relu')
        self.conv2 = nn.Conv2D(planes, kernel_size=3, strides=1,
                               padding=1, use_bias=False)
        self.bn2 = nn.BatchNorm()
        self.relu2 = nn.Activation('relu')
        self.downsample = downsample
        self.strides = strides

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

    def __init__(self, planes, strides=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2D(planes, kernel_size=1, use_bias=False)
        self.bn1 = nn.BatchNorm()
        self.relu1 = nn.Activation('relu')
        self.conv2 = nn.Conv2D(planes, kernel_size=3, strides=strides,
                               padding=1, use_bias=False)
        self.bn2 = nn.BatchNorm()
        self.relu2 = nn.Activation('relu')
        self.conv3 = nn.Conv2D(planes * self.expansion, kernel_size=1, use_bias=False)
        self.bn3 = nn.BatchNorm()
        self.relu3 = nn.Activation('relu')
        self.downsample = downsample
        self.strides = strides

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

    def __init__(self, block, layers, num_joints,
                 num_deconv_layers=3,
                 num_deconv_filters=(256, 256, 256),
                 num_deconv_kernels=(4, 4, 4),
                 final_conv_kernel=1, deconv_with_bias=False, **kwargs):
        self.inplanes = 64
        self.deconv_with_bias = deconv_with_bias

        super(PoseResNet, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(64, kernel_size=7, strides=2, padding=3,
                               use_bias=False)
        self.bn1 = nn.BatchNorm()
        self.relu = nn.Activation('relu')
        self.maxpool = nn.MaxPool2D(pool_size=3, strides=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stage_index=1)
        self.layer2 = self._make_layer(block, 128, layers[1],
                                       stage_index=2, strides=2)
        self.layer3 = self._make_layer(block, 256, layers[2],
                                       stage_index=3, strides=2)
        self.layer4 = self._make_layer(block, 512, layers[3],
                                       stage_index=4, strides=2)

        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            num_deconv_layers,
            num_deconv_filters,
            num_deconv_kernels,
        )

        self.final_layer = nn.Conv2D(
            channels=num_joints,
            kernel_size=final_conv_kernel,
            strides=1,
            padding=1 if final_conv_kernel == 3 else 0
        )

    def _make_layer(self, block, planes, blocks, stage_index, strides=1):
        downsample = None
        if strides != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.HybridSequential(prefix='')
            downsample.add(nn.Conv2D(planes * block.expansion,
                                     kernel_size=1, strides=strides, use_bias=False))
            downsample.add(nn.BatchNorm())

        layer = nn.HybridSequential(prefix='stage%d_'%stage_index)
        with layer.name_scope():
            layer.add(block(planes, strides, downsample))
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layer.add(block(planes))

        return layer

    def _get_deconv_cfg(self, deconv_kernel):
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
                    self._get_deconv_cfg(num_kernels[i])

                planes = num_filters[i]
                layer.add(
                    nn.Conv2DTranspose(
                        channels=planes,
                        kernel_size=kernel,
                        strides=2,
                        padding=padding,
                        output_padding=output_padding,
                        use_bias=self.deconv_with_bias))
                layer.add(nn.BatchNorm())
                layer.add(nn.Activation('relu'))
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
        net.load_parameters(get_model_file('pose_resnet%d'%(num_layers),
                                           tag=pretrained, root=root), ctx=ctx)

    return net

def pose_resnet18(**kwargs):
    r"""ResNet-18 model from `"Simple Baselines for Human Pose Estimation and Tracking"
    <https://arxiv.org/abs/1804.06208>`_ paper.
    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_pose_resnet(18, **kwargs)

def pose_resnet34(**kwargs):
    r"""ResNet-18 model from `"Simple Baselines for Human Pose Estimation and Tracking"
    <https://arxiv.org/abs/1804.06208>`_ paper.
    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_pose_resnet(34, **kwargs)

def pose_resnet50(**kwargs):
    r"""ResNet-18 model from `"Simple Baselines for Human Pose Estimation and Tracking"
    <https://arxiv.org/abs/1804.06208>`_ paper.
    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_pose_resnet(50, **kwargs)

def pose_resnet101(**kwargs):
    r"""ResNet-18 model from `"Simple Baselines for Human Pose Estimation and Tracking"
    <https://arxiv.org/abs/1804.06208>`_ paper.
    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_pose_resnet(101, **kwargs)

def pose_resnet152(**kwargs):
    r"""ResNet-18 model from `"Simple Baselines for Human Pose Estimation and Tracking"
    <https://arxiv.org/abs/1804.06208>`_ paper.
    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    """
    return get_pose_resnet(152, **kwargs)
