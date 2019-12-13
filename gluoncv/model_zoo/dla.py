"""Deep Layer Aggregation networks, implemented in Gluon."""
# pylint: disable=arguments-differ,unused-argument,missing-docstring
from __future__ import division

import os

import mxnet as mx
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm

__all__ = ['DLA', 'get_dla', 'dla34']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2D(channels=out_planes, kernel_size=3, strides=stride,
                     padding=1, use_bias=False, in_channels=in_planes)


class BasicBlock(HybridBlock):
    def __init__(self, inplanes, planes, stride=1, dilation=1,
                 norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(BasicBlock, self).__init__(**kwargs)
        if norm_kwargs is None:
            norm_kwargs = {}
        with self.name_scope():
            self.conv1 = nn.Conv2D(in_channels=inplanes, channels=planes, kernel_size=3,
                                   strides=stride, padding=dilation,
                                   use_bias=False, dilation=dilation)
            self.bn1 = norm_layer(in_channels=planes, **norm_kwargs)
            self.relu = nn.Activation('relu')
            self.conv2 = nn.Conv2D(in_channels=planes, channels=planes, kernel_size=3,
                                   strides=1, padding=dilation,
                                   use_bias=False, dilation=dilation)
            self.bn2 = norm_layer(in_channels=planes, **norm_kwargs)
            self.stride = stride

    def hybrid_forward(self, F, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck(HybridBlock):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, dilation=1,
                 norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        if norm_kwargs is None:
            norm_kwargs = {}
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        with self.name_scope():
            self.conv1 = nn.Conv2D(in_channels=inplanes, channels=bottle_planes,
                                   kernel_size=1, use_bias=False)
            self.bn1 = norm_layer(in_channels=bottle_planes, **norm_kwargs)
            self.conv2 = nn.Conv2D(in_channels=bottle_planes, channels=bottle_planes, kernel_size=3,
                                   strides=stride, padding=dilation,
                                   use_bias=False, dilation=dilation)
            self.bn2 = norm_layer(in_channels=bottle_planes, **norm_kwargs)
            self.conv3 = nn.Conv2D(in_channels=bottle_planes, channels=planes,
                                   kernel_size=1, use_bias=False)
            self.bn3 = norm_layer(**norm_kwargs)
            self.relu = nn.Activation('relu')
            self.stride = stride

    def hybrid_forward(self, F, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = out + residual
        out = self.relu(out)

        return out


class BottleneckX(HybridBlock):
    expansion = 2
    cardinality = 32

    def __init__(self, inplanes, planes, stride=1, dilation=1,
                 norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(BottleneckX, self).__init__(**kwargs)
        if norm_kwargs is None:
            norm_kwargs = {}
        cardinality = BottleneckX.cardinality
        bottle_planes = planes * cardinality // 32
        with self.name_scope():
            self.conv1 = nn.Conv2D(in_channels=inplanes, channels=bottle_planes,
                                   kernel_size=1, use_bias=False)
            self.bn1 = norm_layer(in_channels=bottle_planes, **norm_kwargs)
            self.conv2 = nn.Conv2D(in_channels=bottle_planes, channels=bottle_planes, kernel_size=3,
                                   strides=stride, padding=dilation, use_bias=False,
                                   dilation=dilation, groups=cardinality)
            self.bn2 = norm_layer(in_channels=bottle_planes, **norm_kwargs)
            self.conv3 = nn.Conv2D(in_channels=bottle_planes, channels=planes,
                                   kernel_size=1, use_bias=False)
            self.bn3 = norm_layer(**norm_kwargs)
            self.relu = nn.Activation('relu')
            self.stride = stride

    def hybrid_forward(self, F, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = out + residual
        out = self.relu(out)

        return out


class Root(HybridBlock):
    def __init__(self, in_channels, out_channels, kernel_size, residual,
                 norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(Root, self).__init__(**kwargs)
        if norm_kwargs is None:
            norm_kwargs = {}
        with self.name_scope():
            self.conv = nn.Conv2D(
                in_channels=in_channels, channels=out_channels, kernel_size=1,
                strides=1, use_bias=False, padding=(kernel_size - 1) // 2)
            self.bn = norm_layer(in_channels=out_channels, **norm_kwargs)
            self.relu = nn.Activation('relu')
            self.residual = residual

    def hybrid_forward(self, F, *x):
        children = x
        x = self.conv(F.concat(*x, dim=1))
        x = self.bn(x)
        if self.residual:
            x = x + children[0]
        x = self.relu(x)

        return x


class Tree(HybridBlock):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False, norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(Tree, self).__init__(**kwargs)
        if norm_kwargs is None:
            norm_kwargs = {}
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim = root_dim + in_channels
        with self.name_scope():
            self.downsample = nn.HybridSequential()
            self.project = nn.HybridSequential()
            if levels == 1:
                self.tree1 = block(in_channels, out_channels, stride,
                                   dilation=dilation, norm_layer=norm_layer,
                                   norm_kwargs=norm_kwargs, prefix='block_tree_1_')
                self.tree2 = block(out_channels, out_channels, 1,
                                   dilation=dilation, norm_layer=norm_layer,
                                   norm_kwargs=norm_kwargs, prefix='block_tree_2_')
                if in_channels != out_channels:
                    self.project.add(*[
                        nn.Conv2D(in_channels=in_channels, channels=out_channels,
                                  kernel_size=1, strides=1, use_bias=False, prefix='proj_conv0_'),
                        norm_layer(in_channels=out_channels, prefix='proj_bn0_', **norm_kwargs)])
            else:
                self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                                  stride, root_dim=0,
                                  root_kernel_size=root_kernel_size,
                                  dilation=dilation, root_residual=root_residual,
                                  norm_layer=norm_layer, norm_kwargs=norm_kwargs, prefix='tree_1_')
                self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                                  root_dim=root_dim + out_channels,
                                  root_kernel_size=root_kernel_size,
                                  dilation=dilation, root_residual=root_residual,
                                  norm_layer=norm_layer, norm_kwargs=norm_kwargs, prefix='tree_2_')
            if levels == 1:
                self.root = Root(root_dim, out_channels, root_kernel_size,
                                 root_residual, norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                                 prefix='root_')
            self.level_root = level_root
            self.root_dim = root_dim
            self.levels = levels
            if stride > 1:
                self.downsample.add(nn.MaxPool2D(stride, strides=stride, prefix='maxpool'))

    def hybrid_forward(self, F, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x)
        residual = self.project(bottom)
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, None, children)
        return x

class DLA(HybridBlock):
    def __init__(self, levels, channels, classes=1000,
                 block=BasicBlock, momentum=0.9,
                 norm_layer=BatchNorm, norm_kwargs=None,
                 residual_root=False, linear_root=False,
                 use_feature=False, **kwargs):
        super(DLA, self).__init__(**kwargs)
        if norm_kwargs is None:
            norm_kwargs = {}
        norm_kwargs['momentum'] = momentum
        self._use_feature = use_feature
        self.channels = channels
        self.base_layer = nn.HybridSequential('base')
        self.base_layer.add(nn.Conv2D(in_channels=3, channels=channels[0], kernel_size=7, strides=1,
                                      padding=3, use_bias=False))
        self.base_layer.add(norm_layer(in_channels=channels[0], **norm_kwargs))
        self.base_layer.add(nn.Activation('relu'))

        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0], norm_layer, norm_kwargs)
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], norm_layer, norm_kwargs, stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False, root_residual=residual_root,
                           norm_layer=norm_layer, norm_kwargs=norm_kwargs, prefix='level2_')
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root,
                           norm_layer=norm_layer, norm_kwargs=norm_kwargs, prefix='level3_')
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root,
                           norm_layer=norm_layer, norm_kwargs=norm_kwargs, prefix='level4_')
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root,
                           norm_layer=norm_layer, norm_kwargs=norm_kwargs, prefix='level5_')

        if not self._use_feature:
            self.global_avg_pool = nn.GlobalAvgPool2D()
            self.fc = nn.Dense(units=classes)

    def _make_level(self, block, inplanes, planes, blocks, norm_layer, norm_kwargs, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.HybridSequential()
            downsample.add(*[
                nn.MaxPool2D(stride, strides=stride),
                nn.Conv2D(channels=planes, in_channels=inplanes,
                          kernel_size=1, strides=1, use_bias=False),
                norm_layer(in_channels=planes, **norm_kwargs)])

        layers = []
        layers.append(block(inplanes, planes, stride,
                            norm_layer=norm_layer, norm_kwargs=norm_kwargs, downsample=downsample))
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, norm_layer=norm_layer, norm_kwargs=norm_kwargs))

        curr_level = nn.HybridSequential()
        curr_level.add(*layers)
        return curr_level

    def _make_conv_level(self, inplanes, planes, convs, norm_layer, norm_kwargs,
                         stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2D(in_channels=inplanes, channels=planes, kernel_size=3,
                          strides=stride if i == 0 else 1,
                          padding=dilation, use_bias=False, dilation=dilation),
                norm_layer(**norm_kwargs),
                nn.Activation('relu')])
            inplanes = planes
        curr_level = nn.HybridSequential()
        curr_level.add(*modules)
        return curr_level

    def hybrid_forward(self, F, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            if self._use_feature:
                y.append(x)
            else:
                y.append(F.flatten(self.global_avg_pool(x)))
        if self._use_feature:
            return y
        flat = F.concat(*y, dim=1)
        out = self.fc(flat)
        return out

def get_dla(layers, pretrained=False, ctx=mx.cpu(),
            root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    """Get a center net instance.

    Parameters
    ----------
    name : str or int
        Layers of the network.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : mxnet.Context
        Context such as mx.cpu(), mx.gpu(0).
    root : str
        Model weights storing path.

    Returns
    -------
    HybridBlock
        A DLA network.

    """
    # pylint: disable=unused-variable
    net = DLA(**kwargs)
    if pretrained:
        from .model_store import get_model_file
        full_name = 'dla{}'.format(layers)
        net.load_parameters(get_model_file(full_name, tag=pretrained, root=root),
                            ctx=ctx, ignore_extra=True)
        from ..data import ImageNet1kAttr
        attrib = ImageNet1kAttr()
        net.synset = attrib.synset
        net.classes = attrib.classes
        net.classes_long = attrib.classes_long
    return net

def dla34(**kwargs):
    """DLA 34 layer network for image classification.

    Returns
    -------
    HybridBlock
        A DLA34 network.

    """
    model = get_dla(34, levels=[1, 1, 1, 2, 2, 1],
                    channels=[16, 32, 64, 128, 256, 512],
                    block=BasicBlock, **kwargs)
    return model
