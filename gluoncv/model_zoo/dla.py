"""Deep Layer Aggregation networks, implemented in Gluon."""
# pylint: disable=arguments-differ,unused-argument,missing-docstring
from __future__ import division

from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm

__all__ = ['DLA', 'dla_34']

class DLA(HybridBlock):
    def __init__(self, levels, channels, num_classes=1000,
                 block=BasicBlock, momentum=0.9,
                 norm_layer=BatchNorm, norm_kwargs=None,
                 residual_root=False, linear_root=False, **kwargs):
        super(DLA, self).__init__(**kwargs)
        norm_kwargs['momentum'] = momentum
        self.base_layer = nn.HybridSequential('base')
        self.base_layer.add(nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                            padding=3, bias=False))
        self.base_layer.add(norm_layer(in_channels=channels[0], **norm_kwargs))
        self.base_layer.add(nn.Activation('relu'))

        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False,
                           root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def _make_level(self, block, inplanes, planes, blocks, stride=1, norm_layer, norm_kwargs):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.HybridSequential()
            downsample.add([
                nn.MaxPool2D(stride, stride=stride),
                nn.Conv2D(planes, in_channels=inplanes,
                          kernel_size=1, stride=1, use_bias=False),
                norm_layer(in_channels=planes, **norm_kwargs)]
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample=downsample))
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    def _make_conv_level(self, inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                nn.BatchNorm2d(planes, momentum=BN_MOMENTUM),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def hybrid_forward(self, F, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        return y
