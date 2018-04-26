"""Dilated_ResNets, implemented in Gluon."""
# pylint: disable=arguments-differ,unused-argument
from __future__ import division

__all__ = ['DilatedResNetV2', 'DilatedBasicBlockV2', 'DilatedBottleneckV2',
           'dilated_resnet18', 'dilated_resnet34', 'dilated_resnet50',
           'dilated_resnet101', 'dilated_resnet152', 'get_dilated_resnet']

from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm

# Helpers
def _conv3x3(channels, stride, in_channels, dilation=1):
    return nn.Conv2D(channels, kernel_size=3, strides=stride, padding=dilation,
                     use_bias=False, in_channels=in_channels, dilation=dilation)

# Blocks
class DilatedBasicBlockV2(HybridBlock):
    r"""BasicBlock V2 from
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.
    This is used for Dilated_ResNet V2 for 18, 34 layers.

    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    """
    def __init__(self, channels, stride, downsample=False, in_channels=0,
                 dilation=1, previous_dilation=1, norm_layer=None, **kwargs):
        super(DilatedBasicBlockV2, self).__init__(**kwargs)
        self.bn1 = norm_layer()
        self.conv1 = _conv3x3(channels, stride, in_channels, dilation=dilation)
        self.bn2 = norm_layer()
        self.conv2 = _conv3x3(channels, 1, channels, dilation=previous_dilation)
        if downsample:
            self.downsample = nn.Conv2D(channels, 1, stride, use_bias=False,
                                        in_channels=in_channels)
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        """Hybrid forward"""
        residual = x
        x = self.bn1(x)
        x = F.Activation(x, act_type='relu')
        if self.downsample:
            residual = self.downsample(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = F.Activation(x, act_type='relu')
        x = self.conv2(x)

        return x + residual


class DilatedBottleneckV2(HybridBlock):
    r"""Bottleneck V2 from
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.
    This is used for Dilated_ResNet V2 for 50, 101, 152 layers.

    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    """
    def __init__(self, channels, stride, downsample=False, in_channels=0,
                 dilation=1, previous_dilation=1, norm_layer=None, **kwargs):
        super(DilatedBottleneckV2, self).__init__(**kwargs)
        self.bn1 = norm_layer()
        self.conv1 = nn.Conv2D(channels//4, kernel_size=1, strides=1, use_bias=False)
        self.bn2 = norm_layer()
        self.conv2 = _conv3x3(channels//4, stride, channels//4, dilation)
        self.bn3 = norm_layer()
        self.conv3 = nn.Conv2D(channels, kernel_size=1, strides=1, use_bias=False)
        if downsample:
            self.downsample = nn.Conv2D(channels, 1, stride, use_bias=False,
                                        in_channels=in_channels)
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        """Hybrid forward"""
        residual = x
        x = self.bn1(x)
        x = F.Activation(x, act_type='relu')
        if self.downsample:
            residual = self.downsample(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = F.Activation(x, act_type='relu')
        x = self.conv2(x)

        x = self.bn3(x)
        x = F.Activation(x, act_type='relu')
        x = self.conv3(x)

        return x + residual


# Nets
class DilatedResNetV2(HybridBlock):
    r"""Dilated_ResNet V2 model from
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    block : Block
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    channels : list of int
        Numbers of channels in each block. Length should be one larger than layers list.
    classes : int, default 1000
        Number of classification classes.
    thumbnail : bool, default False
        Enable thumbnail.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    """
    def __init__(self, block, layers, channels, classes=1000, thumbnail=False,
                 norm_layer=BatchNorm, **kwargs):
        super(DilatedResNetV2, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(norm_layer(scale=False, center=False))
            if thumbnail:
                self.features.add(_conv3x3(channels[0], 1, 3))
            else:
                self.features.add(nn.Conv2D(channels[0], 7, 2, 3, use_bias=False,
                                            in_channels=3))
                self.features.add(norm_layer())
                self.features.add(nn.Activation('relu'))
                self.features.add(nn.MaxPool2D(3, 2, 1))

            in_channels = channels[0]
            strides = [1, 2, 1, 1, 1]
            dilations = [1, 1, 2, 4, 4]
            for i, num_layer in enumerate(layers):
                dilation = dilations[i]
                stride = strides[i]
                self.features.add(self._make_layer(block, num_layer, channels[i+1],
                                                   stride, i+1, in_channels=in_channels,
                                                   dilation=dilation, norm_layer=norm_layer))
                in_channels = channels[i+1]
            self.features.add(norm_layer())
            self.features.add(nn.Activation('relu'))
            # self.features.add(nn.GlobalAvgPool2D())
            # self.features.add(nn.Flatten())
            # for loading pre-trained weights
            self.output = nn.Dense(classes, in_units=in_channels)

    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0,
                    dilation=1, norm_layer=None):
        layer = nn.HybridSequential(prefix='stage%d_'%stage_index)
        with layer.name_scope():
            if dilation == 1 or dilation == 2:
                layer.add(block(channels, stride, channels != in_channels,
                                in_channels=in_channels, dilation=1,
                                previous_dilation=dilation, norm_layer=norm_layer,
                                prefix=''))
            elif dilation == 4:
                layer.add(block(channels, stride, channels != in_channels,
                                in_channels=in_channels, dilation=2,
                                previous_dilation=dilation, norm_layer=norm_layer,
                                prefix=''))
            for _ in range(layers-1):
                layer.add(block(channels, 1, False, in_channels=channels,
                                dilation=dilation, previous_dilation=dilation,
                                norm_layer=norm_layer, prefix=''))
        return layer

    def hybrid_forward(self, F, x):
        x = self.features(x)
        #x = self.output(x)
        return x


# Specification
dilated_resnet_spec = {
    18: ('basic_block', [2, 2, 2, 2], [64, 64, 128, 256, 512]),
    34: ('basic_block', [3, 4, 6, 3], [64, 64, 128, 256, 512]),
    50: ('bottle_neck', [3, 4, 6, 3], [64, 256, 512, 1024, 2048]),
    101: ('bottle_neck', [3, 4, 23, 3], [64, 256, 512, 1024, 2048]),
    152: ('bottle_neck', [3, 8, 36, 3], [64, 256, 512, 1024, 2048])}

dilated_resnet_block_versions = {
    'basic_block': DilatedBasicBlockV2,
    'bottle_neck': DilatedBottleneckV2}


# Constructor
def get_dilated_resnet(version, num_layers, pretrained=False, ctx=cpu(), root='~/.mxnet/models',
                       **kwargs):
    r"""Dilated_ResNet V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    Dilated_ResNet V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    version : int
        Version of Dilated_ResNet. Options are 1, 2.
    num_layers : int
        Numbers of layers. Options are 18, 34, 50, 101, 152.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    """
    block_type, layers, channels = dilated_resnet_spec[num_layers]
    block_class = dilated_resnet_block_versions[block_type]
    net = DilatedResNetV2(block_class, layers, channels, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        net.load_params(get_model_file('resnet%d_v%d'%(num_layers, version),
                                       root=root), ctx=ctx)
    return net

def dilated_resnet18(**kwargs):
    r"""Dilated_ResNet-18 V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    """
    return get_dilated_resnet(2, 18, **kwargs)

def dilated_resnet34(**kwargs):
    r"""Dilated_ResNet-34 V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    """
    return get_dilated_resnet(2, 34, **kwargs)

def dilated_resnet50(**kwargs):
    r"""Dilated_ResNet-50 V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    """
    return get_dilated_resnet(2, 50, **kwargs)

def dilated_resnet101(**kwargs):
    r"""Dilated_ResNet-101 V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    """
    return get_dilated_resnet(2, 101, **kwargs)

def dilated_resnet152(**kwargs):
    r"""Dilated_ResNet-152 V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    """
    return get_dilated_resnet(2, 152, **kwargs)
