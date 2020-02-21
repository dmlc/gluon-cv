from __future__ import division

from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm
from mxnet.gluon.contrib.nn import HybridConcurrent

__all__ = ['get_blmodel']


# reference: https://github.com/IBM/BigLittleNet

def _conv3x3(channels, stride, in_channels):
    return nn.Conv2D(channels, kernel_size=3, strides=stride, padding=1,
                     use_bias=False, in_channels=in_channels)

def make_l_layer(channel, norm_layer, alpha, prefix):
    out = nn.HybridSequential(prefix=prefix)
    with out.name_scope():
        out.add(nn.Conv2D(channel // alpha, 3, 1, 1, use_bias=False, in_channels=channel))
        out.add(norm_layer(in_channels=channel // alpha))
        out.add(nn.Activation('relu'))
        out.add(nn.Conv2D(channel // alpha, 3, 2, 1, use_bias=False, in_channels=channel // alpha))
        out.add(norm_layer(in_channels=channel // alpha))
        out.add(nn.Activation('relu'))
        out.add(nn.Conv2D(channel, 1, 1, use_bias=False, in_channels=channel // alpha))
        out.add(norm_layer(in_channels=channel))
    return out

def make_b_layer(channel, norm_layer, prefix):
    out = nn.HybridSequential(prefix=prefix)
    with out.name_scope():
        out.add(nn.Conv2D(channel, 3, 2, 1, use_bias=False, in_channels=channel))
        out.add(norm_layer(in_channels=channel))
    return out

def make_bl_layer(channel, norm_layer, prefix):
    out = nn.HybridSequential(prefix=prefix)
    with out.name_scope():
        out.add(nn.Conv2D(channel, 1, 1, use_bias=False, in_channels=channel))
        out.add(norm_layer(in_channels=channel))
        out.add(nn.Activation('relu'))
    return out

class BLModule_0(HybridBlock):
    def __init__(self, channel, alpha, norm_layer):
        super(BLModule_0, self).__init__()
        self.relu = nn.Activation('relu')
        with self.name_scope():
            self.big_branch = make_b_layer(channel, norm_layer, 'b_branch')
            self.little_branch = make_l_layer(channel, norm_layer, alpha, 'l_branch')
            self.fusion = make_bl_layer(channel, norm_layer, 'make_bl_layer')

    def hybrid_forward(self, F, x):
        bx = self.big_branch(x)
        lx = self.little_branch(x)
        x = self.relu(bx + lx)
        out = self.fusion(x)
        return out

def blm_make_layer(block, inplanes, planes, blocks, stride=1, last_relu=True):
    downsample = nn.HybridSequential(prefix='')
    with downsample.name_scope():
        if stride != 1:
            downsample.add(nn.AvgPool2D(3, strides=2, padding=1))
        if inplanes != planes:
            downsample.add(nn.Conv2D(planes, kernel_size=1, strides=1, in_channels=inplanes))
            downsample.add(BatchNorm(in_channels=planes))
    layers = nn.HybridSequential(prefix='')
    with layers.name_scope():
        if blocks == 1:
            layers.add(block(inplanes, planes, stride=stride, downsample=downsample))
        else:
            layers.add(block(inplanes, planes, stride=stride, downsample=downsample))
            for i in range(1, blocks):
                layers.add(block(planes, planes, last_relu=last_relu if i == blocks - 1 else True))
    return layers

def BLModule_4(block, inplanes, planes, blocks, stride=1):
    downsample = nn.HybridSequential(prefix='downsample')
    with downsample.name_scope():
        if stride != 1:
            downsample.add(nn.AvgPool2D(3, strides=2, padding=1))
        if inplanes != planes:
            downsample.add(nn.Conv2D(planes, kernel_size=1, strides=1, in_channels=inplanes))
            downsample.add(BatchNorm(in_channels=planes))
    layers = nn.HybridSequential(prefix='')
    with layers.name_scope():
        layers.add(block(inplanes, planes, stride=stride, downsample=downsample))
        for i in range(1, blocks):
            layers.add(block(planes, planes))
    return layers

class BottleneckV1(HybridBlock):
    r"""Bottleneck V1 from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    This is used for ResNet V1 for 50, 101, 152 layers.

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
    last_gamma : bool, default False
        Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
    use_se : bool, default False
        Whether to use Squeeze-and-Excitation module
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, last_relu=True, **kwargs):
        super(BottleneckV1, self).__init__(**kwargs)
        self.body = nn.HybridSequential(prefix='')
        self.body.add(nn.Conv2D(planes // self.expansion, kernel_size=1, strides=1, in_channels=inplanes))
        self.body.add(BatchNorm(in_channels=planes // self.expansion))  # ->
        self.body.add(nn.Activation('relu'))
        self.body.add(_conv3x3(planes // self.expansion, stride, planes // self.expansion))
        self.body.add(BatchNorm(in_channels=planes // self.expansion))
        self.body.add(nn.Activation('relu'))
        self.body.add(nn.Conv2D(planes, kernel_size=1, strides=1, in_channels=planes // self.expansion))
        self.body.add(BatchNorm(in_channels=planes))
        self.body.add(nn.Activation('relu'))
        self.downsample = downsample
        self.last_relu = last_relu

    def hybrid_forward(self, F, x):
        residual = x
        out = self.body(x)
        if self.downsample is not None:
            residual = self.downsample(x)
        if self.last_relu:
            out = F.Activation(out + residual, act_type='relu')
            return out
        else:
            return out + residual

class BLModule(HybridBlock):
    def __init__(self, block, int_channels, out_channels, blocks, alpha, beta, stride, hw):
        super(BLModule, self).__init__()
        self.hw = hw
        self.big = blm_make_layer(block, int_channels, out_channels, blocks - 1, 2, last_relu=False)
        self.little_e = nn.HybridSequential(prefix='')
        self.little_e.add(blm_make_layer(block, int_channels, out_channels // alpha, max(1, blocks // beta - 1)))
        self.little_e.add(nn.Conv2D(out_channels, kernel_size=1, in_channels=out_channels // alpha))
        self.little_e.add(BatchNorm(in_channels=out_channels))
        self.relu = nn.Activation('relu')
        self.fusion = blm_make_layer(block, out_channels, out_channels, 1, stride=stride)

    def hybrid_forward(self, F, x):
        big = self.big(x)
        little = self.little_e(x)
        big = F.contrib.BilinearResize2D(data=big, height=self.hw, width=self.hw)
        out = self.relu(big + little)
        out = self.fusion(out)
        return out

class BLResNetV1(HybridBlock):
    def __init__(self, block, layers, channels, alpha=2, beta=4, classes=1000, thumbnail=False,
                 last_gamma=False, use_se=False, norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(BLResNetV1, self).__init__(**kwargs)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            with self.features.name_scope():
                self.features.add(nn.Conv2D(channels[0], 7, 2, 3, use_bias=False, in_channels=3))
                self.features.add(norm_layer(in_channels=channels[0]))
                self.features.add(nn.Activation('relu'))
                self.features.add(
                    BLModule_0(channels[0], alpha, norm_layer))
                self.features.add(
                    BLModule(block, channels[0], channels[0] * block.expansion, layers[0], alpha, beta, stride=2, hw=56))
                self.features.add(
                    BLModule(block, channels[0] * block.expansion, channels[1] * block.expansion, layers[1], alpha, beta, stride=2, hw=28))
                self.features.add(
                    BLModule(block, channels[1] * block.expansion, channels[2] * block.expansion, layers[2], alpha, beta, stride=1, hw=14))
                self.features.add(
                    BLModule_4(block, channels[2] * block.expansion, channels[3] * block.expansion, layers[3], stride=2))
                self.features.add(nn.GlobalAvgPool2D())
                self.features.add(nn.Flatten())
                self.fc = nn.Dense(classes, in_units=channels[-1] * block.expansion)

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.fc(x)
        return x

# Specification
resnet_spec = {50: ('bottle_neck', [3, 4, 6, 3], [64, 128, 256, 512])}


def blget_resnet(version, num_layers, pretrained=False, ctx=cpu(),
                 root='~/.mxnet/models', use_se=False, **kwargs):
    r"""ResNet V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    ResNet V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    version : int
        Version of ResNet. Options are 1, 2.
    num_layers : int
        Numbers of layers. Options are 18, 34, 50, 101, 152.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    use_se : bool, default False
        Whether to use Squeeze-and-Excitation module
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    assert num_layers in resnet_spec, \
        "Invalid number of layers: %d. Options are %s" % (
            num_layers, str(resnet_spec.keys()))
    block_type, layers, channels = resnet_spec[num_layers]
    assert 1 <= version <= 2, \
        "Invalid resnet version: %d. Options are 1 and 2." % version

    blresnet_class = BLResNetV1
    block_class = BottleneckV1
    alpha = 2
    beta = 4
    net = blresnet_class(block_class, layers, channels, alpha, beta, **kwargs)
    return net


def blresnet50_v1(**kwargs):
    r"""ResNet-50 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '$MXNET_HOME/models'
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    return blget_resnet(1, 50, use_se=False, **kwargs)


_models = {'blresnet50_v1': blresnet50_v1}


def get_blmodel(name, **kwargs):
    """Returns a pre-defined model by name

    Parameters
    ----------
    name : str
        Name of the model.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    classes : int
        Number of classes for the output layer.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Returns
    -------
    HybridBlock
        The model.
    """
    name = name.lower()
    if name not in _models:
        err_str = '"%s" is not among the following model list:\n\t' % (name)
        err_str += '%s' % ('\n\t'.join(sorted(_models.keys())))
        raise ValueError(err_str)
    net = _models[name](**kwargs)
    return net


