"""Multi-style Generative Network for Real-time Transfer"""
import numpy as np
import mxnet as mx
from mxnet import cpu
from mxnet.gluon import nn, Block, HybridBlock
import mxnet.ndarray as F
# pylint: disable=arguments-differ,redefined-outer-name,unused-argument

__all__ = ['MSGNet', 'Inspiration', 'get_msgnet', 'gram_matrix']

class MSGNet(Block):
    r"""Multi-style Generative Network for Real-time Transfer

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    ngf : int
        Number of filters in the generative network.
    norm_layer : object
        Normalization layer used in the network (default: :class:`mxnet.gluon.nn.InstanceNorm`;


    Reference:

        Hang Zhang and Kristin Dana. "Multi-style Generative Network for Real-time Transfer."
        *arXiv preprint arXiv:1703.06953 (2017)*

    Examples
    --------
    >>> model = MSGNet()
    >>> print(model)
    """
    def __init__(self, in_channels=3, out_channels=3, ngf=128,
                 norm_layer=nn.InstanceNorm, n_blocks=6):
        super(MSGNet, self).__init__()

        block = Bottleneck
        upblock = UpBottleneck
        expansion = 4

        with self.name_scope():
            self.model1 = nn.Sequential()
            self.ins = Inspiration(ngf*expansion)
            self.model = nn.Sequential()

            self.model1.add(ConvLayer(in_channels, 64, kernel_size=7, stride=1))
            self.model1.add(norm_layer(in_channels=64))
            self.model1.add(nn.Activation('relu'))
            self.model1.add(block(64, 32, 2, 1, norm_layer))
            self.model1.add(block(32*expansion, ngf, 2, 1, norm_layer))


            self.model.add(self.model1)
            self.model.add(self.ins)

            for _ in range(n_blocks):
                self.model.add(block(ngf*expansion, ngf, 1, None, norm_layer))

            self.model.add(upblock(ngf*expansion, 32, 2, norm_layer))
            self.model.add(upblock(32*expansion, 16, 2, norm_layer))
            self.model.add(norm_layer(in_channels=16*expansion))
            self.model.add(nn.Activation('relu'))
            self.model.add(ConvLayer(16*expansion, out_channels, kernel_size=7, stride=1))


    def set_target(self, Xs):
        feature = self.model1(Xs)
        gram = gram_matrix(feature)
        self.ins.set_target(gram)

    def forward(self, x):
        return self.model(x)


class Inspiration(Block):
    r"""
    Inspiration Layer (CoMatch Layer) enables the multi-style transfer in feed-forward
    network, which learns to match the target feature statistics during the training.
    This module is differentialble and can be inserted in standard feed-forward network
    to be learned directly from the loss function without additional supervision.

    .. math::
        Y = \phi^{-1}[\phi(\mathcal{F}^T)W\mathcal{G}]

    Please see the `example of MSG-Net <./experiments/style.html>`_
    training multi-style generative network for real-time transfer.

    Reference:
        Hang Zhang and Kristin Dana. "Multi-style Generative Network for Real-time Transfer."
        *arXiv preprint arXiv:1703.06953 (2017)*
    """
    def __init__(self, C, B=1):
        super(Inspiration, self).__init__()
        # B is equal to 1 or input mini_batch
        self.C = C
        self.weight = self.params.get('weight', shape=(1, C, C),
                                      init=mx.initializer.Uniform(),
                                      allow_deferred_init=True)
        self.gram = F.random.uniform(shape=(B, C, C))

    def set_target(self, target):
        self.gram = target

    def forward(self, X):
        # input X is a 3D feature map
        self.P = F.batch_dot(F.broadcast_to(self.weight.data(), shape=(self.gram.shape)), self.gram)
        return F.batch_dot(
            F.SwapAxis(self.P, 1, 2).broadcast_to((X.shape[0], self.C, self.C)),
            X.reshape((0, 0, X.shape[2]*X.shape[3]))).reshape(X.shape)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'N x ' + str(self.C) + ')'


def get_msgnet(styles=21, pretrained=False, root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    r"""Multi-style Generative Network for Real-time Transfer

    Parameters
    ----------
    styles : int, default 21
        Number of styles for the pretrained model.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_msgnet(pretrained=True)
    >>> print(model)
    """
    # infer number of classes
    model = MSGNet(ngf=128, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_params(get_model_file('msgnet_%dstyles'%(styles), root=root), ctx=ctx)
    return model


def gram_matrix(y):
    r""" Gram Matrix for a 4D convolutional featuremaps as a mini-batch

    .. math::
        \mathcal{G} = \sum_{h=1}^{H_i}\sum_{w=1}^{W_i} \mathcal{F}_{h,w}\mathcal{F}_{h,w}^T
    """
    (b, ch, h, w) = y.shape
    features = y.reshape((b, ch, w * h))
    gram = F.batch_dot(features, features, transpose_b=True) / (ch * h * w)
    return gram


class Bottleneck(HybridBlock):
    """ Pre-activation residual block
    Identity Mapping in Deep Residual Networks
    ref https://arxiv.org/abs/1603.05027
    """
    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.InstanceNorm):
        super(Bottleneck, self).__init__()
        self.expansion = 4
        self.downsample = downsample
        if self.downsample is not None:
            self.residual_layer = nn.Conv2D(in_channels=inplanes,
                                            channels=planes * self.expansion,
                                            kernel_size=1, strides=(stride, stride))
        self.conv_block = nn.HybridSequential()
        with self.conv_block.name_scope():
            self.conv_block.add(norm_layer(in_channels=inplanes))
            self.conv_block.add(nn.Activation('relu'))
            self.conv_block.add(nn.Conv2D(in_channels=inplanes, channels=planes,
                                          kernel_size=1))
            self.conv_block.add(norm_layer(in_channels=planes))
            self.conv_block.add(nn.Activation('relu'))
            self.conv_block.add(ConvLayer(planes, planes, kernel_size=3, stride=stride))
            self.conv_block.add(norm_layer(in_channels=planes))
            self.conv_block.add(nn.Activation('relu'))
            self.conv_block.add(nn.Conv2D(in_channels=planes, channels=planes * self.expansion,
                                          kernel_size=1))

    def hybrid_forward(self, F, x):
        if self.downsample is not None:
            residual = self.residual_layer(x)
        else:
            residual = x
        return residual + self.conv_block(x)


class UpBottleneck(HybridBlock):
    """ Up-sample residual block (from MSG-Net paper)
    Enables passing identity all the way through the generator
    ref https://arxiv.org/abs/1703.06953
    """
    def __init__(self, inplanes, planes, stride=2, norm_layer=nn.InstanceNorm):
        super(UpBottleneck, self).__init__()
        self.expansion = 4
        self.residual_layer = UpsampleConvLayer(inplanes, planes * self.expansion,
                                                kernel_size=1, stride=1, upsample=stride)
        self.conv_block = nn.HybridSequential()
        with self.conv_block.name_scope():
            self.conv_block.add(norm_layer(in_channels=inplanes))
            self.conv_block.add(nn.Activation('relu'))
            self.conv_block.add(nn.Conv2D(in_channels=inplanes, channels=planes,
                                          kernel_size=1))
            self.conv_block.add(norm_layer(in_channels=planes))
            self.conv_block.add(nn.Activation('relu'))
            self.conv_block.add(UpsampleConvLayer(planes, planes, kernel_size=3, stride=1,
                                                  upsample=stride))
            self.conv_block.add(norm_layer(in_channels=planes))
            self.conv_block.add(nn.Activation('relu'))
            self.conv_block.add(nn.Conv2D(in_channels=planes, channels=planes * self.expansion,
                                          kernel_size=1))

    def hybrid_forward(self, F, x):
        return  self.residual_layer(x) + self.conv_block(x)


class ConvLayer(HybridBlock):
    """Convolution with Reflection Padding
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        pad_size = int(np.floor(kernel_size / 2))
        self.pad = nn.ReflectionPad2D(padding=pad_size)
        self.conv2d = nn.Conv2D(in_channels=in_channels, channels=out_channels,
                                kernel_size=kernel_size, strides=(stride, stride),
                                padding=0)

    def hybrid_forward(self, F, x):
        x = self.pad(x)
        out = self.conv2d(x)
        return out


class UpsampleConvLayer(HybridBlock):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        self.reflection_padding = int(np.floor(kernel_size / 2))
        self.conv2d = nn.Conv2D(in_channels=in_channels, channels=out_channels,
                                kernel_size=kernel_size, strides=(stride, stride),
                                padding=self.reflection_padding)

    def hybrid_forward(self, F, x):
        if self.upsample:
            x = F.UpSampling(x, scale=self.upsample, sample_type='nearest')
        out = self.conv2d(x)
        return out
