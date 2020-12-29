import numpy as np
from math import sqrt

import mxnet as mx
import mxnet.ndarray as nd
import mxnet.gluon.nn as nn
from numpy import prod
# pylint: disable-all

def get_weight_key(module):
    for k in module.params.keys():
        if 'weight' in k:
            weight_key = k

    return weight_key


def compute_weight(weight_orig):
    fan_in = weight_orig.shape[1] * weight_orig[0][0].size

    return weight_orig * sqrt(2 / (fan_in + 1e-8))


class FusedUpsample(nn.HybridBlock):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super().__init__()

        fan_in = in_channel * kernel_size * kernel_size
        self.multiplier = sqrt(2 / (fan_in))
        self.weight = self.params.get('weight', shape=(in_channel, out_channel, kernel_size, kernel_size),
                                      init=mx.init.Normal(sigma=1))
        self.bias = self.params.get('bias', shape=(out_channel), init=mx.init.Zero())
        self.pad = (padding, padding)

    def hybrid_forward(self, F, x, **kwargs):
        weight = F.pad(kwargs['weight'] * self.multiplier, mode='constant', 
                       constant_value=0, pad_width=(0, 0, 0, 0, 1, 1, 1, 1))
        weight = (
            weight[:, :, 1:, 1:]
            + weight[:, :, :-1, 1:]
            + weight[:, :, 1:, :-1]
            + weight[:, :, :-1, :-1]
        ) / 4

        out = F.Deconvolution(x, weight, kwargs['bias'], kernel=weight.shape[-2:], stride=(2, 2), 
                              pad=self.pad, num_filter=weight.shape[1], no_bias=False)
        return out


class FusedDownsample(nn.HybridBlock):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super().__init__()

        self.weight = self.params.get('weight', shape=(in_channel, out_channel, kernel_size, kernel_size),
                                      grad_req='write', init=mx.init.Normal(sigma=1))
        self.bias = self.params.get('bias', shape=(out_channel), grad_req='write', init=mx.init.Zero())
        fan_in = in_channel * kernel_size * kernel_size
        self.multiplier = sqrt(2 / (fan_in + 1e-8))
        self.pad = (padding, padding)


    def hybrid_forward(self, F, x, **kwargs):
        weight = F.pad(kwargs['weight'] * self.multiplier, mode='constant', 
                       constant_value=0, pad_width=(0, 0, 0, 0, 1, 1, 1, 1))
        weight = (
            weight[:, :, 1:, 1:]
            + weight[:, :, :-1, 1:]
            + weight[:, :, 1:, :-1]
            + weight[:, :, :-1, :-1]
        ) / 4

        out = F.Convolution(x, weight, kwargs['bias'], kernel=weight.shape[-2:], stride=(2, 2), 
                            pad=self.pad, num_filter=weight.shape[1], no_bias=False)
        return out


class PixelNorm(nn.HybridBlock):
    def __init__(self):
        super().__init__()

    def hybrid_forward(self, F, x):
        return x / F.sqrt(F.mean(x ** 2, axis=1, keepdims=True) + 1e-8)


class Blur(nn.HybridBlock):
    def __init__(self, channel):
        super().__init__() 

        self.channel = channel

        weight = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32)
        weight = weight.reshape((1, 1, 3, 3))
        weight = weight / (weight.sum()+1e-8)

        self. weight = nd.array(weight).tile((channel, 1, 1, 1))
        weight_flip = np.flip(weight, 2)
        self.weight_flip = nd.array(weight_flip).tile((channel, 1, 1, 1))

    def hybrid_forward(self, F, x, **kwargs):

        weight = nd.array(self.weight, ctx=x.context)
        output = F.Convolution(x, weight, kernel=self.weight.shape[-2:], pad=(1, 1), 
                               num_filter=self.channel, num_group=x.shape[1], no_bias=True)
        return output


class EqualConv2d(nn.HybridBlock):
    def __init__(self, in_dim, out_dim, kernel, padding=0):
        super().__init__()

        with self.name_scope():
            self.weight = self.params.get('weight_orig', shape=(out_dim, in_dim, kernel, kernel), grad_req='write',
                                          init=mx.init.Normal(1))
            self.bias = self.params.get('bias', shape=(out_dim), grad_req='write', init=mx.init.Zero())
            self.kernel = (kernel, kernel)
            self.channel = out_dim
            self.padding = (padding, padding)

    def hybrid_forward(self, F, x, **kwargs):

        size = kwargs['weight'].shape
        fan_in = prod(size[1:])
        multiplier = sqrt(2.0 / fan_in)

        out = F.Convolution(x, kwargs['weight']*multiplier, kwargs['bias'], kernel=self.kernel, pad=self.padding,
                               num_filter=self.channel)
        return out 


class EqualLinear(nn.HybridBlock):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.weight = self.params.get('weight_orig', shape=(out_dim, in_dim), grad_req='write', init=mx.init.Normal(1))
        self.bias = self.params.get('bias', shape=(out_dim), grad_req='write', init=mx.init.Zero())
        self.num_hidden = out_dim

    def hybrid_forward(self, F, x, **kwargs):

        size = kwargs['weight'].shape
        fan_in = prod(size[1:])
        multiplier = sqrt(2.0 / fan_in)

        out = F.FullyConnected(x, kwargs['weight']*multiplier, kwargs['bias'], num_hidden=self.num_hidden)

        return out


class AdaptiveInstanceNorm(nn.HybridBlock):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.InstanceNorm(in_channels=in_channel)
        self.style = EqualLinear(style_dim, in_channel * 2)
        self.style.initialize()

        mx_params = self.style.collect_params()
        for k in mx_params.keys():
            if 'bias' in k:
                mx_params[k].data()[:in_channel] = 1
                mx_params[k].data()[in_channel:] = 0

    def hybrid_forward(self, F, x, style, **kwargs):
        style = self.style(style).expand_dims(2).expand_dims(3)
        gamma, beta = style.split(2, 1)
        out = self.norm(x)
        out = gamma * out + beta

        return out


class NoiseInjection(nn.HybridBlock):
    def __init__(self, channel):
        super().__init__()

        self.weight = self.params.get('weight_orig', shape=(1, channel, 1, 1), init=mx.init.Zero())

    def hybrid_forward(self, F, image, noise, **kwargs):
        new_weight = compute_weight(kwargs['weight'])

        return image + new_weight * noise


class ConstantInput(nn.HybridBlock):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = self.params.get('input', shape=(1, channel, size, size), init=mx.init.Normal(sigma=1))

    def hybrid_forward(self, F, x, **kwargs):
        batch = x.shape[0]
        out = kwargs['input'].tile((batch, 1, 1, 1))

        return out

class ConvBlock(nn.HybridBlock):
    def __init__(self, in_channel, out_channel, kernel_size, padding, kernel_size2=None, 
                 padding2=None, downsample=False, fused=False):
        super().__init__()

        pad1 = padding
        pad2 = padding
        if padding2 is not None:
            pad2 = padding2

        kernel1 = kernel_size
        kernel2 = kernel_size
        if kernel_size2 is not None:
            kernel2 = kernel_size2

        self.conv1 = nn.HybridSequential()
        with self.conv1.name_scope():
            self.conv1.add(EqualConv2d(in_channel, out_channel, kernel1, padding=pad1))
            self.conv1.add(nn.LeakyReLU(0.2))
        
        if downsample:
            if fused:
                self.conv2 = nn.HybridSequential()
                with self.conv2.name_scope():
                    self.conv2.add(FusedDownsample(out_channel, out_channel, kernel2, padding=pad2))
                    self.conv2.add(nn.LeakyReLU(0.2))
                
            else:
                self.conv2 = nn.HybridSequential()
                with self.conv2.name_scope():
                    self.conv2.add(EqualConv2d(out_channel, out_channel, kernel2, padding=pad2))
                    self.conv2.add(nn.AvgPool2D(pool_size=(2, 2)))
                    self.conv2.add(nn.LeakyReLU(0.2))
                
        else:
            self.conv2 = nn.HybridSequential()
            with self.conv2.name_scope():
                self.conv2.add(EqualConv2d(out_channel, out_channel, kernel2, padding=pad2))
                self.conv2.add(nn.LeakyReLU(0.2))

    def hybrid_forward(self, F, x):
        out = self.conv1(x)
        out = self.conv2(out)

        return out


class StyledConvBlock(nn.HybridBlock):
    def __init__(self, in_channel, out_channel, kernel_size=3, padding=1, style_dim=512,
                 initial=False, upsample=False, fused=False, blur=False):

        super().__init__()

        self.upsample = None
        if initial:
            self.conv1 = ConstantInput(in_channel)
        else:
            if upsample:
                if fused:
                    self.upsample = 'fused'
                    self.conv1 = nn.HybridSequential()
                    with self.conv1.name_scope():
                        self.conv1.add(FusedUpsample(in_channel, out_channel, kernel_size, padding=padding))
                        if blur:
                            self.conv1.add(Blur(out_channel))
                else:
                    self.upsample = 'nearest'
                    self.conv1 = nn.HybridSequential()
                    with self.conv1.name_scope():
                        self.conv1.add(EqualConv2d(in_dim=in_channel, out_dim=out_channel, 
                                                   kernel=kernel_size, padding=padding))
                        if blur:
                            self.conv1.add(Blur(out_channel))
            else:
                self.conv1 = EqualConv2d(in_dim=in_channel, out_dim=out_channel, 

                                         kernel=kernel_size, padding=padding)

        self.noise1 = NoiseInjection(out_channel)
        self.adain1 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.lrelu1 = nn.LeakyReLU(0.2)

        self.conv2 = EqualConv2d(in_dim=out_channel, out_dim=out_channel, kernel=kernel_size, padding=padding)
        self.noise2 = NoiseInjection(out_channel)
        self.adain2 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.lrelu2 = nn.LeakyReLU(0.2)

    def hybrid_forward(self, F, x, style, noise):
        #  Upsample
        if self.upsample == 'nearest':
            x = F.UpSampling(x, scale=2, sample_type='nearest')
        out = self.conv1(x)
        out = self.noise1(out, noise)
        out = self.lrelu1(out)
        out = self.adain1(out, style)

        out = self.conv2(out)
        out = self.noise2(out, noise)
        out = self.lrelu2(out)
        out = self.adain2(out, style)

        return out
