# pylint: disable=abstract-method,unused-argument,arguments-differ,missing-docstring
"""Customized Layers.
"""
from __future__ import absolute_import
from mxnet import initializer
from mxnet.gluon import nn, contrib
from mxnet.gluon.nn import BatchNorm, HybridBlock

__all__ = ['BatchNormCudnnOff', 'Consensus', 'ReLU6', 'HardSigmoid', 'HardSwish']

class BatchNormCudnnOff(BatchNorm):
    """Batch normalization layer without CUDNN. It is a temporary solution.

    Parameters
    ----------
    kwargs : arguments goes to mxnet.gluon.nn.BatchNorm
    """
    def __init__(self, **kwargs):
        super(BatchNormCudnnOff, self).__init__(**kwargs)

    def hybrid_forward(self, F, x, gamma, beta, running_mean, running_var):
        return F.BatchNorm(x, gamma, beta, running_mean, running_var,
                           name='fwd', cudnn_off=True, **self._kwargs)

class Consensus(HybridBlock):
    """Consensus used in temporal segment networks.

    Parameters
    ----------
    nclass : number of classses
    num_segments : number of segments
    kwargs : arguments goes to mxnet.gluon.nn.Consensus
    """

    def __init__(self, nclass, num_segments, **kwargs):
        super(Consensus, self).__init__(**kwargs)
        self.nclass = nclass
        self.num_segments = num_segments

    def hybrid_forward(self, F, x):
        reshape_out = x.reshape((-1, self.num_segments, self.nclass))
        consensus_out = reshape_out.mean(axis=1)
        return consensus_out

class ReLU6(HybridBlock):
    """RelU6 used in MobileNetV2 and MobileNetV3.

    Parameters
    ----------
    kwargs : arguments goes to mxnet.gluon.nn.ReLU6
    """

    def __init__(self, **kwargs):
        super(ReLU6, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return F.clip(x, 0, 6, name="relu6")

class HardSigmoid(HybridBlock):
    """HardSigmoid used in MobileNetV3.

    Parameters
    ----------
    kwargs : arguments goes to mxnet.gluon.nn.HardSigmoid
    """
    def __init__(self, **kwargs):
        super(HardSigmoid, self).__init__(**kwargs)
        self.act = ReLU6()

    def hybrid_forward(self, F, x):
        return self.act(x + 3.) / 6.

class HardSwish(HybridBlock):
    """HardSwish used in MobileNetV3.

    Parameters
    ----------
    kwargs : arguments goes to mxnet.gluon.nn.HardSwish
    """
    def __init__(self, **kwargs):
        super(HardSwish, self).__init__(**kwargs)
        self.act = HardSigmoid()

    def hybrid_forward(self, F, x):
        return x * self.act(x)

class SoftmaxHD(HybridBlock):
    """Softmax on multiple dimensions

    Parameters
    ----------
    axis : the axis for softmax normalization
    """
    def __init__(self, axis=(2, 3), **kwargs):
        super(SoftmaxHD, self).__init__(**kwargs)
        self.axis = axis

    def hybrid_forward(self, F, x):
        x_max = F.max(x, axis=self.axis, keepdims=True)
        x_exp = F.exp(F.broadcast_minus(x, x_max))
        norm = F.sum(x_exp, axis=self.axis, keepdims=True)
        res = F.broadcast_div(x_exp, norm)
        return res

class DSNT(HybridBlock):
    '''DSNT module to translate heatmap to coordinates

    Parameters
    ----------
    size : int or tuple,
        (width, height) of the input heatmap
    norm : str, the normalization method for heatmap
        available methods are 'softmax', or 'sum'
    axis : the axis for input heatmap
    '''
    def __init__(self, size, norm='sum', axis=(2, 3), **kwargs):
        super(DSNT, self).__init__(**kwargs)
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.axis = axis
        self.norm = norm
        if self.norm == 'softmax':
            self.softmax = SoftmaxHD(self.axis)
        elif self.norm != 'sum':
            raise ValueError("argument `norm` only accepts 'softmax' or 'sum'.")

        self.wfirst = 1 / (2 * self.size[0])
        self.wlast = 1 - 1 / (2 * self.size[0])
        self.hfirst = 1 / (2 * self.size[1])
        self.hlast = 1 - 1 / (2 * self.size[1])

    def hybrid_forward(self, F, M):
        # pylint: disable=missing-function-docstring
        if self.norm == 'softmax':
            Z = self.softmax(M)
        elif self.norm == 'sum':
            norm = F.sum(M, axis=self.axis, keepdims=True)
            Z = F.broadcast_div(M, norm)
        else:
            Z = M
        x = F.linspace(self.wfirst, self.wlast, self.size[0]).expand_dims(0)
        y = F.linspace(self.hfirst, self.hlast, self.size[1]).expand_dims(0).transpose()
        output_x = F.sum(F.broadcast_mul(Z, x), axis=self.axis)
        output_y = F.sum(F.broadcast_mul(Z, y), axis=self.axis)
        res = F.stack(output_x, output_y, axis=2)
        return res, Z

class DUC(HybridBlock):
    '''Upsampling layer with pixel shuffle
    '''
    def __init__(self, planes, upscale_factor=2, **kwargs):
        super(DUC, self).__init__(**kwargs)
        self.conv = nn.Conv2D(planes, kernel_size=3, padding=1, use_bias=False)
        self.bn = BatchNormCudnnOff(gamma_initializer=initializer.One(),
                                    beta_initializer=initializer.Zero())
        self.relu = nn.Activation('relu')
        self.pixel_shuffle = contrib.nn.PixelShuffle2D(upscale_factor)

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x
