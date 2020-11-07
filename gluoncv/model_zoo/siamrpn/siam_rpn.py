"""siamRPN RPN
Code adapted from https://github.com/STVIR/pysot"""
# pylint: disable=arguments-differ
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet.context import cpu

class RPN(HybridBlock):
    """RPN head"""
    def __init__(self):
        super(RPN, self).__init__()

    def hybrid_forward(self, F, z_f, x_f):
        raise NotImplementedError


class DepthwiseXCorr(HybridBlock):
    """
        SiamRPN RPN after network backbone, regard output of x backbone as kernel,
        and regard output of z backbone as search feature, make Depthwise conv.
        get cls and loc though two streams netwrok

    Parameters
    ----------
        hidden : int
            hidden feature channel.
        out_channels : int
            output feature channel.
        bz : int
            batch size for train, bz = 1 if test.
        is_train : str
            is_train is True if train, False if test.
        kernel_size : float
            hidden kernel size.
        ctx : mxnet.Context
            Context such as mx.cpu(), mx.gpu(0).
    """
    def __init__(self, hidden, out_channels, bz=1, is_train=False, kernel_size=3, ctx=cpu()):
        super(DepthwiseXCorr, self).__init__()
        self.conv_kernel = nn.HybridSequential(prefix='')
        self.conv_search = nn.HybridSequential(prefix='')
        self.head = nn.HybridSequential(prefix='')

        self.conv_kernel.add(nn.Conv2D(hidden, kernel_size=kernel_size, use_bias=False),
                             nn.BatchNorm(),
                             nn.Activation('relu'))
        self.conv_search.add(nn.Conv2D(hidden, kernel_size=kernel_size, use_bias=False),
                             nn.BatchNorm(),
                             nn.Activation('relu'))
        self.head.add(nn.Conv2D(hidden, kernel_size=1, use_bias=False),
                      nn.BatchNorm(),
                      nn.Activation('relu'),
                      nn.Conv2D(out_channels, kernel_size=1))
        if is_train:
            self.kernel_size = [bz, 256, 4, 4]
            self.search_size = [bz, 256, 20, 20]
            self.out_size = [bz, 256, 17, 17]
        else:
            self.kernel_size = [1, 256, 4, 4]
            self.search_size = [1, 256, 24, 24]
            self.out_size = [1, 256, 21, 21]

        self.conv_kernel.initialize(ctx=ctx)
        self.conv_search.initialize(ctx=ctx)
        self.head.initialize(ctx=ctx)

    def hybrid_forward(self, F, kernel, search):
        """hybrid_forward"""
        kernel = self.conv_kernel(kernel)
        search = self.conv_search(search)
        batch = self.kernel_size[0]
        channel = self.kernel_size[1]
        search = search.reshape((1, batch*channel, self.search_size[2], self.search_size[3]))
        kernel = kernel.reshape((batch*channel, 1, self.kernel_size[2], self.kernel_size[3]))
        out = F.Convolution(data=search, weight=kernel,
                            kernel=[self.kernel_size[2], self.kernel_size[3]],
                            no_bias=True, num_filter=batch*channel, num_group=batch*channel)
        out = out.reshape((batch, channel, self.out_size[2], self.out_size[3]))
        out = self.head(out)
        return out

class DepthwiseRPN(RPN):
    """DepthwiseRPN
    get cls and loc throught z_f and x_f

    Parameters
    ----------
        bz : int
            batch size for train, bz = 1 if test.
        is_train : str
            is_train is True if train, False if test.
        ctx : mxnet.Context
            Context such as mx.cpu(), mx.gpu(0).
        anchor_num : int
            number of anchor.
        out_channels : int
            hidden feature channel.
    """
    def __init__(self, bz=1, is_train=False, ctx=cpu(), anchor_num=5, out_channels=256):
        super(DepthwiseRPN, self).__init__()
        self.cls = DepthwiseXCorr(out_channels, 2 * anchor_num, bz=bz, is_train=is_train, ctx=ctx)
        self.loc = DepthwiseXCorr(out_channels, 4 * anchor_num, bz=bz, is_train=is_train, ctx=ctx)

    def hybrid_forward(self, F, z_f, x_f):
        cls = self.cls(z_f, x_f)
        loc = self.loc(z_f, x_f)
        return cls, loc
