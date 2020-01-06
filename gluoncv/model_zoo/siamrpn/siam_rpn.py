from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
import mxnet as mx

class RPN(nn.HybridBlock):
    def __init__(self):
        super(RPN, self).__init__()

    def forward(self, z_f, x_f):
        raise NotImplementedError

class DepthwiseXCorr(HybridBlock):
    def __init__(self,in_channels,hidden,out_channels, kernel_size=3):
        super(DepthwiseXCorr, self).__init__()
        self.conv_kernel = nn.HybridSequential(prefix='')
        self.conv_search = nn.HybridSequential(prefix='')
        self.head = nn.HybridSequential(prefix='')

        self.conv_kernel.add(
                nn.Conv2D(hidden, kernel_size=kernel_size,use_bias=False),
                nn.BatchNorm(),
                nn.Activation('relu')
                )
        self.conv_search.add(
                nn.Conv2D(hidden, kernel_size=kernel_size,use_bias=False),
                nn.BatchNorm(),
                nn.Activation('relu')
                )
        self.head.add(
                nn.Conv2D(hidden, kernel_size=1,use_bias=False),
                nn.BatchNorm(),
                nn.Activation('relu'),
                nn.Conv2D(out_channels, kernel_size=1)
                )
        

    def hybrid_forward(self,F, kernel, search):
        kernel = self.conv_kernel(kernel)#[1, 256, 24, 24]
        search = self.conv_search(search)#[1, 256, 4, 4]
        batch = kernel.shape[0]
        channel = kernel.shape[1]
        w = kernel.shape[2]
        h = kernel.shape[3]
        search = search.reshape(1, batch*channel, search.shape[2], search.shape[3])
        kernel = kernel.reshape(batch*channel, 1, kernel.shape[2], kernel.shape[3])
        out = F.Convolution(data=search, weight=kernel,kernel=[h,w],no_bias=True,num_filter=channel,num_group=batch*channel)
        #print(out.shape)
        out = out.reshape(batch, channel, out.shape[2], out.shape[3])#[1, 256, 21, 21]
        out = self.head(out)
        return out


class DepthwiseRPN(RPN):
    def __init__(self, anchor_num=5, in_channels=256, out_channels=256):
        super(DepthwiseRPN, self).__init__()
        self.cls = DepthwiseXCorr(in_channels, out_channels, 2 * anchor_num)
        self.loc = DepthwiseXCorr(in_channels, out_channels, 4 * anchor_num)

    def forward(self, z_f, x_f):
        cls = self.cls(z_f, x_f)
        loc = self.loc(z_f, x_f)
        cls = self.cls(z_f, x_f)
        return cls, loc

if __name__ == '__main__':
    model = DepthwiseRPN()
    print(model)
    # model.initialize()
    # z_f =  mx.nd.random.normal(shape = [1,256,6,6])
    # x_f =  mx.nd.random.normal(shape = [1,256,26,26])
    # model(z_f,x_f)
    # model.hybridize()