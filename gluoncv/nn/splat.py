# pylint: disable=arguments-differ,line-too-long,missing-docstring,missing-module-docstring
from mxnet.gluon import nn
from mxnet.gluon.nn import Conv2D, HybridBlock, BatchNorm, Activation

__all__ = ['SplitAttentionConv']


class SplitAttentionConv(HybridBlock):
    # pylint: disable=keyword-arg-before-vararg
    def __init__(self, channels, kernel_size, strides=(1, 1), padding=(0, 0),
                 dilation=(1, 1), groups=1, radix=2, in_channels=None, r=2,
                 norm_layer=BatchNorm, norm_kwargs=None, drop_ratio=0,
                 *args, **kwargs):
        super(SplitAttentionConv, self).__init__()
        norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
        inter_channels = max(in_channels*radix//2//r, 32)
        self.radix = radix
        self.cardinality = groups
        self.conv = Conv2D(channels*radix, kernel_size, strides, padding, dilation,
                           groups=groups*radix, *args, in_channels=in_channels, **kwargs)
        self.use_bn = norm_layer is not None
        if self.use_bn:
            self.bn = norm_layer(in_channels=channels*radix, **norm_kwargs)
        self.relu = Activation('relu')
        self.fc1 = Conv2D(inter_channels, 1, in_channels=channels, groups=self.cardinality)
        if self.use_bn:
            self.bn1 = norm_layer(in_channels=inter_channels, **norm_kwargs)
        self.relu1 = Activation('relu')
        if drop_ratio > 0:
            self.drop = nn.Dropout(drop_ratio)
        else:
            self.drop = None
        self.fc2 = Conv2D(channels*radix, 1, in_channels=inter_channels, groups=self.cardinality)
        self.channels = channels

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.relu(x)
        if self.radix > 1:
            splited = F.reshape(x.expand_dims(1), (0, self.radix, self.channels, 0, 0))
            gap = F.sum(splited, axis=1)
        else:
            gap = x
        gap = F.contrib.AdaptiveAvgPooling2D(gap, 1)
        gap = self.fc1(gap)
        if self.use_bn:
            gap = self.bn1(gap)
        atten = self.relu1(gap)
        if self.drop:
            atten = self.drop(atten)
        atten = self.fc2(atten).reshape((0, self.cardinality, self.radix, -1)).swapaxes(1, 2)
        if self.radix > 1:
            atten = F.softmax(atten, axis=1).reshape((0, self.radix, -1, 1, 1))
        else:
            atten = F.sigmoid(atten).reshape((0, -1, 1, 1))
        if self.radix > 1:
            outs = F.broadcast_mul(atten, splited)
            out = F.sum(outs, axis=1)
        else:
            out = F.broadcast_mul(atten, x)
        return out
