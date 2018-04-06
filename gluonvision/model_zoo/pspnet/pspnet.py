import numpy as np
import mxnet as mx
from mxnet import nd, init
from mxnet.gluon import nn
import mxnet.ndarray as F
from mxnet.gluon.nn import HybridBlock

from .. import dilatedresnet as resnet
from ..segbase import SegBaseModel

class PSPNet(SegBaseModel):
    r"""Pyramid Scene Parsing Network

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50', 'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    aux : bool
        Auxilary loss.


    Reference:

        Zhao, Hengshuang, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, and Jiaya Jia. "Pyramid scene parsing network." *CVPR*, 2017

    """
    def __init__(self, nclass, backbone='resnet50', norm_layer=nn.BatchNorm, aux=True):
        super(PSPNet, self).__init__(backbone, norm_layer)
        self.aux = aux
        with self.name_scope():
            self.head = _PSPHead(nclass)
        self.head.initialize(init=init.Xavier())

    # def hybrid_forward(self, F, x):
    def forward(self, x):
        B,C,H,W = x.shape
        x = self.pretrained(x)
        x = self.head(x)
        x = F.contrib.BilinearResize2D(x, height=H, width=W)
        return x


def _PSP1x1Conv(in_channels, out_channels):
    block = nn.HybridSequential(prefix='')
    with block.name_scope():
        block.add(nn.BatchNorm(in_channels=in_channels))
        block.add(nn.Activation('relu'))
        block.add(nn.Conv2D(in_channels=in_channels,
                            channels=out_channels, kernel_size=1))
    return block


class _PyramidPooling(HybridBlock):
    def __init__(self, in_channels):
        super(_PyramidPooling, self).__init__()
        out_channels = int(in_channels/4)
        with self.name_scope():
            self.conv1 = _PSP1x1Conv(in_channels, out_channels)
            self.conv2 = _PSP1x1Conv(in_channels, out_channels)
            self.conv3 = _PSP1x1Conv(in_channels, out_channels)
            self.conv4 = _PSP1x1Conv(in_channels, out_channels)

    def pool(self, x, size):
        return F.contrib.AdaptiveAvgPooling2D(x, output_size=size)

    def upsample(self, x, h, w):
        return F.contrib.BilinearResize2D(x, height=h, width=w)

    def hybrid_forward(self, F, x):
        _, _, h, w = x.shape
        feat1 = self.upsample(self.conv1(self.pool(x, 1)),h,w)
        feat2 = self.upsample(self.conv2(self.pool(x, 2)),h,w)
        feat3 = self.upsample(self.conv3(self.pool(x, 3)),h,w)
        feat4 = self.upsample(self.conv4(self.pool(x, 4)),h,w)
        return F.concat(x, feat1, feat2, feat3, feat4, dim=1)


class _PSPHead(HybridBlock):
    def __init__(self, nclass):
        super(_PSPHead, self).__init__()
        self.psp = _PyramidPooling(2048)
        with self.name_scope():
            self.block = nn.HybridSequential(prefix='')
            self.block.add(nn.BatchNorm(in_channels=4096))
            self.block.add(nn.Activation('relu'))
            self.block.add(nn.Conv2D(in_channels=4096, channels=512,
                           kernel_size=3, padding=1))
            self.block.add(nn.BatchNorm(in_channels=512))
            self.block.add(nn.Activation('relu'))
            self.block.add(nn.Dropout(0.1))
            self.block.add(nn.Conv2D(in_channels=512, channels=nclass,
                           kernel_size=1))
    
    def hybrid_forward(self, F, x):
        x = self.psp(x)
        return self.block(x)

# acronym for easy load
class _Net(PSPNet):
    def __init__(self, args):
        super(_Net, self).__init__(args.nclass, args.backbone, args.norm_layer, args.aux)
