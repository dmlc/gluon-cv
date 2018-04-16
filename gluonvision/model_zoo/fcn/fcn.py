"""Fully Convolutional Network with Strdie of 8"""
from __future__ import division
from mxnet.gluon import nn
import mxnet.ndarray as F
from mxnet.gluon.nn import HybridBlock

from ..segbase import SegBaseModel
# pylint: disable=unused-argument,abstract-method

class FCN(SegBaseModel):
    r"""Fully Convolutional Networks for Semantic Segmentation

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;


    Reference:

        Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks
        for semantic segmentation." *CVPR*, 2015
    """
    # pylint: disable=arguments-differ
    def __init__(self, nclass, backbone='resnet50', norm_layer=nn.BatchNorm,
                 aux=True, **kwargs):
        super(FCN, self).__init__(aux, backbone, norm_layer=norm_layer, **kwargs)
        self.nclass = nclass
        with self.name_scope():
            self.head = _FCNHead(2048, nclass, norm_layer=norm_layer, **kwargs)
            self.head.initialize()
            self.head.collect_params().setattr('lr_mult', 10)
            if self.aux:
                self.auxlayer = _FCNHead(1024, nclass, norm_layer=norm_layer, **kwargs)
                self.auxlayer.initialize()
                self.auxlayer.collect_params().setattr('lr_mult', 10)

    def forward(self, x):
        _, _, H, W = x.shape
        c3, c4 = self.base_forward(x)

        outputs = []
        x = self.head(c4)
        x = F.contrib.BilinearResize2D(x, height=H, width=W)
        outputs.append(x)

        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.contrib.BilinearResize2D(auxout, height=H, width=W)
            outputs.append(auxout)
            return tuple(outputs)
        else:
            return x


class _FCNHead(HybridBlock):
    # pylint: disable=redefined-outer-name
    def __init__(self, in_channels, channels, norm_layer, **kwargs):
        super(_FCNHead, self).__init__()
        with self.name_scope():
            self.block = nn.HybridSequential()
            inter_channels = in_channels // 4
            with self.block.name_scope():
                self.block.add(nn.Conv2D(in_channels=in_channels, channels=inter_channels,
                                         kernel_size=3, padding=1))
                self.block.add(norm_layer(in_channels=inter_channels))
                self.block.add(nn.Activation('relu'))
                self.block.add(nn.Dropout(0.1))
                self.block.add(nn.Conv2D(in_channels=inter_channels, channels=channels,
                                         kernel_size=1))

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, x):
        return self.block(x)

# acronym for easy load
_Net = FCN
