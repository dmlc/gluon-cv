import numpy as np
import mxnet as mx
from mxnet import nd, init
import mxnet.ndarray as F
from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock

from . import dilatedresnet

class SegBaseModel(HybridBlock):
    r"""Base Model for Semantic Segmentation

    Parameters
    ----------
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50', 'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    """
    def __init__(self, backbone='resnet50', norm_layer=nn.BatchNorm):
        super(SegBaseModel, self).__init__()
        if backbone == 'resnet50':
            pretrained = dilatedresnet.dilated_resnet50(pretrained=True, norm_layer=norm_layer)
        elif backbone == 'resnet101':
            pretrained = dilatedresnet.dilated_resnet101(pretrained=True, norm_layer=norm_layer)
        elif backbone == 'resnet152':
            pretrained = dilatedresnet.dilated_resnet152(pretrained=True, norm_layer=norm_layer)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

        with self.name_scope():
            self.pretrained = nn.HybridSequential(prefix='')
            for layer in pretrained.features:
                self.pretrained.add(layer)

    def forward(self, x):
        pass
