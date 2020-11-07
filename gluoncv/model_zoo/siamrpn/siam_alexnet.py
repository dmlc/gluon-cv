"""Alexnet, implemented in Gluon.
Code adapted from https://github.com/STVIR/pysot"""
# coding: utf-8
# pylint: disable=arguments-differ,unused-argument
from __future__ import division
from mxnet.gluon import nn
from mxnet.gluon.block import HybridBlock
from mxnet.context import cpu

class AlexNetLegacy(HybridBlock):
    """AlexNetLegacy model as backbone"""
    configs = [3, 96, 256, 384, 384, 256]
    def __init__(self, width_mult=1, ctx=cpu(), **kwargs):
        configs = list(map(lambda x: 3 if x == 3 else
                           int(x*width_mult), AlexNetLegacy.configs))
        super(AlexNetLegacy, self).__init__(**kwargs)
        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            with self.features.name_scope():
                self.features.add(nn.Conv2D(configs[1], kernel_size=11, strides=2),
                                  nn.BatchNorm(),
                                  nn.MaxPool2D(pool_size=3, strides=2),
                                  nn.Activation('relu'))
                self.features.add(nn.Conv2D(configs[2], kernel_size=5),
                                  nn.BatchNorm(),
                                  nn.MaxPool2D(pool_size=3, strides=2),
                                  nn.Activation('relu'))
                self.features.add(nn.Conv2D(configs[3], kernel_size=3),
                                  nn.BatchNorm(),
                                  nn.Activation('relu'))
                self.features.add(nn.Conv2D(configs[4], kernel_size=3),
                                  nn.BatchNorm(),
                                  nn.Activation('relu'))
                self.features.add(nn.Conv2D(configs[5], kernel_size=3),
                                  nn.BatchNorm())
            self.features.initialize(ctx=ctx)

    def hybrid_forward(self, F, x):
        x = self.features(x)
        return x

def alexnetlegacy(**kwargs):
    """Alexnetlegacy """
    return AlexNetLegacy(**kwargs)
