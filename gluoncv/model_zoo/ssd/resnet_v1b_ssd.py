"""ResNet V1b series modified for SSD detection"""
from __future__ import absolute_import

import mxnet as mx
from mxnet.gluon import HybridBlock
from mxnet.gluon.nn import HybridSequential, Conv2D, Activation, BatchNorm
from ..resnetv1b import *
from ..model_zoo import get_model


class ResNetV1bSSD(HybridBlock):
    def __init__(self, name, add_filters,
                 norm_layer=BatchNorm, norm_kwargs=None,
                 batch_norm=False, reduce_ratio=1.0, min_depth=128, **kwargs):
        super(ResNetV1bSSD, self).__init__()
        assert name.endswith('v1b')
        if norm_kwargs is None:
            norm_kwargs = {}
        res = get_model(name, **kwargs)
        weight_init = mx.init.Xavier(rnd_type='gaussian', factor_type='out', magnitude=2)
        with self.name_scope():
            self.stage1 = HybridSequential('stage1')
            for l in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2']:
                self.stage1.add(getattr(res, l))
            self.stage2 = HybridSequential('stage2')
            self.stage2.add(res.layer3)
            # set stride from (2, 2) -> (1, 1) in first conv of layer3
            self.stage2[0][0].conv1._kwargs['stride'] = (1, 1)
            # also the residuel path
            self.stage2[0][0].downsample[0]._kwargs['stride'] = (1, 1)
            self.stage2.add(res.layer4)
            self.more_stages = HybridSequential('more_stages')
            for i, num_filter in enumerate(add_filters):
                stage = HybridSequential('more_stages_' + str(i))
                num_trans = max(min_depth, int(round(num_filter * reduce_ratio)))
                stage.add(Conv2D(channels=num_trans, kernel_size=1, use_bias=not batch_norm,
                                 weight_initializer=weight_init))
                if batch_norm:
                    stage.add(norm_layer(**norm_kwargs))
                stage.add(Activation('relu'))
                padding = 0 if i == len(add_filters) - 1 else 1
                stage.add(Conv2D(channels=num_filter, kernel_size=3,
                                 strides=2, padding=padding, use_bias=not batch_norm,
                                 weight_initializer=weight_init))
                if batch_norm:
                    stage.add(norm_layer(**norm_kwargs))
                stage.add(Activation('relu'))
                self.more_stages.add(stage)

    def hybrid_forward(self, F, x):
        y1 = self.stage1(x)
        y2 = self.stage2(y1)
        more_out = [y1, y2]
        out = y2
        for stage in self.more_stages:
            out = stage(out)
            more_out.append(out)
        return more_out

def resnet34_v1b_ssd(**kwargs):
    return ResNetV1bSSD(name='resnet34_v1b', add_filters=[256, 256, 128, 128], **kwargs)
