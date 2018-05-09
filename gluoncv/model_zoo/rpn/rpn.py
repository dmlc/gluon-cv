"""Region Proposal Networks Definition."""
from __future__ import absolute_import

import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from .anchor import RPNAnchorGenerator


class RPN(gluon.HybridBlock):
    def __init__(self, channels, stride, base_size=16, ratios=(0.5, 1, 2),
                 scales=(8, 16, 32), alloc_size=(128, 128),
                 nms_thresh, train_pre_nms, train_post_nms,
                 test_pre_nms, test_post_nms, min_size,
                 weight_initializer=None, **kwargs):
        super(RPN, self).__init__(**kwargs)
        if weight_initializer is None:
            weight_initializer = mx.init.Normal(0.1)
        with self.name_scope():
            self.anchor_generator = RPNAnchorGenerator(
                stride, base_size, ratios, scales, alloc_size)
            self.conv1 = nn.Conv2D(channels, 3, 1, 1, weight_initializer=weight_initializer)
