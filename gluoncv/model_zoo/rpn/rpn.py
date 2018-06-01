"""Region Proposal Networks Definition."""
from __future__ import absolute_import

import mxnet as mx
from mxnet import gluon
from mxnet import autograd
from mxnet.gluon import nn
from .anchor import RPNAnchorGenerator
from .proposal import RPNProposal


class RPN(gluon.HybridBlock):
    def __init__(self, channels, stride, base_size=16, ratios=(0.5, 1, 2),
                 scales=(8, 16, 32), alloc_size=(128, 128),
                 nms_thresh=0.7, train_pre_nms=12000, train_post_nms=2000,
                 test_pre_nms=6000, test_post_nms=300, min_size=16, stds=(1., 1., 1., 1.),
                 weight_initializer=None, max_batch=32, **kwargs):
        super(RPN, self).__init__(**kwargs)
        if weight_initializer is None:
            weight_initializer = mx.init.Normal(0.1)
        with self.name_scope():
            self.anchor_generator = RPNAnchorGenerator(
                stride, base_size, ratios, scales, alloc_size)
            anchor_depth = self.anchor_generator.num_depth
            self.region_proposaler = RPNProposal(
                nms_thresh, train_pre_nms, train_post_nms,
                test_pre_nms, test_post_nms, min_size, stds, max_batch=max_batch)
            self.conv1 = nn.HybridSequential()
            self.conv1.add(
                nn.Conv2D(channels, 3, 1, 1, weight_initializer=weight_initializer))
            self.conv1.add(nn.Activation('relu'))
            # use sigmoid instead of softmax, reduce channel numbers
            self.score = nn.Conv2D(anchor_depth, 1, 1, 0, weight_initializer=weight_initializer)
            self.loc = nn.Conv2D(anchor_depth * 4, 1, 1, 0, weight_initializer=weight_initializer)

    def hybrid_forward(self, F, x, img):
        anchors = self.anchor_generator(x)
        x = self.conv1(x)
        rpn_scores = F.sigmoid(self.score(x)).transpose(axes=(0, 2, 3, 1)).reshape((0, -1, 1))
        rpn_box_pred = self.loc(x).transpose(axes=(0, 2, 3, 1)).reshape((0, -1, 4))
        rpn_score, rpn_box, batch_roi, roi = self.region_proposaler(
            anchors, rpn_scores, rpn_box_pred, img)
        if autograd.is_training():
            # return raw predictions as well in training for bp
            return rpn_score, rpn_box, batch_roi, roi, rpn_scores, rpn_box_pred
        return rpn_score, rpn_box, batch_roi, roi
