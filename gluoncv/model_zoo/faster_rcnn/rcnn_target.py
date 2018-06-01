"""RCNN Target Generator."""
from __future__ import absolute_import

import numpy as np
import mxnet as mx
from mxnet import gluon
from ...nn.bbox import BBoxSplit
from ...nn.coder import MultiClassEncoder, NormalizedBoxCenterEncoderV1
from ...utils.nn.matcher import CompositeMatcher, BipartiteMatcher, MaximumMatcher


class RCNNTargetGenerator(gluon.HybridBlock):
    def __init__(self, num_sample=128, pos_iou_thresh=0.5, neg_iou_thresh_high=0.5,
                 neg_iou_thresh_low=0.0, pos_ratio=0.25, means=(0., 0., 0., 0.),
                 stds=(.1, .1, .2, .2)):
        super(RCNNTargetGenerator, self).__init__()
        self._num_sample = num_sample
        self._pos_iou_thresh = pos_iou_thresh
        self._neg_iou_thresh_high = neg_iou_thresh_high
        self._neg_iou_thresh_low = neg_iou_thresh_low
        self._pos_ratio = pos_ratio
        # self._bbox_split = BBoxSplit(axis=-1)
        self._matcher = CompositeMatcher([BipartiteMatcher(), MaximumMatcher(pos_iou_thresh)])
        # self._sampler = QuotaSampler(num_sample, pos_iou_thresh, neg_iou_thresh, 0., pos_ratio)
        self._cls_encoder = MultiClassEncoder()
        self._box_encoder = NormalizedBoxCenterEncoderV1(means=means, stds=stds)

    def hybrid_forward(self, F, roi, gt_label, gt_box):
        """
        Only support batch_size=1 now.
        """
        # cocnat rpn roi with ground truths
        all_roi = F.concat(roi, gt_box.squeeze(axis=0), dim=0)
        # calculate ious between (N, 4) anchors and (M, 4) bbox ground-truths
        # ious is (N, M)
        ious = F.contrib.box_iou(all_roi, gt_box, format='corner').transpose((1, 0, 2))
        matches = self._matcher(ious)
        samples = F.Custom(matches, ious, op_type='quota_sampler',
                             num_sample=self._num_sample,
                             pos_thresh=self._pos_iou_thresh,
                             neg_thresh_high=self._neg_iou_thresh_high,
                             neg_thresh_low=self._neg_iou_thresh_low,
                             pos_ratio=self._pos_ratio)
        # samples = self._sampler(matches, ious)[0]
        # training targets for RPN
        cls_target = self._cls_encoder(samples, matches, gt_label)
        box_target, box_mask = self._box_encoder(
            samples, matches, all_roi, gt_box)
        return cls_target, box_target, box_mask
