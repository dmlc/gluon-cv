"""RCNN Target Generator."""
from __future__ import absolute_import

import numpy as np
import mxnet as mx
from mxnet import gluon
from ...nn.bbox import BBoxSplit
from ...nn.coder import MultiClassEncoder, NormalizedBoxCenterEncoder
from ...utils.nn.matcher import CompositeMatcher, BipartiteMatcher, MaximumMatcher


class RCNNTargetSampler(gluon.HybridBlock):
    def __init__(self, num_sample=128, pos_iou_thresh=0.5, neg_iou_thresh_high=0.5,
                 neg_iou_thresh_low=0.0, pos_ratio=0.25):
        super(RCNNTargetSampler, self).__init__()
        self._num_sample = num_sample
        self._pos_iou_thresh = pos_iou_thresh
        self._neg_iou_thresh_high = neg_iou_thresh_high
        self._neg_iou_thresh_low = neg_iou_thresh_low
        self._pos_ratio = pos_ratio
        self._matcher = CompositeMatcher([BipartiteMatcher(), MaximumMatcher(pos_iou_thresh)])

    def hybrid_forward(self, F, roi, gt_box):
        """
        Only support batch_size=1 now.
        """
        # cocnat rpn roi with ground truths
        all_roi = F.concat(roi.squeeze(axis=0), gt_box.squeeze(axis=0), dim=0)
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
        samples = samples.squeeze(axis=0)   # remove batch axis
        matches = matches.squeeze(axis=0)

        # shuffle and argsort, take first num_sample samples
        sf_samples = F.where(samples == 0, F.ones_like(samples) * -999, samples)
        sf_samples = F.shuffle(sf_samples)
        indices = F.argsort(sf_samples).slice_axis(axis=0, begin=0, end=self._num_sample)
        return all_roi.take(indices), samples.take(indices), matches.take(indices)


class RCNNTargetGenerator(gluon.Block):
    def __init__(self, means=(0., 0., 0., 0.), stds=(.1, .1, .2, .2)):
        super(RCNNTargetGenerator, self).__init__()
        self._cls_encoder = MultiClassEncoder()
        self._box_encoder = NormalizedBoxCenterEncoder(means=means, stds=stds)

    def forward(self, roi, samples, matches, gt_label, gt_box):
        """
        Only support batch_size=1 now.
        """
        cls_target = self._cls_encoder(samples, matches, gt_label)
        box_target, box_mask = self._box_encoder(
            samples, matches, roi, gt_box)
        return cls_target, box_target, box_mask
