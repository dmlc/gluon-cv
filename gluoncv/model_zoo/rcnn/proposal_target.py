"""Proposal Target Generation."""
from __future__ import absolute_import

import mxnet as mx
from mxnet import gluon
import numpy as np
from ...nn.coder import NormalizedBoxCenterEncoder, MultiClassEncoder
from ...utils.nn.matcher import BipartiteMatcher, MaximumMatcher
from ...utils.nn.sampler import QuotaSampler


class ProposalTargetGenerator(gluon.Block):
    def __init__(self, num_sample, pos_iou_thresh, neg_iou_thresh_high,
                 neg_iou_thresh_low, pos_ratio,
                 means=(0., 0., 0., 0.), stds=(0.1, 0.1, 0.2, 0.2)):
        self._num_sample = num_sample
        # self._pos_ratio = pos_ratio
        # self._pos_iou_thresh = pos_iou_thresh
        # self._neg_iou_thresh_high = neg_iou_thresh_high
        # self._neg_iou_thresh_low = neg_iou_thresh_low
        self._box_encoder = NormalizedBoxCenterEncoder(means=means, stds=stds)
        self._cls_encoder = MultiClassEncoder(ignore_label=-1)
        self._matcher = CompositeMatcher([BipartiteMatcher(), MaximumMatcher(pos_iou_thresh)])
        self._sampler = QuotaSampler(
            num_sample, pos_iou_thresh, neg_iou_thresh_high, neg_iou_thresh_low, pos_ratio)

    def forward(self, bbox, label, roi):
        """

        """
        F = mx.nd
        # concat proposal rois with ground-truth boxes
        roi = F.concat(*[roi, bbox], dim=0)

        pos_roi_per_image = np.round(self._num_sample * self._pos_ratio)
        # roi: (N, 4), bbox: (M, 4),  iou: (N, M)
        iou = F.contrib.box_iou(bbox, roi, format='corner')
        matches = self._matcher(ious)
        samples = self._sampler(matches, ious)
        # training targets for RCNN
        cls_target = self._cls_encoder(samples)
        box_target, box_mask = self._box_encoder(samples, matches, roi, bbox)
        return cls_target, box_target, box_mask
