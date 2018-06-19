"""Region Proposal Target Generator."""
from __future__ import absolute_import

import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet import autograd
from ...nn.bbox import BBoxSplit
from ...nn.coder import SigmoidClassEncoder, NormalizedBoxCenterEncoder
from ...nn.matcher import CompositeMatcher, BipartiteMatcher, MaximumMatcher
from ...nn.sampler import QuotaSampler


class RPNTargetGenerator(gluon.Block):
    """RPN target generator network.

    Parameters
    ----------
    num_sample : int, default is 256
        Number of samples for RPN targets.
    pos_iou_thresh : float, default is 0.7
        Anchor with IOU larger than ``pos_iou_thresh`` is regarded as positive samples.
    neg_iou_thresh : float, default is 0.3
        Anchor with IOU smaller than ``neg_iou_thresh`` is regarded as negative samples.
        Anchors with IOU in between ``pos_iou_thresh`` and ``neg_iou_thresh`` are
        ignored.
    pos_ratio : float, default is 0.5
        ``pos_ratio`` defines how many positive samples (``pos_ratio * num_sample``) is
        to be sampled.
    stds : array-like of size 4, default is (1., 1., 1., 1.)
        Std value to be divided from encoded regression targets.
    allowed_border : int or float, default is 0
        The allowed distance of anchors which are off the image border. This is used to clip out of
        border anchors. You can set it to very large value to keep all anchors.

    """
    def __init__(self, num_sample=256, pos_iou_thresh=0.7, neg_iou_thresh=0.3,
                 pos_ratio=0.5, stds=(1., 1., 1., 1.), allowed_border=0):
        super(RPNTargetGenerator, self).__init__()
        self._num_sample = num_sample
        self._pos_iou_thresh = pos_iou_thresh
        self._neg_iou_thresh = neg_iou_thresh
        self._pos_ratio = pos_ratio
        self._allowed_border = allowed_border
        self._bbox_split = BBoxSplit(axis=-1)
        self._matcher = CompositeMatcher([BipartiteMatcher(), MaximumMatcher(pos_iou_thresh)])
        self._sampler = QuotaSampler(num_sample, pos_iou_thresh, neg_iou_thresh, 0., pos_ratio)
        self._cls_encoder = SigmoidClassEncoder()
        self._box_encoder = NormalizedBoxCenterEncoder(stds=stds)

    # pylint: disable=arguments-differ
    def forward(self, bbox, anchor, width, height):
        """
        Only support batch_size=1 now.
        Be careful there's numpy operations inside
        """
        F = mx.nd
        with autograd.pause():
            # anchor with shape (N, 4)
            a_xmin, a_ymin, a_xmax, a_ymax = self._bbox_split(anchor)
            # invalid anchor mask with shape (N, 1)
            imask = (
                (a_xmin >= -self._allowed_border) *
                (a_ymin >= -self._allowed_border) *
                (a_xmax <= (width + self._allowed_border)) *
                (a_ymax <= (height + self._allowed_border))) <= 0
            imask = mx.nd.array(np.where(imask.asnumpy() > 0)[0], ctx=anchor.context)

            # calculate ious between (N, 4) anchors and (M, 4) bbox ground-truths
            # ious is (N, M)
            ious = F.contrib.box_iou(anchor, bbox, format='corner').transpose((1, 0, 2))
            ious[:, imask, :] = -1
            matches = self._matcher(ious)
            samples = self._sampler(matches, ious)

            # training targets for RPN
            cls_target, _ = self._cls_encoder(samples)
            box_target, box_mask = self._box_encoder(
                samples, matches, anchor.expand_dims(axis=0), bbox)
        return cls_target, box_target, box_mask
