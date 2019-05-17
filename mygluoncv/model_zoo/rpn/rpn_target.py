"""Region Proposal Target Generator."""
from __future__ import absolute_import

import numpy as np
import mxnet as mx
from mxnet import autograd, gluon

from ...nn.bbox import BBoxSplit
from ...nn.coder import SigmoidClassEncoder, NumPyNormalizedBoxCenterEncoder


class RPNTargetSampler(object):
    """A sampler to choose positive/negative samples from RPN anchors

    Parameters
    ----------
    num_sample : int
        Number of samples for RCNN targets.
    pos_iou_thresh : float
        Proposal whose IOU larger than ``pos_iou_thresh`` is regarded as positive samples.
    neg_iou_thresh : float
        Proposal whose IOU smaller than ``neg_iou_thresh`` is regarded as negative samples.
    pos_ratio : float
        ``pos_ratio`` defines how many positive samples (``pos_ratio * num_sample``) is
        to be sampled.

    """

    def __init__(self, num_sample, pos_iou_thresh, neg_iou_thresh, pos_ratio):
        super(RPNTargetSampler, self).__init__()
        self._num_sample = num_sample
        self._max_pos = int(round(num_sample * pos_ratio))
        self._pos_iou_thresh = pos_iou_thresh
        self._neg_iou_thresh = neg_iou_thresh
        self._eps = np.spacing(np.float32(1.0))

    def __call__(self, ious):
        """RPNTargetSampler is only used in data transform with no batch dimension.

        Parameters
        ----------
        ious: (N, M) i.e. (num_anchors, num_gt).

        Returns
        -------
        samples: (num_anchors,) value 1: pos, -1: neg, 0: ignore.
        matches: (num_anchors,) value [0, M).

        """
        matches = np.argmax(ious, axis=1)

        # samples init with 0 (ignore)
        ious_max_per_anchor = np.max(ious, axis=1)
        samples = np.zeros_like(ious_max_per_anchor)

        # set argmax (1, num_gt)
        ious_max_per_gt = np.max(ious, axis=0, keepdims=True)
        # ious (num_anchor, num_gt) >= argmax (1, num_gt) -> mark row as positive
        mask = (ious + self._eps) > ious_max_per_gt
        # reduce column (num_anchor, num_gt) -> (num_anchor)
        mask = np.sum(mask, axis=1)
        # row maybe sampled by 2 columns but still only matches to most overlapping gt
        samples = np.where(mask, 1.0, samples)

        # set positive overlap to 1
        samples = np.where(ious_max_per_anchor >= self._pos_iou_thresh, 1.0, samples)
        # set negative overlap to -1
        tmp = (ious_max_per_anchor < self._neg_iou_thresh) * (ious_max_per_anchor >= 0)
        samples = np.where(tmp, -1.0, samples)

        # subsample fg labels
        num_pos = int((samples > 0).sum())
        if num_pos > self._max_pos:
            disable_indices = np.random.choice(
                np.where(samples > 0)[0], size=(num_pos - self._max_pos), replace=False)
            samples[disable_indices] = 0  # use 0 to ignore

        # subsample bg labels
        num_neg = int((samples < 0).sum())
        # if pos_sample is less than quota, we can have negative samples filling the gap
        max_neg = self._num_sample - min(num_pos, self._max_pos)
        if num_neg > max_neg:
            disable_indices = np.random.choice(
                np.where(samples < 0)[0], size=(num_neg - max_neg), replace=False)
            samples[disable_indices] = 0

        return samples, matches


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
        self._sampler = RPNTargetSampler(num_sample, pos_iou_thresh, neg_iou_thresh, pos_ratio)
        self._cls_encoder = SigmoidClassEncoder()
        self._box_encoder = NumPyNormalizedBoxCenterEncoder(stds=stds)

    # pylint: disable=arguments-differ
    def forward(self, bbox, anchor, width, height):
        """
        RPNTargetGenerator is only used in data transform with no batch dimension.
        Be careful there's numpy operations inside

        Parameters
        ----------
        bbox: (M, 4) ground truth boxes with corner encoding.
        anchor: (N, 4) anchor boxes with corner encoding.
        width: int width of input image
        height: int height of input image

        Returns
        -------
        cls_target: (N,) value +1: pos, 0: neg, -1: ignore
        box_target: (N, 4) only anchors whose cls_target > 0 has nonzero box target
        box_mask: (N, 4) only anchors whose cls_target > 0 has nonzero mask

        """
        with autograd.pause():
            # calculate ious between (N, 4) anchors and (M, 4) bbox ground-truths
            # ious is (N, M)
            ious = mx.nd.contrib.box_iou(anchor, bbox, format='corner').asnumpy()

            # mask out invalid anchors, (N, 4)
            a_xmin, a_ymin, a_xmax, a_ymax = mx.nd.split(anchor, 4, axis=-1)
            invalid_mask = (a_xmin < 0) + (a_ymin < 0) + (a_xmax >= width) + (a_ymax >= height)
            ious = np.where(invalid_mask.asnumpy(), -1.0, ious)
            samples, matches = self._sampler(ious)

            # training targets for RPN
            cls_target, _ = self._cls_encoder(samples)
            box_target, box_mask = self._box_encoder(
                np.expand_dims(samples, axis=0), np.expand_dims(matches, axis=0),
                np.expand_dims(anchor.asnumpy(), axis=0), np.expand_dims(bbox.asnumpy(), axis=0))
        return mx.nd.array(cls_target, ctx=bbox.context), \
               mx.nd.array(box_target[0], ctx=bbox.context), \
               mx.nd.array(box_mask[0], ctx=bbox.context)
