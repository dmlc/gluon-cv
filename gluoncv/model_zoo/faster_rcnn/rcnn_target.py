"""RCNN Target Generator."""
from __future__ import absolute_import

from mxnet import gluon
from mxnet import autograd
from ...nn.coder import MultiClassEncoder, NormalizedPerClassBoxCenterEncoder
from ...nn.matcher import MaximumMatcher


class RCNNTargetSampler(gluon.HybridBlock):
    """A sampler to choose positive/negative samples from RCNN Proposals

    Parameters
    ----------
    num_sample : int, default is 128
        Number of samples for RCNN targets.
    pos_iou_thresh : float, default is 0.5
        Proposal whose IOU larger than ``pos_iou_thresh`` is regarded as positive samples.
    neg_iou_thresh_high : float, default is 0.5
        Proposal whose IOU smaller than ``neg_iou_thresh_high``
        and larger than ``neg_iou_thresh_low``
        is regarded as negative samples.
        Proposals with IOU in between ``pos_iou_thresh`` and ``neg_iou_thresh`` are
        ignored.
    neg_iou_thresh_low : float, default is 0.0
        See ``neg_iou_thresh_high``.
    pos_ratio : float, default is 0.25
        ``pos_ratio`` defines how many positive samples (``pos_ratio * num_sample``) is
        to be sampled.

    """
    def __init__(self, num_sample=128, pos_iou_thresh=0.5, neg_iou_thresh_high=0.5,
                 neg_iou_thresh_low=0.0, pos_ratio=0.25):
        super(RCNNTargetSampler, self).__init__()
        self._num_sample = num_sample
        self._pos_iou_thresh = pos_iou_thresh
        self._neg_iou_thresh_high = neg_iou_thresh_high
        self._neg_iou_thresh_low = neg_iou_thresh_low
        self._pos_ratio = pos_ratio
        self._matcher = MaximumMatcher(pos_iou_thresh)

    #pylint: disable=arguments-differ
    def hybrid_forward(self, F, roi, gt_box):
        """
        Only support batch_size=1 now.
        """
        with autograd.pause():
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
            indices = F.argsort(sf_samples, is_ascend=False).slice_axis(
                axis=0, begin=0, end=self._num_sample)
            new_roi = all_roi.take(indices).expand_dims(0)
            new_samples = samples.take(indices).expand_dims(0)
            new_matches = matches.take(indices).expand_dims(0)
        return new_roi, new_samples, new_matches


class RCNNTargetGenerator(gluon.Block):
    """RCNN target encoder to generate matching target and regression target values.

    Parameters
    ----------
    num_class : int
        Number of total number of positive classes.
    means : iterable of float, default is (0., 0., 0., 0.)
        Mean values to be subtracted from regression targets.
    stds : iterable of float, default is (.1, .1, .2, .2)
        Standard deviations to be divided from regression targets.

    """
    def __init__(self, num_class, means=(0., 0., 0., 0.), stds=(.1, .1, .2, .2)):
        super(RCNNTargetGenerator, self).__init__()
        self._cls_encoder = MultiClassEncoder()
        self._box_encoder = NormalizedPerClassBoxCenterEncoder(
            num_class=num_class, means=means, stds=stds)

    #pylint: disable=arguments-differ
    def forward(self, roi, samples, matches, gt_label, gt_box):
        """
        Only support batch_size=1 now.
        """
        with autograd.pause():
            cls_target = self._cls_encoder(samples, matches, gt_label)
            box_target, box_mask = self._box_encoder(
                samples, matches, roi, gt_label, gt_box)
            # modify shapes to match predictions
            cls_target = cls_target[0]
            box_target = box_target.transpose((1, 2, 0, 3))[0]
            box_mask = box_mask.transpose((1, 2, 0, 3))[0]
        return cls_target, box_target, box_mask
