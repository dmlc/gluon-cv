# pylint: disable=arguments-differ
"""Samplers for positive/negative/ignore sample selections.
This module is used to select samples during training.
Based on different strategies, we would like to choose different number of
samples as positive, negative or ignore(don't care). The purpose is to alleviate
unbalanced training target in some circumstances.
The output of sampler is an NDArray of the same shape as the matching results.
Note: 1 for positive, -1 for negative, 0 for ignore.
"""
from __future__ import absolute_import
import numpy as np
from mxnet import gluon
from mxnet import nd
from mxnet import autograd


class NaiveSampler(gluon.HybridBlock):
    """A naive sampler that take all existing matching results.
    There is no ignored sample in this case.
    """
    def __init__(self):
        super(NaiveSampler, self).__init__()

    def hybrid_forward(self, F, x):
        """Hybrid forward"""
        marker = F.ones_like(x)
        y = F.where(x >= 0, marker, marker * -1)
        return y


class OHEMSampler(gluon.Block):
    """A sampler implementing Online Hard-negative mining.
    As described in paper https://arxiv.org/abs/1604.03540.

    Parameters
    ----------
    ratio : float
        Ratio of negative vs. positive samples. Values >= 1.0 is recommended.
    min_samples : int, default 0
        Minimum samples to be selected regardless of positive samples.
        For example, if positive samples is 0, we sometimes still want some num_negative
        samples to be selected.
    thresh : float, default 0.5
        IOU overlap threshold of selected negative samples. IOU must not exceed
        this threshold such that good matching anchors won't be selected as
        negative samples.

    """
    def __init__(self, ratio, min_samples=0, thresh=0.5):
        super(OHEMSampler, self).__init__()
        assert ratio > 0, "OHEMSampler ratio must > 0, {} given".format(ratio)
        self._ratio = ratio
        self._min_samples = min_samples
        self._thresh = thresh

    # pylint: disable=arguments-differ
    def forward(self, x, logits, ious):
        """Forward"""
        F = nd
        num_positive = F.sum(x > -1, axis=1)
        num_negative = self._ratio * num_positive
        num_total = x.shape[1]  # scalar
        num_negative = F.minimum(F.maximum(self._min_samples, num_negative),
                                 num_total - num_positive)
        positive = logits.slice_axis(axis=2, begin=1, end=-1)
        background = logits.slice_axis(axis=2, begin=0, end=1).reshape((0, -1))
        maxval = positive.max(axis=2)
        esum = F.exp(logits - maxval.reshape((0, 0, 1))).sum(axis=2)
        score = -F.log(F.exp(background - maxval) / esum)
        mask = F.ones_like(score) * -1
        score = F.where(x < 0, score, mask)  # mask out positive samples
        if len(ious.shape) == 3:
            ious = F.max(ious, axis=2)
        score = F.where(ious < self._thresh, score, mask)  # mask out if iou is large
        argmaxs = F.argsort(score, axis=1, is_ascend=False)

        # neg number is different in each batch, using dynamic numpy operations.
        y = np.zeros(x.shape)
        y[np.where(x.asnumpy() >= 0)] = 1  # assign positive samples
        argmaxs = argmaxs.asnumpy()
        for i, num_neg in zip(range(x.shape[0]), num_negative.asnumpy().astype(np.int32)):
            indices = argmaxs[i, :num_neg]
            y[i, indices.astype(np.int32)] = -1  # assign negative samples
        return F.array(y, ctx=x.context)


class QuotaSampler(autograd.Function):
    def __init__(self, num_sample, pos_thresh, neg_thresh_high, neg_thresh_low=0.,
                 pos_ratio=0.5, neg_ratio=None):
        self._num_sample = num_sample
        if neg_ratio is None:
            self._neg_ratio = neg_ratio
        self._pos_ratio = pos_ratio
        assert (self._neg_ratio + self._pos_ratio) <= 1.0, (
            "Positive and negative ratio exceed 1".format(self._neg_ratio + self._pos_ratio))
        self._pos_thresh = min(1., max(0., pos_thresh))
        self._neg_thresh = min(1., max(0., neg_thresh))

    def forward(self, matches, ious):
        max_pos = self._pos_ratio * self._num_sample
        max_neg = self._neg_ratio * self._num_sample
        # init with 0s, which are ignored
        result = F.zeros_like(matches)
        # negative samples with label -1
        neg_mask = ious.max(axis=1) < self._neg_thresh
        result = F.where(ious.max(axis=1) < self._neg_thresh,
                         F.ones_like(matches) * -1, result)
        # positive samples
        result = F.where(matches >= 0, F.ones_like(result), result)
        result = F.where(ious.max(axis=1) >= self._pos_thresh, F.ones_like(result), result)

        # re-balance if number of postive or negative exceed limits
        result = result.asnumpy()
        num_pos = (result > 0).sum().asscalar()
        if num_pos > max_pos:
            disable_indices = np.random.choice(
                np.where(result > 0)[0], size=(num_pos - max_pos), replace=False)
            result[disable_indices] = 0   # use 0 to ignore
        num_neg = (result < 0).sum().asscalar()
        if num_neg > max_neg:
            disable_indices = np.random.choice(
                np.where(result < 0)[0], size=(num_neg - max_neg), replace=False)

        # some non-related gradients
        g1 = F.zeros_like(matches)
        g2 = F.zeros_like(ious)
        self.save_for_backward(g1, g2)
        return mx.nd.array(result)

    def backward(self, dy):
        g1, g2 = self.saved_tensors
        return g1, g2
