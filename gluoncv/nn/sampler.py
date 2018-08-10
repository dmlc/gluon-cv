# pylint: disable=arguments-differ, unused-argument
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
import mxnet as mx
from mxnet import gluon
from mxnet import nd


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


class QuotaSampler(gluon.Block):
    """Sampler that handles limited quota for positive and negative samples.

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
    neg_ratio : float or None
        ``neg_ratio`` defines how many negative samples (``pos_ratio * num_sample``) is
        to be sampled. If ``None`` is provided, it equals to ``1 - pos_ratio``.
    fill_negative : bool
        If ``True``, negative samples will fill the gap caused by insufficient positive samples.
        For example, if ``num_sample`` is 100, ``pos_ratio`` and ``neg_ratio`` are both ``0.5``.
        Available positive sample and negative samples are 10 and 10000, which are typical values.
        Now, the output positive samples is 10(intact), since it's smaller than ``50(100 * 0.5)``,
        the negative samples will fill the rest ``40`` slots.
        If ``fill_negative == False``, the ``40`` slots is filled with ``-1(ignore)``.

    """
    def __init__(self, num_sample, pos_thresh, neg_thresh_high, neg_thresh_low=-np.inf,
                 pos_ratio=0.5, neg_ratio=None, fill_negative=True):
        super(QuotaSampler, self).__init__()
        self._fill_negative = fill_negative
        self._num_sample = num_sample
        if neg_ratio is None:
            self._neg_ratio = 1. - pos_ratio
        self._pos_ratio = pos_ratio
        assert (self._neg_ratio + self._pos_ratio) <= 1.0, (
            "Positive and negative ratio {} exceed 1".format(self._neg_ratio + self._pos_ratio))
        self._pos_thresh = min(1., max(0., pos_thresh))
        self._neg_thresh_high = min(1., max(0., neg_thresh_high))
        self._neg_thresh_low = neg_thresh_low

    def forward(self, matches, ious):
        """Quota Sampler

        Parameters:
        ----------
        matches : NDArray or Symbol
            Matching results, postive number for postive matching, -1 for not matched.
        ious : NDArray or Symbol
            IOU overlaps with shape (N, M), batching is supported.

        Returns:
        --------
        NDArray or Symbol
            Sampling results with same shape as ``matches``.
            1 for positive, -1 for negative, 0 for ignore.

        """
        F = mx.nd
        max_pos = int(round(self._pos_ratio * self._num_sample))
        max_neg = int(self._neg_ratio * self._num_sample)
        results = []
        for i in range(matches.shape[0]):
            # init with 0s, which are ignored
            result = F.zeros_like(matches[0])
            # positive samples
            ious_max = ious.max(axis=-1)[i]
            result = F.where(matches[i] >= 0, F.ones_like(result), result)
            result = F.where(ious_max >= self._pos_thresh, F.ones_like(result), result)
            # negative samples with label -1
            neg_mask = ious_max < self._neg_thresh_high
            neg_mask = neg_mask * (ious_max >= self._neg_thresh_low)
            result = F.where(neg_mask, F.ones_like(result) * -1, result)

            # re-balance if number of postive or negative exceed limits
            result = result.asnumpy()
            num_pos = int((result > 0).sum())
            if num_pos > max_pos:
                disable_indices = np.random.choice(
                    np.where(result > 0)[0], size=(num_pos - max_pos), replace=False)
                result[disable_indices] = 0   # use 0 to ignore
            num_neg = int((result < 0).sum())
            if self._fill_negative:
                # if pos_sample is less than quota, we can have negative samples filling the gap
                max_neg = max(self._num_sample - min(num_pos, max_pos), max_neg)
            if num_neg > max_neg:
                disable_indices = np.random.choice(
                    np.where(result < 0)[0], size=(num_neg - max_neg), replace=False)
                result[disable_indices] = 0
            results.append(mx.nd.array(result))

        return mx.nd.stack(*results, axis=0)


class QuotaSamplerOp(mx.operator.CustomOp):
    """Sampler that handles limited quota for positive and negative samples.

    This is a custom Operator used inside HybridBlock.

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
    neg_ratio : float or None
        ``neg_ratio`` defines how many negative samples (``pos_ratio * num_sample``) is
        to be sampled. If ``None`` is provided, it equals to ``1 - pos_ratio``.
    fill_negative : bool
        If ``True``, negative samples will fill the gap caused by insufficient positive samples.
        For example, if ``num_sample`` is 100, ``pos_ratio`` and ``neg_ratio`` are both ``0.5``.
        Available positive sample and negative samples are 10 and 10000, which are typical values.
        Now, the output positive samples is 10(intact), since it's smaller than ``50(100 * 0.5)``,
        the negative samples will fill the rest ``40`` slots.
        If ``fill_negative == False``, the ``40`` slots is filled with ``-1(ignore)``.

    """
    def __init__(self, num_sample, pos_thresh, neg_thresh_high=0.5, neg_thresh_low=-np.inf,
                 pos_ratio=0.5, neg_ratio=None, fill_negative=True):
        super(QuotaSamplerOp, self).__init__()
        self._num_sample = num_sample
        self._fill_negative = fill_negative
        if neg_ratio is None:
            self._neg_ratio = 1. - pos_ratio
        self._pos_ratio = pos_ratio
        assert (self._neg_ratio + self._pos_ratio) <= 1.0, (
            "Positive and negative ratio {} exceed 1".format(self._neg_ratio + self._pos_ratio))
        self._pos_thresh = min(1., max(0., pos_thresh))
        self._neg_thresh_high = min(1., max(0., neg_thresh_high))
        self._neg_thresh_low = neg_thresh_low

    def forward(self, is_train, req, in_data, out_data, aux):
        """Quota Sampler

        Parameters:
        ----------
        in_data: array-like of Symbol
            [matches, ious], see below.
        matches : NDArray or Symbol
            Matching results, postive number for postive matching, -1 for not matched.
        ious : NDArray or Symbol
            IOU overlaps with shape (N, M), batching is supported.

        Returns:
        --------
        NDArray or Symbol
            Sampling results with same shape as ``matches``.
            1 for positive, -1 for negative, 0 for ignore.

        """
        matches = in_data[0]
        ious = in_data[1]
        F = mx.nd
        max_pos = int(round(self._pos_ratio * self._num_sample))
        max_neg = int(self._neg_ratio * self._num_sample)
        for i in range(matches.shape[0]):
            # init with 0s, which are ignored
            result = F.zeros_like(matches[i])
            # negative samples with label -1
            ious_max = ious.max(axis=-1)[i]
            neg_mask = ious_max < self._neg_thresh_high
            neg_mask = neg_mask * (ious_max >= self._neg_thresh_low)
            result = F.where(neg_mask, F.ones_like(result) * -1, result)
            # positive samples
            result = F.where(matches[i] >= 0, F.ones_like(result), result)
            result = F.where(ious_max >= self._pos_thresh, F.ones_like(result), result)

            # re-balance if number of postive or negative exceed limits
            result = result.asnumpy()
            num_pos = int((result > 0).sum())
            if num_pos > max_pos:
                disable_indices = np.random.choice(
                    np.where(result > 0)[0], size=(num_pos - max_pos), replace=False)
                result[disable_indices] = 0   # use 0 to ignore
            num_neg = int((result < 0).sum())
            if self._fill_negative:
                # if pos_sample is less than quota, we can have negative samples filling the gap
                max_neg = max(self._num_sample - min(num_pos, max_pos), max_neg)
            if num_neg > max_neg:
                disable_indices = np.random.choice(
                    np.where(result < 0)[0], size=(num_neg - max_neg), replace=False)
                result[disable_indices] = 0  # use 0 to ignore

            self.assign(out_data[0][i], req[0], mx.nd.array(result))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)


@mx.operator.register('quota_sampler')
class QuotaSamplerProp(mx.operator.CustomOpProp):
    """Property for QuotaSampleOp.

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
    neg_ratio : float or None
        ``neg_ratio`` defines how many negative samples (``pos_ratio * num_sample``) is
        to be sampled. If ``None`` is provided, it equals to ``1 - pos_ratio``.
    fill_negative : bool
        If ``True``, negative samples will fill the gap caused by insufficient positive samples.
        For example, if ``num_sample`` is 100, ``pos_ratio`` and ``neg_ratio`` are both ``0.5``.
        Available positive sample and negative samples are 10 and 10000, which are typical values.
        Now, the output positive samples is 10(intact), since it's smaller than ``50(100 * 0.5)``,
        the negative samples will fill the rest ``40`` slots.
        If ``fill_negative == False``, the ``40`` slots is filled with ``-1(ignore)``.

    """
    def __init__(self, num_sample, pos_thresh, neg_thresh_high=0.5, neg_thresh_low=0.,
                 pos_ratio=0.5, neg_ratio=None, fill_negative=True):
        super(QuotaSamplerProp, self).__init__(need_top_grad=False)
        self.num_sample = int(num_sample)
        self.pos_thresh = float(pos_thresh)
        self.neg_thresh_high = float(neg_thresh_high)
        self.neg_thresh_low = float(neg_thresh_low)
        self.pos_ratio = float(pos_ratio)
        self.neg_ratio = None if neg_ratio is None else float(neg_ratio)
        self.fill_negative = bool(fill_negative)

    def list_arguments(self):
        return ['matches', 'ious']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        return in_shape, [in_shape[0]], []

    def infer_type(self, in_type):
        return [in_type[0], in_type[0]], [in_type[0]], []

    # pylint: disable=unused-argument
    def create_operator(self, ctx, in_shapes, in_dtypes):
        return QuotaSamplerOp(self.num_sample, self.pos_thresh, self.neg_thresh_high,
                              self.neg_thresh_low, self.pos_ratio, self.neg_ratio,
                              self.fill_negative)
