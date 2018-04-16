"""Custom losses for object detection.
Losses are used to penalize incorrect classification and inaccurate box regression.
Losses are subclasses of gluon.loss.Loss which is a HybridBlock actually.
"""
from __future__ import absolute_import
import mxnet
from mxnet.gluon.loss import _apply_weighting

# pylint: disable=arguments-differ

class FocalLoss(mxnet.gluon.loss.Loss):
    """Focal Loss for inbalanced classification.
    Focal loss was described in https://arxiv.org/abs/1708.02002

    Parameters
    ----------
    axis : int, default -1
        The axis to sum over when computing softmax and entropy.
    alpha : float, default 0.25
        The alpha which controls loss curve.
    gamma : float, default 2
        The gamma which controls loss curve.
    sparse_label : bool, default True
        Whether label is an integer array instead of probability distribution.
    from_logits : bool, default False
        Whether input is a log probability (usually from log_softmax) instead.
    batch_axis : int, default 0
        The axis that represents mini-batch.
    weight : float or None
        Global scalar weight for loss.
    num_class : int
        Number of classification categories. It is required is `sparse_label` is `True`.
    eps : float
        Eps to avoid numerical issue.
    size_average : bool, default True
        If `True`, will take mean of the output loss on every axis except `batch_axis`.

    Inputs:
        - **pred**: the prediction tensor, where the `batch_axis` dimension
          ranges over batch size and `axis` dimension ranges over the number
          of classes.
        - **label**: the truth tensor. When `sparse_label` is True, `label`'s
          shape should be `pred`'s shape with the `axis` dimension removed.
          i.e. for `pred` with shape (1,2,3,4) and `axis = 2`, `label`'s shape
          should be (1,2,4) and values should be integers between 0 and 2. If
          `sparse_label` is False, `label`'s shape must be the same as `pred`
          and values should be floats in the range `[0, 1]`.
        - **sample_weight**: element-wise weighting tensor. Must be broadcastable
          to the same shape as label. For example, if label has shape (64, 10)
          and you want to weigh each sample in the batch separately,
          sample_weight should have shape (64, 1).
    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimenions other than
          batch_axis are averaged out.
    """
    def __init__(self, axis=-1, alpha=0.25, gamma=2, sparse_label=True,
                 from_logits=False, batch_axis=0, weight=None, num_class=None,
                 eps=1e-12, size_average=True, **kwargs):
        super(FocalLoss, self).__init__(weight, batch_axis, **kwargs)
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma
        self._sparse_label = sparse_label
        if sparse_label and (not isinstance(num_class, int) or (num_class < 1)):
            raise ValueError("Number of class > 0 must be provided if sparse label is used.")
        self._num_class = num_class
        self._from_logits = from_logits
        self._eps = eps
        self._size_average = size_average

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        """Loss forward"""
        if not self._from_logits:
            pred = F.sigmoid(pred)
        if self._sparse_label:
            one_hot = F.one_hot(label, self._num_class)
        else:
            one_hot = label > 0
        pt = F.where(one_hot, pred, 1 - pred)
        t = F.ones_like(one_hot)
        alpha = F.where(one_hot, self._alpha * t, (1 - self._alpha) * t)
        loss = -alpha * ((1 - pt) ** self._gamma) * F.log(F.minimum(pt + self._eps, 1))
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        if self._size_average:
            return F.mean(loss, axis=self._batch_axis, exclude=True)
        else:
            return F.sum(loss, axis=self._batch_axis, exclude=True)
