# pylint: disable=arguments-differ
"""Custom losses.
Losses are subclasses of gluon.loss.Loss which is a HybridBlock actually.
"""
from __future__ import absolute_import
from mxnet import gluon
from mxnet import nd
from mxnet.gluon.loss import Loss, _apply_weighting, _reshape_like


class FocalLoss(gluon.loss.Loss):
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

def _as_list(arr):
    """Make sure input is a list of mxnet NDArray"""
    if not isinstance(arr, (list, tuple)):
        return [arr]
    return arr


class SSDMultiBoxLoss(gluon.Block):
    r"""Single-Shot Multibox Object Detection Loss.

    .. note::

        Since cross device synchronization is required to compute batch-wise statistics,
        it is slightly sub-optimal compared with non-sync version. However, we find this
        is better for converged model performance.

    Parameters
    ----------
    negative_mining_ratio : float, default is 3
        Ratio of negative vs. positive samples.
    rho : float, default is 1.0
        Threshold for trimmed mean estimator. This is the smooth parameter for the
        L1-L2 transition.
    lambd : float, default is 1.0
        Relative weight between classification and box regression loss.
        The overall loss is computed as :math:`L = loss_{class} + \lambda \times loss_{loc}`.

    """
    def __init__(self, negative_mining_ratio=3, rho=1.0, lambd=1.0, **kwargs):
        super(SSDMultiBoxLoss, self).__init__(**kwargs)
        self._negative_mining_ratio = max(0, negative_mining_ratio)
        self._rho = rho
        self._lambd = lambd

    def forward(self, cls_pred, box_pred, cls_target, box_target):
        """Compute loss in entire batch across devices."""
        # require results across different devices at this time
        cls_pred, box_pred, cls_target, box_target = [_as_list(x) \
            for x in (cls_pred, box_pred, cls_target, box_target)]
        # cross device reduction to obtain positive samples in entire batch
        num_pos = []
        for cp, bp, ct, bt in zip(*[cls_pred, box_pred, cls_target, box_target]):
            pos_samples = (ct > 0)
            num_pos.append(pos_samples.sum())
        num_pos_all = sum([p.asscalar() for p in num_pos])
        if num_pos_all < 1:
            # no positive samples found, return dummy losses
            return nd.zeros((1,)), nd.zeros((1,)), nd.zeros((1,))

        # compute element-wise cross entropy loss and sort, then perform negative mining
        cls_losses = []
        box_losses = []
        sum_losses = []
        for cp, bp, ct, bt in zip(*[cls_pred, box_pred, cls_target, box_target]):
            pred = nd.log_softmax(cp, axis=-1)
            pos = ct > 0
            cls_loss = -nd.pick(pred, ct, axis=-1, keepdims=False)
            rank = (cls_loss * (pos - 1)).argsort(axis=1).argsort(axis=1)
            hard_negative = rank < (pos.sum(axis=1) * self._negative_mining_ratio).expand_dims(-1)
            # mask out if not positive or negative
            cls_loss = nd.where((pos + hard_negative) > 0, cls_loss, nd.zeros_like(cls_loss))
            cls_losses.append(nd.sum(cls_loss, axis=0, exclude=True) / num_pos_all)

            bp = _reshape_like(nd, bp, bt)
            box_loss = nd.abs(bp - bt)
            box_loss = nd.where(box_loss > self._rho, box_loss - 0.5 * self._rho,
                                (0.5 / self._rho) * nd.square(box_loss))
            # box loss only apply to positive samples
            box_loss = box_loss * pos.expand_dims(axis=-1)
            box_losses.append(nd.sum(box_loss, axis=0, exclude=True) / num_pos_all)
            sum_losses.append(cls_losses[-1] + self._lambd * box_losses[-1])

        return sum_losses, cls_losses, box_losses

class SoftmaxCrossEntropyLoss(Loss):
    r"""SoftmaxCrossEntropyLoss with ignore labels

    Parameters
    ----------
    axis : int, default -1
        The axis to sum over when computing softmax and entropy.
    sparse_label : bool, default True
        Whether label is an integer array instead of probability distribution.
    from_logits : bool, default False
        Whether input is a log probability (usually from log_softmax) instead
        of unnormalized numbers.
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.
    ignore_label : int, default -1
        The label to ignore.
    size_average : bool, default False
        Whether to re-scale loss with regard to ignored labels.
    valid_size : int, default 1000
        The size of label, equal to the number of classes for classification models.
        It affects the re-scaled loss if size_average is True.
    """
    # pylint: disable=unused-argument
    def __init__(self, axis=1, sparse_label=True, from_logits=False, weight=None,
                 batch_axis=0, ignore_label=-1, size_average=False, valid_size=1000, **kwargs):
        super(SoftmaxCrossEntropyLoss, self).__init__(weight, batch_axis, **kwargs)
        self._axis = axis
        self._sparse_label = sparse_label
        self._from_logits = from_logits
        self._ignore_label = ignore_label
        self._size_average = size_average

    def hybrid_forward(self, F, pred, label, sample_weight=None):
        """Compute loss"""
        if not self._from_logits:
            pred = F.log_softmax(pred, axis=self._axis)
        if self._sparse_label:
            if self._size_average:
                valid_label_map = (label != self._ignore_label).astype('float32')
            loss = -F.pick(pred, label, axis=self._axis, keepdims=True)
            loss = F.where(label.expand_dims(axis=self._axis) == self._ignore_label,
                           F.zeros_like(loss), loss)
        else:
            label = _reshape_like(F, label, pred)
            loss = -F.sum(pred*label, axis=self._axis, keepdims=True)
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        if self._size_average and self._sparse_label:
            return F.mean(loss, axis=self._batch_axis, exclude=True) * \
                valid_size / F.sum(valid_label_map)
        else:
            return F.mean(loss, axis=self._batch_axis, exclude=True)

class SoftmaxCrossEntropyLossWithAux(SoftmaxCrossEntropyLoss):
    """SoftmaxCrossEntropyLoss2D with Auxiliary Loss

    Parameters
    ----------
    aux : bool, default True
        Whether to use auxiliary loss.
    aux_weight : float, default 0.2
        The weight for aux loss.
    ignore_label : int, default -1
        The label to ignore.
    """
    def __init__(self, aux=True, aux_weight=0.2, ignore_label=-1, **kwargs):
        super(SoftmaxCrossEntropyLossWithAux, self).__init__(
            axis=1, ignore_label=ignore_label, **kwargs)
        self.aux = aux
        self.aux_weight = aux_weight

    def aux_forward(self, F, pred1, pred2, label, **kwargs):
        """Compute loss including auxiliary output"""
        loss1 = super(SoftmaxCrossEntropyLossWithAux, self). \
            hybrid_forward(F, pred1, label, **kwargs)
        loss2 = super(SoftmaxCrossEntropyLossWithAux, self). \
            hybrid_forward(F, pred2, label, **kwargs)
        return loss1 + self.aux_weight * loss2

    def hybrid_forward(self, F, *inputs, **kwargs):
        """Compute loss"""
        if self.aux:
            return self.aux_forward(F, *inputs, **kwargs)
        else:
            return super(SoftmaxCrossEntropyLossWithAux, self). \
                hybrid_forward(F, *inputs, **kwargs)
