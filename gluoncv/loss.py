# pylint: disable=arguments-differ
"""Custom losses.
Losses are subclasses of gluon.loss.Loss which is a HybridBlock actually.
"""
from __future__ import absolute_import
from mxnet import gluon
from mxnet import nd
from mxnet.gluon.loss import Loss, _apply_weighting, _reshape_like

__all__ = ['FocalLoss', 'SSDMultiBoxLoss', 'YOLOV3Loss',
           'MixSoftmaxCrossEntropyLoss', 'MixSoftmaxCrossEntropyOHEMLoss']

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


class YOLOV3Loss(gluon.loss.Loss):
    """Losses of YOLO v3.

    Parameters
    ----------
    batch_axis : int, default 0
        The axis that represents mini-batch.
    weight : float or None
        Global scalar weight for loss.

    """
    def __init__(self, batch_axis=0, weight=None, **kwargs):
        super(YOLOV3Loss, self).__init__(weight, batch_axis, **kwargs)
        self._sigmoid_ce = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
        self._l1_loss = gluon.loss.L1Loss()

    def hybrid_forward(self, F, objness, box_centers, box_scales, cls_preds,
                       objness_t, center_t, scale_t, weight_t, class_t, class_mask):
        """Compute YOLOv3 losses.

        Parameters
        ----------
        objness : mxnet.nd.NDArray
            Predicted objectness (B, N), range (0, 1).
        box_centers : mxnet.nd.NDArray
            Predicted box centers (x, y) (B, N, 2), range (0, 1).
        box_scales : mxnet.nd.NDArray
            Predicted box scales (width, height) (B, N, 2).
        cls_preds : mxnet.nd.NDArray
            Predicted class predictions (B, N, num_class), range (0, 1).
        objness_t : mxnet.nd.NDArray
            Objectness target, (B, N), 0 for negative 1 for positive, -1 for ignore.
        center_t : mxnet.nd.NDArray
            Center (x, y) targets (B, N, 2).
        scale_t : mxnet.nd.NDArray
            Scale (width, height) targets (B, N, 2).
        weight_t : mxnet.nd.NDArray
            Loss Multipliers for center and scale targets (B, N, 2).
        class_t : mxnet.nd.NDArray
            Class targets (B, N, num_class).
            It's relaxed one-hot vector, i.e., (1, 0, 1, 0, 0).
            It can contain more than one positive class.
        class_mask : mxnet.nd.NDArray
            0 or 1 mask array to mask out ignored samples (B, N, num_class).

        Returns
        -------
        tuple of NDArrays
            obj_loss: sum of objectness logistic loss
            center_loss: sum of box center logistic regression loss
            scale_loss: sum of box scale l1 loss
            cls_loss: sum of per class logistic loss

        """
        # compute some normalization count, except batch-size
        denorm = F.cast(
            F.shape_array(objness_t).slice_axis(axis=0, begin=1, end=None).prod(), 'float32')
        obj_loss = F.broadcast_mul(self._sigmoid_ce(objness, objness_t, objness_t >= 0), denorm)
        center_loss = F.broadcast_mul(self._sigmoid_ce(box_centers, center_t, weight_t), denorm * 2)
        scale_loss = F.broadcast_mul(self._l1_loss(box_scales, scale_t, weight_t), denorm * 2)
        denorm_class = F.cast(
            F.shape_array(class_t).slice_axis(axis=0, begin=1, end=None).prod(), 'float32')
        cls_loss = F.broadcast_mul(self._sigmoid_ce(cls_preds, class_t, class_mask), denorm_class)
        return obj_loss, center_loss, scale_loss, cls_loss


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
    """
    def __init__(self, sparse_label=True, batch_axis=0, ignore_label=-1,
                 size_average=True, **kwargs):
        super(SoftmaxCrossEntropyLoss, self).__init__(None, batch_axis, **kwargs)
        self._sparse_label = sparse_label
        self._ignore_label = ignore_label
        self._size_average = size_average

    def hybrid_forward(self, F, pred, label):
        """Compute loss"""
        softmaxout = F.SoftmaxOutput(
            pred, label.astype(pred.dtype), ignore_label=self._ignore_label,
            multi_output=self._sparse_label,
            use_ignore=True, normalization='valid' if self._size_average else 'null')
        loss = -F.pick(F.log(softmaxout), label, axis=1, keepdims=True)
        loss = F.where(label.expand_dims(axis=1) == self._ignore_label,
                       F.zeros_like(loss), loss)
        return F.mean(loss, axis=self._batch_axis, exclude=True)

class MixSoftmaxCrossEntropyLoss(SoftmaxCrossEntropyLoss):
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
    def __init__(self, aux=True, mixup=False, aux_weight=0.2, ignore_label=-1, **kwargs):
        super(MixSoftmaxCrossEntropyLoss, self).__init__(
            ignore_label=ignore_label, **kwargs)
        self.aux = aux
        self.mixup = mixup
        self.aux_weight = aux_weight

    def _aux_forward(self, F, pred1, pred2, label, **kwargs):
        """Compute loss including auxiliary output"""
        loss1 = super(MixSoftmaxCrossEntropyLoss, self). \
            hybrid_forward(F, pred1, label, **kwargs)
        loss2 = super(MixSoftmaxCrossEntropyLoss, self). \
            hybrid_forward(F, pred2, label, **kwargs)
        return loss1 + self.aux_weight * loss2

    def _aux_mixup_forward(self, F, pred1, pred2, label1, label2, lam):
        """Compute loss including auxiliary output"""
        loss1 = self._mixup_forwar(F, pred1, label1, label2, lam)
        loss2 = self._mixup_forwar(F, pred2, label1, label2, lam)
        return loss1 + self.aux_weight * loss2

    def _mixup_forward(self, F, pred, label1, label2, lam, sample_weight=None):
        if not self._from_logits:
            pred = F.log_softmax(pred, self._axis)
        if self._sparse_label:
            loss1 = -F.pick(pred, label1, axis=self._axis, keepdims=True)
            loss2 = -F.pick(pred, label2, axis=self._axis, keepdims=True)
            loss = lam * loss1 + (1 - lam) * loss2
        else:
            label1 = _reshape_like(F, label1, pred)
            label2 = _reshape_like(F, label2, pred)
            loss1 = -F.sum(pred*label1, axis=self._axis, keepdims=True)
            loss2 = -F.sum(pred*label2, axis=self._axis, keepdims=True)
            loss = lam * loss1 + (1 - lam) * loss2
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        return F.mean(loss, axis=self._batch_axis, exclude=True)

    def hybrid_forward(self, F, *inputs, **kwargs):
        """Compute loss"""
        if self.aux:
            if self.mixup:
                return self._aux_mixup_forward(F, *inputs, **kwargs)
            else:
                return self._aux_forward(F, *inputs, **kwargs)
        else:
            if self.mixup:
                return self._mixup_forward(F, *inputs, **kwargs)
            else:
                return super(MixSoftmaxCrossEntropyLoss, self). \
                    hybrid_forward(F, *inputs, **kwargs)

class SoftmaxCrossEntropyOHEMLoss(Loss):
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
    """
    def __init__(self, sparse_label=True, batch_axis=0, ignore_label=-1,
                 size_average=True, **kwargs):
        super(SoftmaxCrossEntropyOHEMLoss, self).__init__(None, batch_axis, **kwargs)
        self._sparse_label = sparse_label
        self._ignore_label = ignore_label
        self._size_average = size_average

    def hybrid_forward(self, F, pred, label):
        """Compute loss"""
        softmaxout = F.contrib.SoftmaxOHEMOutput(
            pred, label.astype(pred.dtype), ignore_label=self._ignore_label,
            multi_output=self._sparse_label,
            use_ignore=True, normalization='valid' if self._size_average else 'null',
            thresh=0.6, min_keep=256)
        loss = -F.pick(F.log(softmaxout), label, axis=1, keepdims=True)
        loss = F.where(label.expand_dims(axis=1) == self._ignore_label,
                       F.zeros_like(loss), loss)
        return F.mean(loss, axis=self._batch_axis, exclude=True)

class MixSoftmaxCrossEntropyOHEMLoss(SoftmaxCrossEntropyOHEMLoss):
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
        super(MixSoftmaxCrossEntropyOHEMLoss, self).__init__(
            ignore_label=ignore_label, **kwargs)
        self.aux = aux
        self.aux_weight = aux_weight

    def _aux_forward(self, F, pred1, pred2, label, **kwargs):
        """Compute loss including auxiliary output"""
        loss1 = super(MixSoftmaxCrossEntropyOHEMLoss, self). \
            hybrid_forward(F, pred1, label, **kwargs)
        loss2 = super(MixSoftmaxCrossEntropyOHEMLoss, self). \
            hybrid_forward(F, pred2, label, **kwargs)
        return loss1 + self.aux_weight * loss2

    def hybrid_forward(self, F, *inputs, **kwargs):
        """Compute loss"""
        if self.aux:
            return self._aux_forward(F, *inputs, **kwargs)
        else:
            return super(MixSoftmaxCrossEntropyOHEMLoss, self). \
                hybrid_forward(F, *inputs, **kwargs)
