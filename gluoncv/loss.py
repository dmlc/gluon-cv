# pylint: disable=arguments-differ
"""Custom losses.
Losses are subclasses of gluon.loss.Loss which is a HybridBlock actually.
"""
from __future__ import absolute_import
from mxnet import gluon
from mxnet import nd
from mxnet.gluon.loss import Loss, _apply_weighting, _reshape_like

__all__ = ['FocalLoss', 'SSDMultiBoxLoss', 'YOLOV3Loss',
           'MixSoftmaxCrossEntropyLoss', 'ICNetLoss', 'MixSoftmaxCrossEntropyOHEMLoss',
           'SegmentationMultiLosses', 'DistillationSoftmaxCrossEntropyLoss', 'SiamRPNLoss']

class FocalLoss(Loss):
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
        - **loss**: loss tensor with shape (batch_size,). Dimensions other than
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
    min_hard_negatives : int, default is 0
        Minimum number of negatives samples.

    """
    def __init__(self, negative_mining_ratio=3, rho=1.0, lambd=1.0,
                 min_hard_negatives=0, **kwargs):
        super(SSDMultiBoxLoss, self).__init__(**kwargs)
        self._negative_mining_ratio = max(0, negative_mining_ratio)
        self._rho = rho
        self._lambd = lambd
        self._min_hard_negatives = max(0, min_hard_negatives)

    def forward(self, cls_pred, box_pred, cls_target, box_target):
        """Compute loss in entire batch across devices.

        Parameters
        ----------
        cls_pred : mxnet.nd.NDArray
        Predicted classes.
        box_pred : mxnet.nd.NDArray
        Predicted bounding-boxes.
        cls_target : mxnet.nd.NDArray
        Ground-truth classes.
        box_target : mxnet.nd.NDArray
        Ground-truth bounding-boxes.

        Returns
        -------
        tuple of NDArrays
            sum_losses : array with containing the sum of
                class prediction and bounding-box regression loss.
            cls_losses : array of class prediction loss.
            box_losses : array of box regression L1 loss.

        """
        # require results across different devices at this time
        cls_pred, box_pred, cls_target, box_target = [_as_list(x) \
            for x in (cls_pred, box_pred, cls_target, box_target)]
        # cross device reduction to obtain positive samples in entire batch
        num_pos = []
        for cp, bp, ct, bt in zip(*[cls_pred, box_pred, cls_target, box_target]):
            pos_samples = (ct > 0)
            num_pos.append(pos_samples.sum())
        num_pos_all = sum([p.asscalar() for p in num_pos])
        if num_pos_all < 1 and self._min_hard_negatives < 1:
            # no positive samples and no hard negatives, return dummy losses
            cls_losses = [nd.sum(cp * 0) for cp in cls_pred]
            box_losses = [nd.sum(bp * 0) for bp in box_pred]
            sum_losses = [nd.sum(cp * 0) + nd.sum(bp * 0) for cp, bp in zip(cls_pred, box_pred)]
            return sum_losses, cls_losses, box_losses


        # compute element-wise cross entropy loss and sort, then perform negative mining
        cls_losses = []
        box_losses = []
        sum_losses = []
        for cp, bp, ct, bt in zip(*[cls_pred, box_pred, cls_target, box_target]):
            pred = nd.log_softmax(cp, axis=-1)
            pos = ct > 0
            cls_loss = -nd.pick(pred, ct, axis=-1, keepdims=False)
            rank = (cls_loss * (pos - 1)).argsort(axis=1).argsort(axis=1)
            hard_negative = rank < nd.maximum(self._min_hard_negatives, pos.sum(axis=1)
                                              * self._negative_mining_ratio).expand_dims(-1)
            # mask out if not positive or negative
            cls_loss = nd.where((pos + hard_negative) > 0, cls_loss, nd.zeros_like(cls_loss))
            cls_losses.append(nd.sum(cls_loss, axis=0, exclude=True) / max(1., num_pos_all))

            bp = _reshape_like(nd, bp, bt)
            box_loss = nd.abs(bp - bt)
            box_loss = nd.where(box_loss > self._rho, box_loss - 0.5 * self._rho,
                                (0.5 / self._rho) * nd.square(box_loss))
            # box loss only apply to positive samples
            box_loss = box_loss * pos.expand_dims(axis=-1)
            box_losses.append(nd.sum(box_loss, axis=0, exclude=True) / max(1., num_pos_all))
            sum_losses.append(cls_losses[-1] + self._lambd * box_losses[-1])

        return sum_losses, cls_losses, box_losses


class YOLOV3Loss(Loss):
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
        weight_t = F.broadcast_mul(weight_t, objness_t)
        hard_objness_t = F.where(objness_t > 0, F.ones_like(objness_t), objness_t)
        new_objness_mask = F.where(objness_t > 0, objness_t, objness_t >= 0)
        obj_loss = F.broadcast_mul(
            self._sigmoid_ce(objness, hard_objness_t, new_objness_mask), denorm)
        center_loss = F.broadcast_mul(self._sigmoid_ce(box_centers, center_t, weight_t), denorm * 2)
        scale_loss = F.broadcast_mul(self._l1_loss(box_scales, scale_t, weight_t), denorm * 2)
        denorm_class = F.cast(
            F.shape_array(class_t).slice_axis(axis=0, begin=1, end=None).prod(), 'float32')
        class_mask = F.broadcast_mul(class_mask, objness_t)
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
        if self._sparse_label:
            loss = -F.pick(F.log(softmaxout), label, axis=1, keepdims=True)
        else:
            label = _reshape_like(F, label, pred)
            loss = -F.sum(F.log(softmaxout) * label, axis=-1, keepdims=True)
        loss = F.where(label.expand_dims(axis=1) == self._ignore_label,
                       F.zeros_like(loss), loss)
        return F.mean(loss, axis=self._batch_axis, exclude=True)


class SegmentationMultiLosses(SoftmaxCrossEntropyLoss):
    """2D Cross Entropy Loss with Multi-Loss"""
    def __init__(self, size_average=True, ignore_label=-1, **kwargs):
        super(SegmentationMultiLosses, self).__init__(size_average, ignore_label, **kwargs)

    def hybrid_forward(self, F, *inputs, **kwargs):
        pred1, pred2, pred3, label = tuple(inputs)

        loss1 = super(SegmentationMultiLosses, self).hybrid_forward(F, pred1, label, **kwargs)
        loss2 = super(SegmentationMultiLosses, self).hybrid_forward(F, pred2, label, **kwargs)
        loss3 = super(SegmentationMultiLosses, self).hybrid_forward(F, pred3, label, **kwargs)
        loss = loss1 + loss2 + loss3
        return loss


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
        loss1 = self._mixup_forward(F, pred1, label1, label2, lam)
        loss2 = self._mixup_forward(F, pred2, label1, label2, lam)
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


class ICNetLoss(SoftmaxCrossEntropyLoss):
    """Weighted SoftmaxCrossEntropyLoss2D for ICNet training

    Parameters
    ----------
    weights : tuple, default (0.4, 0.4, 1.0)
        The weight for cascade label guidance.
    ignore_label : int, default -1
        The label to ignore.
    """

    def __init__(self, weights=(0.4, 0.4, 1.0), height=None, width=None,
                 crop_size=480, ignore_label=-1, **kwargs):
        super(ICNetLoss, self).__init__(ignore_label=ignore_label, **kwargs)
        self.weights = weights
        self.height = height if height is not None else crop_size
        self.width = width if width is not None else crop_size

    def _weighted_forward(self, F, *inputs):
        label = inputs[4]
        loss = []
        for i in range(len(inputs) - 1):
            scale_pred = F.contrib.BilinearResize2D(inputs[i],
                                                    height=self.height,
                                                    width=self.width)
            loss.append(super(ICNetLoss, self).hybrid_forward(F, scale_pred, label))

        return loss[0] + self.weights[0] * loss[1] + \
               self.weights[1] * loss[2] + self.weights[2] * loss[3]

    def hybrid_forward(self, F, *inputs):
        """Compute loss"""
        return self._weighted_forward(F, *inputs)


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

class DistillationSoftmaxCrossEntropyLoss(gluon.HybridBlock):
    """SoftmaxCrossEntrolyLoss with Teacher model prediction

    Parameters
    ----------
    temperature : float, default 1
        The temperature parameter to soften teacher prediction.
    hard_weight : float, default 0.5
        The weight for loss on the one-hot label.
    sparse_label : bool, default True
        Whether the one-hot label is sparse.
    """
    def __init__(self, temperature=1, hard_weight=0.5, sparse_label=True, **kwargs):
        super(DistillationSoftmaxCrossEntropyLoss, self).__init__(**kwargs)
        self._temperature = temperature
        self._hard_weight = hard_weight
        with self.name_scope():
            self.soft_loss = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False, **kwargs)
            self.hard_loss = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=sparse_label, **kwargs)

    def hybrid_forward(self, F, output, label, soft_target):
        # pylint: disable=unused-argument
        """Compute loss"""
        if self._hard_weight == 0:
            return (self._temperature ** 2) * self.soft_loss(output / self._temperature,
                                                             soft_target)
        elif self._hard_weight == 1:
            return self.hard_loss(output, label)
        else:
            soft_loss = (self._temperature ** 2) * self.soft_loss(output / self._temperature,
                                                                  soft_target)
            hard_loss = self.hard_loss(output, label)
            return (1 - self._hard_weight) * soft_loss  + self._hard_weight * hard_loss


class HeatmapFocalLoss(Loss):
    """Focal loss for heatmaps.

    Parameters
    ----------
    from_logits : bool
        Whether predictions are after sigmoid or softmax.
    batch_axis : int
        Batch axis.
    weight : float
        Loss weight.

    """
    def __init__(self, from_logits=False, batch_axis=0, weight=None, **kwargs):
        super(HeatmapFocalLoss, self).__init__(weight, batch_axis, **kwargs)
        self._from_logits = from_logits

    def hybrid_forward(self, F, pred, label):
        """Loss forward"""
        if not self._from_logits:
            pred = F.sigmoid(pred)
        pos_inds = label == 1
        neg_inds = label < 1
        neg_weights = F.power(1 - label, 4)
        pos_loss = F.log(pred) * F.power(1 - pred, 2) * pos_inds
        neg_loss = F.log(1 - pred) * F.power(pred, 2) * neg_weights * neg_inds

        # normalize
        num_pos = F.clip(F.sum(pos_inds), a_min=1, a_max=1e30)
        pos_loss = F.sum(pos_loss)
        neg_loss = F.sum(neg_loss)
        return -(pos_loss + neg_loss) / num_pos


class MaskedL1Loss(Loss):
    r"""Calculates the mean absolute error between `label` and `pred` with `mask`.

    .. math:: L = \sum_i \vert ({label}_i - {pred}_i) * {mask}_i \vert / \sum_i {mask}_i.

    `label`, `pred` and `mask` can have arbitrary shape as long as they have the same
    number of elements. The final loss is normalized by the number of non-zero elements in mask.

    Parameters
    ----------
    weight : float or None
        Global scalar weight for loss.
    batch_axis : int, default 0
        The axis that represents mini-batch.


    Inputs:
        - **pred**: prediction tensor with arbitrary shape
        - **label**: target tensor with the same size as pred.
        - **sample_weight**: element-wise weighting tensor. Must be broadcastable
          to the same shape as pred. For example, if pred has shape (64, 10)
          and you want to weigh each sample in the batch separately,
          sample_weight should have shape (64, 1).

    Outputs:
        - **loss**: loss tensor with shape (batch_size,). Dimenions other than
          batch_axis are averaged out.
    """

    def __init__(self, weight=None, batch_axis=0, **kwargs):
        super(MaskedL1Loss, self).__init__(weight, batch_axis, **kwargs)

    def hybrid_forward(self, F, pred, label, mask, sample_weight=None):
        label = _reshape_like(F, label, pred)
        loss = F.abs(label * mask - pred * mask)
        loss = _apply_weighting(F, loss, self._weight, sample_weight)
        norm = F.sum(mask).clip(1, 1e30)
        return F.sum(loss) / norm

class SiamRPNLoss(gluon.HybridBlock):
    r"""Weighted l1 loss and cross entropy loss for SiamRPN training

    Parameters
    ----------
    batch_size : int, default 128
        training batch size per device (CPU/GPU).

    """
    def __init__(self, batch_size=128, **kwargs):
        super(SiamRPNLoss, self).__init__(**kwargs)
        self.conf_loss = gluon.loss.SoftmaxCrossEntropyLoss()
        self.h = 17
        self.w = 17
        self.b = batch_size
        self.loc_c = 10
        self.cls_c = 5

    def weight_l1_loss(self, F, pred_loc, label_loc, loss_weight):
        """Compute weight_l1_loss"""
        pred_loc = pred_loc.reshape((self.b, 4, -1, self.h, self.w))
        diff = F.abs((pred_loc - label_loc))
        diff = F.sum(diff, axis=1).reshape((self.b, -1, self.h, self.w))
        loss = diff * loss_weight
        return F.sum(loss)/self.b

    def get_cls_loss(self, F, pred, label, select):
        """Compute SoftmaxCrossEntropyLoss"""
        if len(select) == 0:
            return 0
        pred = F.gather_nd(pred, select.reshape(1, -1))
        label = F.gather_nd(label.reshape(-1, 1), select.reshape(1, -1)).reshape(-1)
        return self.conf_loss(pred, label).mean()

    def cross_entropy_loss(self, F, pred, label, pos_index, neg_index):
        """Compute cross_entropy_loss"""
        pred = pred.reshape(self.b, 2, self.loc_c//2, self.h, self.h)
        pred = F.transpose(pred, axes=((0, 2, 3, 4, 1)))
        pred = pred.reshape(-1, 2)
        label = label.reshape(-1)
        loss_pos = self.get_cls_loss(F, pred, label, pos_index)
        loss_neg = self.get_cls_loss(F, pred, label, neg_index)
        return loss_pos * 0.5 + loss_neg * 0.5

    def hybrid_forward(self, F, cls_pred, loc_pred, label_cls, pos_index, neg_index,
                       label_loc, label_loc_weight):
        """Compute loss"""
        loc_loss = self.weight_l1_loss(F, loc_pred, label_loc, label_loc_weight)
        cls_loss = self.cross_entropy_loss(F, cls_pred, label_cls, pos_index, neg_index)
        return cls_loss, loc_loss
