# pylint: disable=arguments-differ, missing-docstring
"""Encoder and Decoder functions.
Encoders are used during training, which assign training targets.
Decoders are used during testing/validation, which convert predictions back to
normal boxes, etc.
"""
from __future__ import absolute_import
from mxnet import nd
from mxnet import gluon
from .bbox import BBoxCornerToCenter


class NormalizedBoxCenterEncoder(gluon.Block):
    """Encode bounding boxes training target with normalized center offsets.

    Input bounding boxes are using corner type: `x_{min}, y_{min}, x_{max}, y_{max}`.

    Parameters
    ----------
    stds : array-like of size 4
        Std value to be divided from encoded values, default is (0.1, 0.1, 0.2, 0.2).

    """
    def __init__(self, stds=(0.1, 0.1, 0.2, 0.2)):
        super(NormalizedBoxCenterEncoder, self).__init__()
        assert len(stds) == 4, "Box Encoder requires 4 std values."
        self._stds = stds
        with self.name_scope():
            self.corner_to_center = BBoxCornerToCenter(split=True)

    def forward(self, samples, matches, anchors, refs):
        """Forward"""
        F = nd
        # TODO(zhreshold): batch_pick, take multiple elements?
        ref_boxes = nd.repeat(refs.reshape((0, 1, -1, 4)), axis=1, repeats=matches.shape[1])
        ref_boxes = nd.split(ref_boxes, axis=-1, num_outputs=4, squeeze_axis=True)
        ref_boxes = nd.concat(*[F.pick(ref_boxes[i], matches, axis=2).reshape((0, -1, 1)) \
            for i in range(4)], dim=2)
        g = self.corner_to_center(ref_boxes)
        a = self.corner_to_center(anchors)
        t0 = (g[0] - a[0]) / a[2] / self._stds[0]
        t1 = (g[1] - a[1]) / a[3] / self._stds[1]
        t2 = F.log(g[2] / a[2]) / self._stds[2]
        t3 = F.log(g[3] / a[3]) / self._stds[3]
        codecs = F.concat(t0, t1, t2, t3, dim=2)
        temp = F.tile(samples.reshape((0, -1, 1)), reps=(1, 1, 4)) > 0.5
        targets = F.where(temp, codecs, F.zeros_like(codecs))
        masks = F.where(temp, F.ones_like(temp), F.zeros_like(temp))
        return targets, masks


class NormalizedBoxCenterDecoder(gluon.HybridBlock):
    """Decode bounding boxes training target with normalized center offsets.
    This decoder must cooperate with NormalizedBoxCenterEncoder of same `stds`
    in order to get properly reconstructed bounding boxes.

    Returned bounding boxes are using corner type: `x_{min}, y_{min}, x_{max}, y_{max}`.

    Parameters
    ----------
    stds : array-like of size 4
        Std value to be divided from encoded values, default is (0.1, 0.1, 0.2, 0.2).

    """
    def __init__(self, stds=(0.1, 0.1, 0.2, 0.2)):
        super(NormalizedBoxCenterDecoder, self).__init__()
        assert len(stds) == 4, "Box Encoder requires 4 std values."
        self._stds = stds

    def hybrid_forward(self, F, x, anchors):
        a = anchors.split(axis=-1, num_outputs=4)
        p = F.split(x, axis=-1, num_outputs=4)
        ox = F.broadcast_add(F.broadcast_mul(p[0] * self._stds[0], a[2]), a[0])
        oy = F.broadcast_add(F.broadcast_mul(p[1] * self._stds[1], a[3]), a[1])
        ow = F.broadcast_mul(F.exp(p[2] * self._stds[2]), a[2]) / 2
        oh = F.broadcast_mul(F.exp(p[3] * self._stds[3]), a[3]) / 2
        return F.concat(ox - ow, oy - oh, ox + ow, oy + oh, dim=-1)


class MultiClassEncoder(gluon.HybridBlock):
    """Encode classification training target given matching results.

    This encoder will assign training target of matched bounding boxes to
    ground-truth label + 1 and negative samples with label 0.
    Ignored samples will be assigned with `ignore_label`, whose default is -1.

    Parameters
    ----------
    ignore_label : float
        Assigned to un-matched samples, they are neither positive or negative during
        training, and should be excluded in loss function. Default is -1.

    """
    def __init__(self, ignore_label=-1):
        super(MultiClassEncoder, self).__init__()
        self._ignore_label = ignore_label

    def hybrid_forward(self, F, samples, matches, refs):
        refs = F.repeat(refs.reshape((0, 1, -1)), axis=1, repeats=matches.shape[1])
        target_ids = F.pick(refs, matches, axis=2) + 1
        targets = F.where(samples > 0.5, target_ids, nd.ones_like(target_ids) * self._ignore_label)
        targets = F.where(samples < -0.5, nd.zeros_like(targets), targets)
        return targets


class MultiClassDecoder(gluon.HybridBlock):
    """Decode classification results.

    This decoder must work with `MultiClassEncoder` to reconstruct valid labels.
    The decoder expect results are after logits, e.g. Softmax.

    Parameters
    ----------
    axis : int
        Axis of class-wise results.
    thresh : float
        Confidence threshold for the post-softmax scores.
        Scores less than `thresh` are marked with `0`, corresponding `cls_id` is
        marked with invalid class id `-1`.

    """
    def __init__(self, axis=-1, thresh=0.01):
        super(MultiClassDecoder, self).__init__()
        self._axis = axis
        self._thresh = thresh

    def hybrid_forward(self, F, x):
        pos_x = x.slice_axis(axis=self._axis, begin=1, end=None)
        cls_id = F.argmax(pos_x, self._axis)
        scores = F.pick(pos_x, cls_id, axis=-1)
        mask = scores > self._thresh
        cls_id = F.where(mask, cls_id, F.ones_like(cls_id) * -1)
        scores = F.where(mask, scores, F.zeros_like(scores))
        return cls_id, scores

class MultiPerClassDecoder(gluon.HybridBlock):
    """Decode classification results.

    This decoder must work with `MultiClassEncoder` to reconstruct valid labels.
    The decoder expect results are after logits, e.g. Softmax.
    This version is different from
    :py:class:`gluoncv.nn.coder.MultiClassDecoder` with the following changes:

    For each position(anchor boxes), each foreground class can have their own
    results, rather than enforced to be the best one.
    For example, for a 5-class prediction with background(totaling 6 class), say
    (0.5, 0.1, 0.2, 0.1, 0.05, 0.05) as (bg, apple, orange, peach, grape, melon),
    `MultiClassDecoder` produce only one class id and score, that is  (orange-0.2).
    `MultiPerClassDecoder` produce 5 results individually:
    (apple-0.1, orange-0.2, peach-0.1, grape-0.05, melon-0.05).

    Parameters
    ----------
    num_class : int
        Number of classes including background.
    axis : int
        Axis of class-wise results.
    thresh : float
        Confidence threshold for the post-softmax scores.
        Scores less than `thresh` are marked with `0`, corresponding `cls_id` is
        marked with invalid class id `-1`.

    """
    def __init__(self, num_class, axis=-1, thresh=0.01):
        super(MultiPerClassDecoder, self).__init__()
        self._fg_class = num_class - 1
        self._axis = axis
        self._thresh = thresh

    def hybrid_forward(self, F, x):
        scores = x.slice_axis(axis=self._axis, begin=1, end=None)  # b x N x fg_class
        template = F.zeros_like(x.slice_axis(axis=-1, begin=0, end=1))
        cls_ids = []
        for i in range(self._fg_class):
            cls_ids.append(template + i)  # b x N x 1
        cls_id = F.concat(*cls_ids, dim=-1)  # b x N x fg_class
        mask = scores > self._thresh
        cls_id = F.where(mask, cls_id, F.ones_like(cls_id) * -1)
        scores = F.where(mask, scores, F.zeros_like(scores))
        return cls_id, scores
