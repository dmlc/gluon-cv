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
    means : array-like of size 4
        Mean value to be subtracted from encoded values, default is (0., 0., 0., 0.).

    """
    def __init__(self, stds=(0.1, 0.1, 0.2, 0.2), means=(0., 0., 0., 0.)):
        super(NormalizedBoxCenterEncoder, self).__init__()
        assert len(stds) == 4, "Box Encoder requires 4 std values."
        self._stds = stds
        self._means = means
        with self.name_scope():
            self.corner_to_center = BBoxCornerToCenter(split=True)

    def forward(self, samples, matches, anchors, refs):
        """Not HybridBlock due to use of matches.shape

        Parameters
        ----------
        samples: (B, N) value +1 (positive), -1 (negative), 0 (ignore)
        matches: (B, N) value range [0, M)
        anchors: (B, N, 4) encoded in corner
        refs: (B, M, 4) encoded in corner

        Returns
        -------
        targets: (B, N, 4) transform anchors to refs picked according to matches
        masks: (B, N, 4) only positive anchors has targets

        """
        F = nd
        # TODO(zhreshold): batch_pick, take multiple elements?
        # refs [B, M, 4], anchors [B, N, 4], samples [B, N], matches [B, N]
        # refs [B, M, 4] -> reshape [B, 1, M, 4] -> repeat [B, N, M, 4]
        ref_boxes = F.repeat(refs.reshape((0, 1, -1, 4)), axis=1, repeats=matches.shape[1])
        # refs [B, N, M, 4] -> 4 * [B, N, M]
        ref_boxes = F.split(ref_boxes, axis=-1, num_outputs=4, squeeze_axis=True)
        # refs 4 * [B, N, M] -> pick from matches [B, N, 1] -> concat to [B, N, 4]
        ref_boxes = F.concat(*[F.pick(ref_boxes[i], matches, axis=2).reshape((0, -1, 1)) \
            for i in range(4)], dim=2)
        # transform based on x, y, w, h
        # g [B, N, 4], a [B, N, 4] -> codecs [B, N, 4]
        g = self.corner_to_center(ref_boxes)
        a = self.corner_to_center(anchors)
        t0 = ((g[0] - a[0]) / a[2] - self._means[0]) / self._stds[0]
        t1 = ((g[1] - a[1]) / a[3] - self._means[1]) / self._stds[1]
        t2 = (F.log(g[2] / a[2]) - self._means[2]) / self._stds[2]
        t3 = (F.log(g[3] / a[3]) - self._means[3]) / self._stds[3]
        codecs = F.concat(t0, t1, t2, t3, dim=2)
        # samples [B, N] -> [B, N, 1] -> [B, N, 4] -> boolean
        temp = F.tile(samples.reshape((0, -1, 1)), reps=(1, 1, 4)) > 0.5
        # fill targets and masks [B, N, 4]
        targets = F.where(temp, codecs, F.zeros_like(codecs))
        masks = F.where(temp, F.ones_like(temp), F.zeros_like(temp))
        return targets, masks


class NormalizedPerClassBoxCenterEncoder(gluon.Block):
    """Encode bounding boxes training target with normalized center offsets.

    Input bounding boxes are using corner type: `x_{min}, y_{min}, x_{max}, y_{max}`.

    Parameters
    ----------
    stds : array-like of size 4
        Std value to be divided from encoded values, default is (0.1, 0.1, 0.2, 0.2).
    means : array-like of size 4
        Mean value to be subtracted from encoded values, default is (0., 0., 0., 0.).

    """
    def __init__(self, num_class, stds=(0.1, 0.1, 0.2, 0.2), means=(0., 0., 0., 0.)):
        super(NormalizedPerClassBoxCenterEncoder, self).__init__()
        assert len(stds) == 4, "Box Encoder requires 4 std values."
        assert num_class > 0, "Number of classes must be positive"
        self._num_class = num_class
        with self.name_scope():
            self.class_agnostic_encoder = NormalizedBoxCenterEncoder(stds=stds, means=means)

    def forward(self, samples, matches, anchors, labels, refs):
        """Encode BBox One entry per category

        Parameters
        ----------
        samples: (B, N) value +1 (positive), -1 (negative), 0 (ignore)
        matches: (B, N) value range [0, M)
        anchors: (B, N, 4) encoded in corner
        labels: (B, N) value range [0, self._num_class), excluding background
        refs: (B, M, 4) encoded in corner

        Returns
        -------
        targets: (C, B, N, 4) transform anchors to refs picked according to matches
        masks: (C, B, N, 4) only positive anchors of the correct class has targets

        """
        F = nd
        # refs [B, M, 4], anchors [B, N, 4], samples [B, N], matches [B, N]
        # encoded targets [B, N, 4], masks [B, N, 4]
        targets, masks = self.class_agnostic_encoder(samples, matches, anchors, refs)
        # labels [B, M] -> [B, N, M]
        ref_labels = F.repeat(labels.reshape((0, 1, -1)), axis=1, repeats=matches.shape[1])
        # labels [B, N, M] -> pick from matches [B, N] -> [B, N, 1]
        ref_labels = F.pick(ref_labels, matches, axis=2).reshape((0, -1, 1))
        # expand class agnostic targets to per class targets
        out_targets = []
        out_masks = []
        for cid in range(self._num_class):
            # boolean array [B, N, 1]
            same_cid = ref_labels == cid
            # keep orig targets
            out_targets.append(targets)
            # but mask out the one not belong to this class [B, N, 1] -> [B, N, 4]
            out_masks.append(masks * same_cid.repeat(axis=-1, repeats=4))
        # targets, masks C * [B, N, 4] -> [C, B, N, 4] -> [B, N, C, 4]
        all_targets = F.stack(*out_targets, axis=0)
        all_masks = F.stack(*out_masks, axis=0)
        return all_targets, all_masks


class NormalizedBoxCenterDecoder(gluon.HybridBlock):
    """Decode bounding boxes training target with normalized center offsets.
    This decoder must cooperate with NormalizedBoxCenterEncoder of same `stds`
    in order to get properly reconstructed bounding boxes.

    Returned bounding boxes are using corner type: `x_{min}, y_{min}, x_{max}, y_{max}`.

    Parameters
    ----------
    stds : array-like of size 4
        Std value to be divided from encoded values, default is (0.1, 0.1, 0.2, 0.2).
    means : array-like of size 4
        Mean value to be subtracted from encoded values, default is (0., 0., 0., 0.).
    clip: float, default is None
        If given, bounding box target will be clipped to this value.

    """
    def __init__(self, stds=(0.1, 0.1, 0.2, 0.2), means=(0., 0., 0., 0.),
                 convert_anchor=False, clip=None):
        super(NormalizedBoxCenterDecoder, self).__init__()
        assert len(stds) == 4, "Box Encoder requires 4 std values."
        self._stds = stds
        self._means = means
        self._clip = clip
        if convert_anchor:
            self.corner_to_center = BBoxCornerToCenter(split=True)
        else:
            self.corner_to_center = None

    def hybrid_forward(self, F, x, anchors):
        if self.corner_to_center is not None:
            a = self.corner_to_center(anchors)
        else:
            a = anchors.split(axis=-1, num_outputs=4)
        p = F.split(x, axis=-1, num_outputs=4)
        ox = F.broadcast_add(F.broadcast_mul(p[0] * self._stds[0] + self._means[0], a[2]), a[0])
        oy = F.broadcast_add(F.broadcast_mul(p[1] * self._stds[1] + self._means[1], a[3]), a[1])
        tw = F.exp(p[2] * self._stds[2] + self._means[2])
        th = F.exp(p[3] * self._stds[3] + self._means[3])
        if self._clip:
            tw = F.minimum(tw, self._clip)
            th = F.minimum(th, self._clip)
        ow = F.broadcast_mul(tw, a[2]) / 2
        oh = F.broadcast_mul(th, a[3]) / 2
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
        """HybridBlock, handle multi batch correctly

        Parameters
        ----------
        samples: (B, N), value +1 (positive), -1 (negative), 0 (ignore)
        matches: (B, N), value range [0, M)
        refs: (B, M), value range [0, num_fg_class), excluding background

        Returns
        -------
        targets: (B, N), value range [0, num_fg_class + 1), including background

        """
        # samples (B, N) (+1, -1, 0: ignore), matches (B, N) [0, M), refs (B, M)
        # reshape refs (B, M) -> (B, 1, M) -> (B, N, M)
        refs = F.repeat(refs.reshape((0, 1, -1)), axis=1, repeats=matches.shape[1])
        # ids (B, N, M) -> (B, N), value [0, M + 1), 0 reserved for background class
        target_ids = F.pick(refs, matches, axis=2) + 1
        # samples 0: set ignore samples to ignore_label
        targets = F.where(samples > 0.5, target_ids, nd.ones_like(target_ids) * self._ignore_label)
        # samples -1: set negative samples to 0
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


class SigmoidClassEncoder(gluon.HybridBlock):
    """Encode class prediction labels for SigmoidCrossEntropy Loss."""
    def __init__(self, **kwargs):
        super(SigmoidClassEncoder, self).__init__(**kwargs)

    def hybrid_forward(self, F, samples):
        """Encode class prediction labels for SigmoidCrossEntropy Loss.

        Parameters
        ----------
        samples : mxnet.nd.NDArray or mxnet.sym.Symbol
            Sampling results with shape (B, N), 1:pos, 0:ignore, -1:negative

        Returns
        -------
        (mxnet.nd.NDArray, mxnet.nd.NDArray)
            (target, mask)
            target is the output label with shape (B, N), 1: pos, 0: negative, -1: ignore
            mask is the mask for label, -1(ignore) labels have mask 0, otherwise mask is 1.

        """
        # notation from samples, 1:pos, 0:ignore, -1:negative
        target = (samples + 1) / 2.
        target = F.where(F.abs(samples) < 1e-5, F.ones_like(target) * -1, target)
        # output: 1: pos, 0: negative, -1: ignore
        mask = F.where(F.abs(samples) > 1e-5, F.ones_like(samples), F.zeros_like(samples))
        return target, mask
