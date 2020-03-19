# pylint: disable=arguments-differ, missing-docstring
"""Encoder and Decoder functions.
Encoders are used during training, which assign training targets.
Decoders are used during testing/validation, which convert predictions back to
normal boxes, etc.
"""
from __future__ import absolute_import

import numpy as np
from mxnet import gluon
from mxnet import nd

from .bbox import BBoxCornerToCenter, NumPyBBoxCornerToCenter

try:
    import cython_bbox
except ImportError:
    cython_bbox = None


class NumPyNormalizedBoxCenterEncoder(object):
    """Encode bounding boxes training target with normalized center offsets using numpy.

    Input bounding boxes are using corner type: `x_{min}, y_{min}, x_{max}, y_{max}`.

    Parameters
    ----------
    stds : array-like of size 4
        Std value to be divided from encoded values, default is (0.1, 0.1, 0.2, 0.2).
    means : array-like of size 4
        Mean value to be subtracted from encoded values, default is (0., 0., 0., 0.).

    """

    def __init__(self, stds=(0.1, 0.1, 0.2, 0.2), means=(0., 0., 0., 0.)):
        super(NumPyNormalizedBoxCenterEncoder, self).__init__()
        assert len(stds) == 4, "Box Encoder requires 4 std values."
        self._stds = stds
        self._means = means
        self.corner_to_center = NumPyBBoxCornerToCenter(split=True)

    def __call__(self, samples, matches, anchors, refs):
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
        if cython_bbox is not None:
            return cython_bbox.np_normalized_box_encoder(samples, matches, anchors, refs,
                                                         np.array(self._means, dtype=np.float32),
                                                         np.array(self._stds, dtype=np.float32))
        # refs [B, M, 4], anchors [B, N, 4], samples [B, N], matches [B, N]
        ref_boxes = np.repeat(refs.reshape((refs.shape[0], 1, -1, 4)), axis=1,
                              repeats=matches.shape[1])
        # refs [B, N, M, 4] -> [B, N, 4]
        ref_boxes = \
            ref_boxes[:, range(matches.shape[1]), matches, :] \
                .reshape(matches.shape[0], -1, 4)
        # g [B, N, 4], a [B, N, 4] -> codecs [B, N, 4]
        g = self.corner_to_center(ref_boxes)
        a = self.corner_to_center(anchors)
        t0 = ((g[0] - a[0]) / a[2] - self._means[0]) / self._stds[0]
        t1 = ((g[1] - a[1]) / a[3] - self._means[1]) / self._stds[1]
        t2 = (np.log(g[2] / a[2]) - self._means[2]) / self._stds[2]
        t3 = (np.log(g[3] / a[3]) - self._means[3]) / self._stds[3]
        codecs = np.concatenate((t0, t1, t2, t3), axis=2)
        # samples [B, N] -> [B, N, 1] -> [B, N, 4] -> boolean
        temp = np.tile(samples.reshape((samples.shape[0], -1, 1)), reps=(1, 1, 4)) > 0.5
        # fill targets and masks [B, N, 4]
        targets = np.where(temp, codecs, 0.0)
        masks = np.where(temp, 1.0, 0.0)
        return targets, masks


class NormalizedBoxCenterEncoder(gluon.HybridBlock):
    """Encode bounding boxes training target with normalized center offsets.

    Input bounding boxes are using corner type: `x_{min}, y_{min}, x_{max}, y_{max}`.

    Parameters
    ----------
    stds : array-like of size 4
        Std value to be divided from encoded values, default is (0.1, 0.1, 0.2, 0.2).
    means : array-like of size 4
        Mean value to be subtracted from encoded values, default is (0., 0., 0., 0.).

    """

    def __init__(self, stds=(0.1, 0.1, 0.2, 0.2), means=(0., 0., 0., 0.), **kwargs):
        super(NormalizedBoxCenterEncoder, self).__init__(**kwargs)
        assert len(stds) == 4, "Box Encoder requires 4 std values."
        assert len(means) == 4, "Box Encoder requires 4 std values."
        self._means = means
        self._stds = stds
        with self.name_scope():
            self.corner_to_center = BBoxCornerToCenter(split=True)

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, samples, matches, anchors, refs):
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
        # TODO(zhreshold): batch_pick, take multiple elements?
        # refs [B, M, 4], anchors [B, N, 4], samples [B, N], matches [B, N]
        # refs [B, M, 4] -> reshape [B, 1, M, 4] -> repeat [B, N, M, 4]
        ref_boxes = F.broadcast_like(refs.reshape((0, 1, -1, 4)), matches, lhs_axes=1, rhs_axes=1)
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


class NormalizedPerClassBoxCenterEncoder(gluon.HybridBlock):
    """Encode bounding boxes training target with normalized center offsets.

    Input bounding boxes are using corner type: `x_{min}, y_{min}, x_{max}, y_{max}`.

    Parameters
    ----------
    max_pos : int, default is 128
        Upper bound of Number of positive samples.
    per_device_batch_size : int, default is 1
        Per device batch size
    stds : array-like of size 4
        Std value to be divided from encoded values, default is (0.1, 0.1, 0.2, 0.2).
    means : array-like of size 4
        Mean value to be subtracted from encoded values, default is (0., 0., 0., 0.).

    """

    def __init__(self, num_class, max_pos=128, per_device_batch_size=1, stds=(0.1, 0.1, 0.2, 0.2),
                 means=(0., 0., 0., 0.)):
        super(NormalizedPerClassBoxCenterEncoder, self).__init__()
        assert len(stds) == 4, "Box Encoder requires 4 std values."
        assert num_class > 0, "Number of classes must be positive"
        self._num_class = num_class
        self._max_pos = max_pos
        self._batch_size = per_device_batch_size
        with self.name_scope():
            self.class_agnostic_encoder = NormalizedBoxCenterEncoder(stds=stds, means=means)
            if 'box_encode' in nd.contrib.__dict__:
                self.means = self.params.get_constant('means', means)
                self.stds = self.params.get_constant('stds', stds)

    def hybrid_forward(self, F, samples, matches, anchors, labels, refs, means=None, stds=None):
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
        targets: (B, N_pos, C, 4) transform anchors to refs picked according to matches
        masks: (B, N_pos, C, 4) only positive anchors of the correct class has targets
        indices : (B, N_pos) positive sample indices

        """
        # refs [B, M, 4], anchors [B, N, 4], samples [B, N], matches [B, N]
        # encoded targets [B, N, 4], masks [B, N, 4]
        if 'box_encode' in F.contrib.__dict__:
            targets, masks = F.contrib.box_encode(samples, matches, anchors, refs, means, stds)
        else:
            targets, masks = self.class_agnostic_encoder(samples, matches, anchors, refs)

        # labels [B, M] -> [B, N, M]
        ref_labels = F.broadcast_like(labels.reshape((0, 1, -1)), matches, lhs_axes=1, rhs_axes=1)
        # labels [B, N, M] -> pick from matches [B, N] -> [B, N, 1]
        ref_labels = F.pick(ref_labels, matches, axis=2).reshape((0, -1)).expand_dims(2)
        # boolean array [B, N, C]
        same_cids = F.broadcast_equal(ref_labels, F.reshape(F.arange(self._num_class),
                                                            shape=(1, 1, -1)))

        # reduce box targets to positive samples only
        indices = F.slice_axis(
            F.reshape(F.argsort(F.slice_axis(masks, axis=-1, begin=0, end=1), axis=1,
                                is_ascend=False), (self._batch_size, -1)),
            axis=1, begin=0, end=self._max_pos)
        targets_tmp = []
        masks_tmp = []
        same_cids_tmp = []
        for i in range(self._batch_size):
            ind = F.slice_axis(indices, axis=0, begin=i, end=i + 1).squeeze(axis=0)
            target = F.slice_axis(targets, axis=0, begin=i, end=i + 1).squeeze(axis=0)
            mask = F.slice_axis(masks, axis=0, begin=i, end=i + 1).squeeze(axis=0)
            same_cid = F.slice_axis(same_cids, axis=0, begin=i, end=i + 1).squeeze(axis=0)
            targets_tmp.append(F.take(target, ind).expand_dims(axis=0))
            masks_tmp.append(F.take(mask, ind).expand_dims(axis=0))
            same_cids_tmp.append(F.take(same_cid, ind).expand_dims(axis=0))
        targets = F.concat(*targets_tmp, dim=0)
        masks = F.concat(*masks_tmp, dim=0)
        same_cids = F.concat(*same_cids_tmp, dim=0).expand_dims(3)

        # targets, masks [B, N_pos, C, 4]
        all_targets = F.broadcast_axes(targets.expand_dims(2), axis=2, size=self._num_class)
        all_masks = F.broadcast_mul(masks.expand_dims(2),
                                    F.broadcast_axes(same_cids, axis=3, size=4))
        return all_targets, all_masks, indices


class NormalizedBoxCenterDecoder(gluon.HybridBlock):
    """Decode bounding boxes training target with normalized center offsets.
    This decoder must cooperate with NormalizedBoxCenterEncoder of same `stds`
    in order to get properly reconstructed bounding boxes.

    Returned bounding boxes are using corner type: `x_{min}, y_{min}, x_{max}, y_{max}`.

    Parameters
    ----------
    stds : array-like of size 4
        Std value to be divided from encoded values, default is (0.1, 0.1, 0.2, 0.2).
    clip : float, default is None
        If given, bounding box target will be clipped to this value.
    convert_anchor : boolean, default is False
        Whether to convert anchor from corner to center format.
    minimal_opset : bool
        We sometimes add special operators to accelerate training/inference, however, for exporting
        to third party compilers we want to utilize most widely used operators.
        If `minimal_opset` is `True`, the network will use a minimal set of operators good
        for e.g., `TVM`.
    """

    def __init__(self, stds=(0.1, 0.1, 0.2, 0.2), convert_anchor=False, clip=None,
                 minimal_opset=False):
        super(NormalizedBoxCenterDecoder, self).__init__()
        assert len(stds) == 4, "Box Encoder requires 4 std values."
        self._stds = stds
        self._clip = clip
        if convert_anchor:
            self.corner_to_center = BBoxCornerToCenter(split=True)
        else:
            self.corner_to_center = None
        self._format = 'corner' if convert_anchor else 'center'
        self._minimal_opset = minimal_opset

    def hybrid_forward(self, F, x, anchors):
        if not self._minimal_opset and 'box_decode' in F.contrib.__dict__:
            x, anchors = F.amp_multicast(x, anchors, num_outputs=2, cast_narrow=True)
            if self._clip is None:
                self._clip = -1  # match the signature of c++ operator
            return F.contrib.box_decode(x, anchors, self._stds[0], self._stds[1], self._stds[2],
                                        self._stds[3], clip=self._clip, format=self._format)
        if self.corner_to_center is not None:
            a = self.corner_to_center(anchors)
        else:
            a = anchors.split(axis=-1, num_outputs=4)
        p = F.split(x, axis=-1, num_outputs=4)
        ox = F.broadcast_add(F.broadcast_mul(p[0] * self._stds[0], a[2]), a[0])
        oy = F.broadcast_add(F.broadcast_mul(p[1] * self._stds[1], a[3]), a[1])
        dw = p[2] * self._stds[2]
        dh = p[3] * self._stds[3]
        if self._clip:
            dw = F.minimum(dw, self._clip)
            dh = F.minimum(dh, self._clip)
        dw = F.exp(dw)
        dh = F.exp(dh)
        ow = F.broadcast_mul(dw, a[2]) * 0.5
        oh = F.broadcast_mul(dh, a[3]) * 0.5
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
        refs = F.broadcast_like(F.reshape(refs, (0, 1, -1)), matches, lhs_axes=1, rhs_axes=1)
        # ids (B, N, M) -> (B, N), value [0, M + 1), 0 reserved for background class
        target_ids = F.pick(refs, matches, axis=2) + 1
        # samples 0: set ignore samples to ignore_label
        targets = F.where(samples > 0.5, target_ids, F.ones_like(target_ids) * self._ignore_label)
        # samples -1: set negative samples to 0
        targets = F.where(samples < -0.5, F.zeros_like(targets), targets)
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
        cls_id = F.broadcast_add(template,
                                 F.reshape(F.arange(self._fg_class), shape=(1, 1, self._fg_class)))
        mask = scores > self._thresh
        cls_id = F.where(mask, cls_id, F.ones_like(cls_id) * -1)
        scores = F.where(mask, scores, F.zeros_like(scores))
        return cls_id, scores


class SigmoidClassEncoder(object):
    """Encode class prediction labels for SigmoidCrossEntropy Loss."""

    def __init__(self, **kwargs):
        super(SigmoidClassEncoder, self).__init__(**kwargs)

    def __call__(self, samples):
        """Encode class prediction labels for SigmoidCrossEntropy Loss.

        Parameters
        ----------
        samples : np.array
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
        target = np.where(np.abs(samples) < 1e-5, -1, target)
        # output: 1: pos, 0: negative, -1: ignore
        mask = np.where(np.abs(samples) > 1e-5, 1.0, 0.0)
        return target, mask


class CenterNetDecoder(gluon.HybridBlock):
    """Decorder for centernet.

    Parameters
    ----------
    topk : int
        Only keep `topk` results.
    scale : float, default is 4.0
        Downsampling scale for the network.

    """
    def __init__(self, topk=100, scale=4.0):
        super(CenterNetDecoder, self).__init__()
        self._topk = topk
        self._scale = scale

    def hybrid_forward(self, F, x, wh, reg):
        """Forward of decoder"""
        _, _, out_h, out_w = x.shape_array().split(num_outputs=4, axis=0)
        scores, indices = x.reshape((0, -1)).topk(k=self._topk, ret_typ='both')
        indices = F.cast(indices, 'int64')
        topk_classes = F.cast(F.broadcast_div(indices, (out_h * out_w)), 'float32')
        topk_indices = F.broadcast_mod(indices, (out_h * out_w))
        topk_ys = F.broadcast_div(topk_indices, out_w)
        topk_xs = F.broadcast_mod(topk_indices, out_w)
        center = reg.transpose((0, 2, 3, 1)).reshape((0, -1, 2))
        wh = wh.transpose((0, 2, 3, 1)).reshape((0, -1, 2))
        batch_indices = F.cast(F.arange(256).slice_like(
            center, axes=(0)).expand_dims(-1).tile(reps=(1, self._topk)), 'int64')
        reg_xs_indices = F.zeros_like(batch_indices, dtype='int64')
        reg_ys_indices = F.ones_like(batch_indices, dtype='int64')
        reg_xs = F.concat(batch_indices, topk_indices, reg_xs_indices, dim=0).reshape((3, -1))
        reg_ys = F.concat(batch_indices, topk_indices, reg_ys_indices, dim=0).reshape((3, -1))
        xs = F.cast(F.gather_nd(center, reg_xs).reshape((-1, self._topk)), 'float32')
        ys = F.cast(F.gather_nd(center, reg_ys).reshape((-1, self._topk)), 'float32')
        topk_xs = F.cast(topk_xs, 'float32') + xs
        topk_ys = F.cast(topk_ys, 'float32') + ys
        w = F.cast(F.gather_nd(wh, reg_xs).reshape((-1, self._topk)), 'float32')
        h = F.cast(F.gather_nd(wh, reg_ys).reshape((-1, self._topk)), 'float32')
        half_w = w / 2
        half_h = h / 2
        results = [topk_xs - half_w, topk_ys - half_h, topk_xs + half_w, topk_ys + half_h]
        results = F.concat(*[tmp.expand_dims(-1) for tmp in results], dim=-1)
        return topk_classes, scores, results * self._scale
