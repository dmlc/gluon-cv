"""RCNN Target Generator."""
from __future__ import absolute_import

from mxnet import gluon
from mxnet import autograd
from ...nn.coder import MultiClassEncoder, NormalizedPerClassBoxCenterEncoder


class RCNNTargetSampler(gluon.HybridBlock):
    """A sampler to choose positive/negative samples from RCNN Proposals

    Parameters
    ----------
    num_image: int
        Number of input images.
    num_proposal: int
        Number of input proposals.
    num_sample : int
        Number of samples for RCNN targets.
    pos_iou_thresh : float
        Proposal whose IOU larger than ``pos_iou_thresh`` is regarded as positive samples.
        Proposal whose IOU smaller than ``pos_iou_thresh`` is regarded as negative samples.
    pos_ratio : float
        ``pos_ratio`` defines how many positive samples (``pos_ratio * num_sample``) is
        to be sampled.
    max_num_gt : int
        Maximum ground-truth number in whole training dataset. This is only an upper bound, not
        necessarily very precise. However, using a very big number may impact the training speed.

    """
    def __init__(self, num_image, num_proposal, num_sample, pos_iou_thresh, pos_ratio, max_num_gt):
        super(RCNNTargetSampler, self).__init__()
        self._num_image = num_image
        self._num_proposal = num_proposal
        self._num_sample = num_sample
        self._max_pos = int(round(num_sample * pos_ratio))
        self._pos_iou_thresh = pos_iou_thresh
        self._max_num_gt = max_num_gt

    #pylint: disable=arguments-differ
    def hybrid_forward(self, F, rois, scores, gt_boxes):
        """Handle B=self._num_image by a for loop.

        Parameters
        ----------
        rois: (B, self._num_input, 4) encoded in (x1, y1, x2, y2).
        scores: (B, self._num_input, 1), value range [0, 1] with ignore value -1.
        gt_boxes: (B, M, 4) encoded in (x1, y1, x2, y2), invalid box should have area of 0.

        Returns
        -------
        rois: (B, self._num_sample, 4), randomly drawn from proposals
        samples: (B, self._num_sample), value +1: positive / 0: ignore / -1: negative.
        matches: (B, self._num_sample), value between [0, M)

        """
        with autograd.pause():
            # collect results into list
            new_rois = []
            new_samples = []
            new_matches = []
            for i in range(self._num_image):
                roi = F.squeeze(F.slice_axis(rois, axis=0, begin=i, end=i+1), axis=0)
                score = F.squeeze(F.slice_axis(scores, axis=0, begin=i, end=i+1), axis=0)
                gt_box = F.squeeze(F.slice_axis(gt_boxes, axis=0, begin=i, end=i+1), axis=0)
                gt_score = F.ones_like(F.sum(gt_box, axis=-1, keepdims=True))

                # concat rpn roi with ground truth
                all_roi = F.concat(roi, gt_box, dim=0)
                all_score = F.concat(score, gt_score, dim=0).squeeze(axis=-1)
                # calculate (N, M) ious between (N, 4) anchors and (M, 4) bbox ground-truths
                # cannot do batch op, will get (B, N, B, M) ious
                ious = F.contrib.box_iou(all_roi, gt_box, format='corner')
                # match to argmax iou
                ious_max = ious.max(axis=-1)
                ious_argmax = ious.argmax(axis=-1)
                # init with 2, which are neg samples
                mask = F.ones_like(ious_max) * 2
                # mark all ignore to 0
                mask = F.where(all_score < 0, F.zeros_like(mask), mask)
                # mark positive samples with 3
                pos_mask = ious_max >= self._pos_iou_thresh
                mask = F.where(pos_mask, F.ones_like(mask) * 3, mask)

                # shuffle mask
                rand = F.random.uniform(0, 1, shape=(self._num_proposal + self._max_num_gt,))
                rand = F.slice_like(rand, ious_argmax)
                index = F.argsort(rand)
                mask = F.take(mask, index)
                ious_argmax = F.take(ious_argmax, index)

                # sample pos samples
                order = F.argsort(mask, is_ascend=False)
                topk = F.slice_axis(order, axis=0, begin=0, end=self._max_pos)
                topk_indices = F.take(index, topk)
                topk_samples = F.take(mask, topk)
                topk_matches = F.take(ious_argmax, topk)
                # reset output: 3 pos 2 neg 0 ignore -> 1 pos -1 neg 0 ignore
                topk_samples = F.where(topk_samples == 3,
                                       F.ones_like(topk_samples), topk_samples)
                topk_samples = F.where(topk_samples == 2,
                                       F.ones_like(topk_samples) * -1, topk_samples)

                # sample neg samples
                index = F.slice_axis(index, axis=0, begin=self._max_pos, end=None)
                mask = F.slice_axis(mask, axis=0, begin=self._max_pos, end=None)
                ious_argmax = F.slice_axis(ious_argmax, axis=0, begin=self._max_pos, end=None)
                # change mask: 4 neg 3 pos 0 ignore
                mask = F.where(mask == 2, F.ones_like(mask) * 4, mask)
                order = F.argsort(mask, is_ascend=False)
                num_neg = self._num_sample - self._max_pos
                bottomk = F.slice_axis(order, axis=0, begin=0, end=num_neg)
                bottomk_indices = F.take(index, bottomk)
                bottomk_samples = F.take(mask, bottomk)
                bottomk_matches = F.take(ious_argmax, bottomk)
                # reset output: 4 neg 3 pos 0 ignore -> 1 pos -1 neg 0 ignore
                bottomk_samples = F.where(bottomk_samples == 3,
                                          F.ones_like(bottomk_samples), bottomk_samples)
                bottomk_samples = F.where(bottomk_samples == 4,
                                          F.ones_like(bottomk_samples) * -1, bottomk_samples)

                # output
                indices = F.concat(topk_indices, bottomk_indices, dim=0)
                samples = F.concat(topk_samples, bottomk_samples, dim=0)
                matches = F.concat(topk_matches, bottomk_matches, dim=0)

                new_rois.append(all_roi.take(indices))
                new_samples.append(samples)
                new_matches.append(matches)
            # stack all samples together
            new_rois = F.stack(*new_rois, axis=0)
            new_samples = F.stack(*new_samples, axis=0)
            new_matches = F.stack(*new_matches, axis=0)
        return new_rois, new_samples, new_matches


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
        """Components can handle batch images

        Parameters
        ----------
        roi: (B, N, 4), input proposals
        samples: (B, N), value +1: positive / -1: negative.
        matches: (B, N), value [0, M), index to gt_label and gt_box.
        gt_label: (B, M), value [0, num_class), excluding background class.
        gt_box: (B, M, 4), input ground truth box corner coordinates.

        Returns
        -------
        cls_target: (B, N), value [0, num_class + 1), including background.
        box_target: (B, N, C, 4), only foreground class has nonzero target.
        box_weight: (B, N, C, 4), only foreground class has nonzero weight.

        """
        with autograd.pause():
            # cls_target (B, N)
            cls_target = self._cls_encoder(samples, matches, gt_label)
            # box_target, box_weight (C, B, N, 4)
            box_target, box_mask = self._box_encoder(
                samples, matches, roi, gt_label, gt_box)
            # modify shapes to match predictions
            # box (C, B, N, 4) -> (B, N, C, 4)
            box_target = box_target.transpose((1, 2, 0, 3))
            box_mask = box_mask.transpose((1, 2, 0, 3))
        return cls_target, box_target, box_mask
