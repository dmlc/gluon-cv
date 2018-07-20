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

    """
    def __init__(self, num_image, num_proposal, num_sample, pos_iou_thresh, pos_ratio):
        super(RCNNTargetSampler, self).__init__()
        self._num_image = num_image
        self._num_proposal = num_proposal
        self._num_sample = num_sample
        self._max_pos = int(round(num_sample * pos_ratio))
        self._pos_iou_thresh = pos_iou_thresh

    #pylint: disable=arguments-differ
    def hybrid_forward(self, F, rois, gt_boxes):
        """Handle B=self._num_image by a for loop.

        Parameters
        ----------
        rois: (B, self._num_input, 4) encoded in (x1, y1, x2, y2).
        gt_boxes: (B, M, 4) encoded in (x1, y1, x2, y2), invalid box should have area of 0.

        Returns
        -------
        rois: (B, self._num_sample, 4), randomly drawn from proposals
        samples: (B, self._num_sample), value +1: positive / -1: negative.
        matches: (B, self._num_sample), value between [0, M)

        """
        with autograd.pause():
            # collect results into list
            new_rois = []
            new_samples = []
            new_matches = []
            for i in range(self._num_image):
                roi = F.squeeze(F.slice_axis(rois, axis=0, begin=i, end=i+1), axis=0)
                gt_box = F.squeeze(F.slice_axis(gt_boxes, axis=0, begin=i, end=i+1), axis=0)

                # concat rpn roi with ground truth
                all_roi = F.concat(roi, gt_box, dim=0)
                # calculate (N, M) ious between (N, 4) anchors and (M, 4) bbox ground-truths
                # cannot do batch op, will get (B, N, B, M) ious
                ious = F.contrib.box_iou(all_roi, gt_box, format='corner')
                # match to argmax iou
                ious_max = ious.max(axis=-1)
                ious_argmax = ious.argmax(axis=-1)
                # init with -1, which are neg samples
                mask = F.ones_like(ious_max) * -1
                # mark positive samples with 1
                pos_mask = ious_max >= self._pos_iou_thresh
                mask = F.where(pos_mask, F.ones_like(mask), mask)

                # shuffle mask
                rand = F.random.uniform(0, 1, shape=(self._num_proposal + 100,))
                rand = F.slice_like(rand, ious_argmax)
                index = F.argsort(rand)
                mask = F.take(mask, index)
                ious_argmax = F.take(ious_argmax, index)

                # sample pos and neg samples
                order = F.argsort(mask, is_ascend=False)
                topk = F.slice_axis(order, axis=0, begin=0, end=self._max_pos)
                num_neg = self._num_sample - self._max_pos
                bottomk = F.slice_axis(order, axis=0, begin=-num_neg, end=None)
                selected = F.concat(topk, bottomk, dim=0)

                # output
                indices = F.take(index, selected)
                samples = F.take(mask, selected)
                matches = F.take(ious_argmax, selected)

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
