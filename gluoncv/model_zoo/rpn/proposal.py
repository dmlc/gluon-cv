"""RPN proposals."""
from __future__  import absolute_import

from mxnet import autograd
from mxnet import gluon
from ...nn.bbox import BBoxCornerToCenter, BBoxClipToImage
from ...nn.coder import NormalizedBoxCenterDecoder


class RPNProposal(gluon.HybridBlock):
    """Proposal generator for RPN.

    RPNProposal takes RPN anchors, RPN prediction scores and box regression preditions.
    It will transform anchors, apply NMS to get clean foreground proposals.

    Parameters
    ----------
    clip : float
        Clip bounding box target to this value.
    nms_thresh : float
        IOU threshold for NMS. It is used to remove overlapping proposals.
    train_pre_nms : int
        Filter top proposals before NMS in training.
    train_post_nms : int
        Return top proposal results after NMS in training.
    test_pre_nms : int
        Filter top proposals before NMS in testing.
    test_post_nms : int
        Return top proposal results after NMS in testing.
    min_size : int
        Proposals whose size is smaller than ``min_size`` will be discarded.
    stds : tuple of float
        Standard deviation to be multiplied from encoded regression targets.
        These values must be the same as stds used in RPNTargetGenerator.
    """
    def __init__(self, clip, nms_thresh, train_pre_nms, train_post_nms,
                 test_pre_nms, test_post_nms, min_size, stds):
        super(RPNProposal, self).__init__()
        self._box_to_center = BBoxCornerToCenter()
        self._box_decoder = NormalizedBoxCenterDecoder(stds=stds, clip=clip)
        self._clipper = BBoxClipToImage()
        # self._compute_area = BBoxArea()
        self._nms_thresh = nms_thresh
        self._train_pre_nms = max(1, train_pre_nms)
        self._train_post_nms = max(1, train_post_nms)
        self._test_pre_nms = max(1, test_pre_nms)
        self._test_post_nms = max(1, test_post_nms)
        self._min_size = min_size

    #pylint: disable=arguments-differ
    def hybrid_forward(self, F, anchor, score, bbox_pred, img):
        """
        Generate proposals. Limit to batch-size=1 in current implementation.
        """
        if autograd.is_training():
            pre_nms = self._train_pre_nms
            post_nms = self._train_post_nms
        else:
            pre_nms = self._test_pre_nms
            post_nms = self._test_post_nms

        with autograd.pause():
            # restore bounding boxes
            roi = self._box_decoder(bbox_pred, self._box_to_center(anchor))

            # clip rois to image's boundary
            # roi = F.Custom(roi, img, op_type='bbox_clip_to_image')
            roi = self._clipper(roi, img)

            # remove bounding boxes that don't meet the min_size constraint
            # by setting them to (-1, -1, -1, -1)
            # width = roi.slice_axis(axis=-1, begin=2, end=3)
            # height = roi.slice_axis(axis=-1, begin=3, end=None)
            xmin, ymin, xmax, ymax = roi.split(axis=-1, num_outputs=4)
            width = xmax - xmin
            height = ymax - ymin
            # TODO:(zhreshold), there's im_ratio to handle here, but it requires
            # add' info, and we don't expect big difference
            invalid = (width < self._min_size) + (height < self._min_size)

            # # remove out of bound anchors
            # axmin, aymin, axmax, aymax = F.split(anchor, axis=-1, num_outputs=4)
            # # it's a bit tricky to get right/bottom boundary in hybridblock
            # wrange = F.arange(0, 2560).reshape((1, 1, 1, 2560)).slice_like(
            #    img, axes=(3)).max().reshape((1, 1, 1))
            # hrange = F.arange(0, 2560).reshape((1, 1, 2560, 1)).slice_like(
            #    img, axes=(2)).max().reshape((1, 1, 1))
            # invalid = (axmin < 0) + (aymin < 0) + F.broadcast_greater(axmax, wrange) + \
            #    F.broadcast_greater(aymax, hrange)
            score = F.where(invalid, F.zeros_like(invalid), score)
            invalid = F.repeat(invalid, axis=-1, repeats=4)
            roi = F.where(invalid, F.ones_like(invalid) * -1, roi)

            # Non-maximum suppression
            pre = F.concat(score, roi, dim=-1)
            tmp = F.contrib.box_nms(pre, overlap_thresh=self._nms_thresh, topk=pre_nms,
                                    coord_start=1, score_index=0, id_index=-1, force_suppress=True)

            # slice post_nms number of boxes
            result = F.slice_axis(tmp, axis=1, begin=0, end=post_nms)
            rpn_scores = F.slice_axis(result, axis=-1, begin=0, end=1)
            rpn_bbox = F.slice_axis(result, axis=-1, begin=1, end=None)

        return rpn_scores, rpn_bbox
