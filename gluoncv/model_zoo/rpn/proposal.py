"""RPN proposals."""
from __future__  import absolute_import

from mxnet import autograd
from mxnet import gluon
from ...nn.bbox import BBoxCornerToCenter
from ...nn.coder import NormalizedBoxCenterDecoder


class RPNProposal(gluon.HybridBlock):
    def __init__(self, nms_thresh=0.7, train_pre_nms=12000, train_post_nms=2000,
                 test_pre_nms=6000, test_post_nms=300, min_size=16, stds=(1., 1., 1., 1.)):
        super(RPNProposal, self).__init__()
        self._box_to_center = BBoxCornerToCenter()
        self._box_decoder = NormalizedBoxCenterDecoder(stds=stds)
        # self._clipper = BBoxClipToImage()
        # self._compute_area = BBoxArea()
        self._nms_thresh = nms_thresh
        self._train_pre_nms = max(1, train_pre_nms)
        self._train_post_nms = max(1, train_post_nms)
        self._test_pre_nms = max(1, test_pre_nms)
        self._test_post_nms = max(1, test_post_nms)
        self._min_size = min_size

    def hybrid_forward(self, F, anchor, score, bbox_pred, img):
        """
        Limit to batch-size=1
        """
        if autograd.is_training():
            pre_nms = self._train_pre_nms
            post_nms = self._train_post_nms
        else:
            pre_nms = self._test_pre_nms
            post_nms = self._test_post_nms

        # restore bounding boxes
        roi = self._box_decoder(bbox_pred, self._box_to_center(anchor))

        # clip rois to image's boundary
        roi = F.Custom(roi, img, op_type='bbox_clip_to_image')
        # roi = self._clipper(roi, width, height)

        # remove bounding boxes that don't meet the min_size constraint
        # by setting them to (-1, -1, -1, -1)
        width = roi.slice_axis(axis=-1, begin=2, end=3)
        height = roi.slice_axis(axis=-1, begin=3, end=None)
        invalid = (width < self._min_size) + (height < self._min_size)
        score = F.where(invalid, F.zeros_like(invalid), score)
        invalid = F.repeat(invalid, axis=-1, repeats=4)
        roi = F.where(invalid, F.ones_like(invalid) * -1, roi)

        # Non-maximum suppression
        pre = F.concat(score, roi, dim=-1)
        tmp = F.contrib.box_nms(pre, self._nms_thresh, pre_nms, coord_start=1,
                                score_index=0, id_index=-1, force_suppress=True)

        # slice post_nms number of boxes
        result = F.slice_axis(tmp, axis=1, begin=0, end=post_nms)
        rpn_scores = F.slice_axis(result, axis=-1, begin=0, end=1)
        rpn_bbox = F.slice_axis(result, axis=-1, begin=1, end=None)

        return rpn_scores, rpn_bbox
