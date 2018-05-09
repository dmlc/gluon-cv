"""RPN proposals."""
from __future__  import absolute_import

from mxnet import autograd
from mxnet import gluon


class RPNProposal(gluon.HybridBlock):
    def __init__(self, nms_thresh=0.7, train_pre_nms=12000, train_post_nms=2000,
                 test_pre_nms=6000, test_post_nms=300, min_size=16):
        self._nms_thresh = nms_thresh
        self._train_pre_nms = max(1, train_pre_nms)
        self._train_post_nms = max(1, train_post_nms)
        self._test_pre_nms = max(1, test_pre_nms)
        self._test_post_nms = max(1, test_post_nms)
        self._min_size = min_size

    def hybrid_forward(self, F, cls_score, bbox_pred, im_info, scale=1.0):
        if autograd.is_training():
            roi = F.contrib.Proposal(
                cls_score, bbox_pred, im_info, rpn_pre_nms_top_n=self._train_pre_nms,
                rpn_post_nms_top_n=self._train_post_nms, threshold=self._nms_thresh,
                rpn_min_size=self._min_size * scale)
