import mxnet as mx
from mxnet import gluon
import mxnet.ndarray as F
import mxnet.gluon.nn as nn
from mxnet.gluon import Block

from .rpn import RPN
from .rcnn import RCNN_ResNet

class FasterRCNN(RCNN_ResNet):
    """ faster RCNN """
    def __init__(self, classes, backbone, **kwargs):
        super(FasterRCNN).__init__(classes, backbone, dilated, **kwargs)
        self.rpn = RPN(in_channels, anchors_scales, anchor_ratios)
        self.region_proposal = RegionProposal()

    def forward(self, x):
        base_feat = self.base_forward(x)
        # Region Proposal for ROIs
        rpn_cls, rpn_reg = net.rpn(base_feat)
        # B, sample_size, 4, TODO: Change to contrib.Proposal operator
        rois = self.region_proposal(rpn_cls, rpn_reg)
        # ROI Pooling or TODO ROI Align
        roi_feat = mx.nd.ROIPooling(base_feat, rois, (7, 7), 1.0/self.stride)
        rcnn_cls, rcnn_reg = self.top_forward(roi_feat)
        return rcnn_cls, rcnn_reg
