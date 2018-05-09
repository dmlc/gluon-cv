import mxnet as mx
from mxnet import gluon
import mxnet.ndarray as F
import mxnet.gluon.nn as nn
from mxnet.gluon import Block

from .anchors import generate_anchors, map_anchors
from .bbox import bbox_inverse_transform, bbox_clip

class RPN(Block):
    """ RPN: region proposal network
    A FCN for making region proposals
    """
    def __init__(self, in_channels, num_anchors, **kwargs):
        super(RPN, self).__init__(**kwargs)
        self.rpn_conv = nn.Conv2D(in_channels=in_channels, channels=512,
                                  kernel_size=(3, 3), padding=(1, 1))
        self.conv_cls = nn.Conv2D(in_channels=512, channels=2 * num_anchors,
                                  kernel_size=(1, 1),padding=(0, 0))
        self.conv_reg = nn.Conv2D(in_channels=512, channels=4 * num_anchors,
                                  kernel_size=(1, 1),padding=(0, 0))

    def forward(self, data, *args):
        # conv featuremap
        rpn_conv1 = F.relu(self.rpn_conv(data))
        # rpn classification and bbox scores
        rpn_cls = self.conv_cls(rpn_conv1)
        rpn_reg = self.conv_reg(rpn_conv1)
        return rpn_cls, rpn_reg


class RegionProposal(object):
    """Region Proposals
    output region proposals by applying predicted transforms to the anchors
    """
    def __init__(self, stride, ratios=F.array([0.5, 1, 2]), scales=F.array([8, 16, 32]),
                 nms_thresh=0.7, nms_topk=300, pre_nms_topN=6000):
        self._feat_stride = stride
        # K,4
        self._anchors = generate_anchors(scales=scales, ratios=ratios)
        self._num_anchors = self._anchors.shape[0]
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.pre_nms_topN = pre_nms_topN

    def __call__(self, rpn_cls, rpn_reg, feature_shape, image_shape):
        # Get basic information of the feature and the image
        B, C, H, W = feature_shape
        img_height, img_width = image_shape[2:]
        # B,K,2,H,W
        rpn_cls = rpn_cls.reshape((B, -1, 2, H, W))
        
        # Recover RPN prediction with anchors
        anchors = map_anchors(self._anchors, rpn_reg.shape, img_height, img_width, rpn_cls.context)
        # B,K,4,H,W
        anchors = anchors.reshape((B, -1, 4, H, W))
        # B,H,W,K,4
        anchors = F.transpose(anchors, (0, 3, 4, 1, 2))
        # B,H,W,K
        rpn_anchor_scores = F.softmax(F.transpose(rpn_cls, (0, 3, 4, 1, 2)), axis=4)[:,:,:,:,1]
        # B,H,W,K,4
        rpn_reg = F.transpose(rpn_reg.reshape((B, -1, 4, H, W)), (0, 3, 4, 1, 2))

        # bbox predict, B*N*W*K, 4
        rpn_bbox_pred = bbox_inverse_transform(anchors.reshape((-1, 4)), rpn_reg.reshape((-1, 4)))
        rpn_bbox_pred = bbox_clip(rpn_bbox_pred, img_height, img_width)

        # B, H, W, K, 4
        rpn_bbox_pred = rpn_bbox_pred.reshape((B, H, W, self._num_anchors, 4))

        # self.pre_nms_topN?
        # NMS
        rpn_scores, rois = self.rpn_nms(
            rpn_anchor_scores, rpn_bbox_pred, self.nms_thresh, self.nms_topk)

        # B, topK & B, topK, 4
        return rpn_scores, rois


    @staticmethod
    def rpn_nms(anchor_scores, bbox_pred, thresh, topk):
        """Non-maximum Suppression for RPN"""
        anchor_scores = anchor_scores.expand_dims(4)
        # B,H,W,K,5
        B = anchor_scores.shape[0]
        data = F.concat(anchor_scores, bbox_pred, dim=4).reshape(B, -1, 5)
        nms_pred = F.contrib.box_nms(data, thresh, topk, coord_start=1,
                                     score_index=0, id_index=-1, force_suppress=True)
        # B, N
        rpn_scores = nms_pred[:,:topk,0]
        # B, N, 4
        rpn_bbox = nms_pred[:,:topk,1:]
        return rpn_scores, rpn_bbox
