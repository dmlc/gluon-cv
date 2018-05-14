"""Region Proposal Network"""
import numpy as np
import mxnet.ndarray as F
import mxnet.gluon.nn as nn
from mxnet.gluon import Block

from .anchors import generate_anchors, map_anchors
# pylint: disable=arguments-differ, unused-variable, invalid-sequence-index

class RPN(Block):
    """ RPN: region proposal network
    A FCN for making region proposals
    """
    def __init__(self, in_channels, num_anchors, **kwargs):
        super(RPN, self).__init__(**kwargs)
        self.rpn_conv = nn.Conv2D(in_channels=in_channels, channels=1024,
                                  kernel_size=(3, 3), padding=(1, 1))
        self.conv_cls = nn.Conv2D(in_channels=1024, channels=num_anchors,
                                  kernel_size=(1, 1), padding=(0, 0))
        self.conv_reg = nn.Conv2D(in_channels=1024, channels=4 * num_anchors,
                                  kernel_size=(1, 1), padding=(0, 0))

    def forward(self, data):
        # conv featuremap
        rpn_conv1 = F.relu(self.rpn_conv(data))
        # rpn classification and bbox scores
        rpn_cls = F.sigmoid(self.conv_cls(rpn_conv1))
        rpn_reg = self.conv_reg(rpn_conv1)
        return rpn_cls, rpn_reg


class RegionProposal(object):
    """Region Proposals
    output region proposals by applying predicted transforms to the anchors
    """
    def __init__(self, train=False, stride=16, ratios=F.array([0.5, 1, 2]),
                 scales=F.array([32, 64, 128, 256, 512]),
                 nms_thresh=0.7, nms_topK=1000, pre_nms_topN=6000):
        self._feat_stride = stride
        # K,4
        self._anchors = generate_anchors(scales=scales/stride, ratios=ratios)
        self._num_anchors = self._anchors.shape[0]
        self.nms_thresh = nms_thresh
        self.nms_topK = 2000 if train else nms_topK
        self.pre_nms_topN = 12000 if train else  pre_nms_topN

    def __call__(self, rpn_cls, rpn_reg, feature_shape, image_shape):
        # Get basic information of the feature and the image
        B, C, H, W = feature_shape
        assert B == 1
        img_height, img_width = image_shape[2:]

        # Recover RPN prediction with anchors
        anchors = map_anchors(self._anchors, rpn_reg.shape, img_height, img_width, rpn_reg.context)
        # (B,K,4,H,W)
        anchors = anchors.reshape((B, self._num_anchors, 4, H, W))
        # (B,H,W,K,4)
        anchors = F.transpose(anchors, (0, 3, 4, 1, 2)).reshape((-1, 4))

        # rpn_cls (B,K,H,W) => B,H,W,K,1 => -1, 1
        rpn_cls = F.transpose(rpn_cls, (0, 2, 3, 1)).reshape(-1, 1)
        rpn_reg = F.transpose(rpn_reg.reshape((B, -1, 4, H, W)), (0, 3, 4, 1, 2)).reshape(-1, 4)

        scores_np = rpn_cls.reshape(-1).asnumpy()
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN (e.g. 6000)
        if self.pre_nms_topN <= 0 or self.pre_nms_topN >= len(scores_np):
            order = np.argsort(-scores_np)
        else:
            # Avoid sorting possibly large arrays; First partition to get top K
            # unsorted and then sort just those (~20x faster for 200k scores)
            inds = np.argpartition(
                -scores_np, self.pre_nms_topN
            )[:self.pre_nms_topN]
            order = np.argsort(-scores_np[inds])
            order = inds[order]

        rpn_reg = rpn_reg[order, :]
        rpn_cls = rpn_cls[order, :]
        anchors = anchors[order, :]
        # B,H,W,K,4

        # bbox predict, B*N*W*K, 4
        rpn_bbox_pred = self.bbox_transform(anchors, rpn_reg)
        rpn_bbox_pred = self.bbox_clip(rpn_bbox_pred, img_height, img_width)

        # B, H, W, K, 4
        # rpn_bbox_pred = rpn_bbox_pred.reshape((B, H, W, self._num_anchors, 4))

        # NMS
        rpn_scores, rois = self.rpn_nms(
            rpn_cls, rpn_bbox_pred, self.nms_thresh, self.pre_nms_topN, self.nms_topK)

        # B, topK & B, topK, 4
        return rpn_scores.expand_dims(0), rois.expand_dims(0)


    @staticmethod
    def rpn_nms(anchor_scores, bbox_pred, thresh, pre_nms_topN, topK):
        """Non-maximum Suppression for RPN"""
        # B,H,W,K,5
        # B = anchor_scores.shape[0]
        data = F.concat(anchor_scores, bbox_pred, dim=1)#.reshape(B, -1, 5)
        nms_pred = F.contrib.box_nms(data, thresh, pre_nms_topN, coord_start=1,
                                     score_index=0, id_index=-1, force_suppress=True)
        # topK with effective rois
        effect = int(F.sum(nms_pred[:, 0] >= 0).asscalar())
        topK = topK if effect > topK else effect
        # B, N
        rpn_scores = nms_pred[:topK, 0]
        # B, N, 4
        rpn_bbox = nms_pred[:topK, 1:]
        return rpn_scores, rpn_bbox

    @staticmethod
    def bbox_clip(bbox, height, width):
        """Box Clipping"""
        zeros_t = F.zeros(bbox[:, 0].shape, ctx=bbox.context)
        bbox[:, 0] = F.maximum(bbox[:, 0], zeros_t)
        bbox[:, 1] = F.maximum(bbox[:, 1], zeros_t)
        bbox[:, 2] = F.minimum(bbox[:, 2], zeros_t + width - 1)
        bbox[:, 3] = F.minimum(bbox[:, 3], zeros_t + height - 1)
        return bbox

    @staticmethod
    def bbox_transform(boxes, deltas, weights=(1.0, 1.0, 1.0, 1.0), clip_value=4.135166556742356):
        """Box Transforms"""
        if boxes.shape[0] == 0:
            return None

        # get boxes dimensions and centers
        widths = boxes[:, 2] - boxes[:, 0] + 1.0
        heights = boxes[:, 3] - boxes[:, 1] + 1.0
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = weights
        dx = deltas[:, 0::4] / wx
        dy = deltas[:, 1::4] / wy
        dw = deltas[:, 2::4] / ww
        dh = deltas[:, 3::4] / wh

        # Prevent sending too large values into np.exp()
        dw = dw.clip(a_max=clip_value, a_min=dw.min().asscalar())
        dh = dh.clip(a_max=clip_value, a_min=dh.min().asscalar())

        pred_ctr_x = dx * widths.expand_dims(1) + ctr_x.expand_dims(1)
        pred_ctr_y = dy * heights.expand_dims(1) + ctr_y.expand_dims(1)
        pred_w = F.exp(dw) * widths.expand_dims(1)
        pred_h = F.exp(dh) * heights.expand_dims(1)

        # x1
        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        # y1
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        # x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w - 1
        # y2 (note: "- 1" is correct; don't be fooled by the asymmetry)
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h - 1

        pred_boxes = F.concat(pred_boxes_x1,
                              pred_boxes_y1,
                              pred_boxes_x2,
                              pred_boxes_y2, dim=1)

        return pred_boxes
