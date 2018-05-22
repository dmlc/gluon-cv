"""RCNN Base Model"""
import numpy as np
from mxnet import gluon
import mxnet.ndarray as F
import mxnet.gluon.nn as nn

from ..resnetv1b import resnet50_v1b, resnet101_v1b, resnet152_v1b
from . import rpn
# pylint: disable=unused-argument, invalid-sequence-index, pointless-string-statement

class RCNN_ResNet(gluon.Block):
    """RCNN Base model"""
    def __init__(self, classes, backbone, dilated=False, **kwargs):
        super(RCNN_ResNet, self).__init__(**kwargs)
        self.classes = classes
        self.stride = 8 if dilated else 16
        with self.name_scope():
            # base network
            if backbone == 'resnet50':
                pretrained = resnet50_v1b(pretrained=True, dilated=dilated, **kwargs)
            elif backbone == 'resnet101':
                pretrained = resnet101_v1b(pretrained=True, dilated=dilated, **kwargs)
            elif backbone == 'resnet152':
                pretrained = resnet152_v1b(pretrained=True, dilated=dilated, **kwargs)
            else:
                raise RuntimeError('unknown backbone: {}'.format(backbone))
            self.conv1 = pretrained.conv1
            self.bn1 = pretrained.bn1
            self.relu = pretrained.relu
            self.maxpool = pretrained.maxpool
            self.layer1 = pretrained.layer1
            self.layer2 = pretrained.layer2
            self.layer3 = pretrained.layer3
            self.layer4 = pretrained.layer4

            # TODO FIXME, disable after testing
            # hacky for load caffe pretrained weight
            self.layer2[0].conv1._kwargs['stride'] = (2, 2)
            self.layer2[0].conv2._kwargs['stride'] = (1, 1)
            self.layer3[0].conv1._kwargs['stride'] = (2, 2)
            self.layer3[0].conv2._kwargs['stride'] = (1, 1)
            self.layer4[0].conv1._kwargs['stride'] = (2, 2)
            self.layer4[0].conv2._kwargs['stride'] = (1, 1)

            # RCNN cls and bbox reg
            self.conv_cls = nn.Dense(in_units=2048, units=classes)
            self.conv_reg = nn.Dense(in_units=2048, units=4*classes)
            self.globalavgpool = nn.GlobalAvgPool2D()
        self.conv_cls.initialize()
        self.conv_reg.initialize()
        # TODO lock BN

    def forward(self, *inputs):
        raise NotImplementedError

    def base_forward(self, x):
        """forwarding base network"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        c3 = self.layer3(x)
        return c3

    def top_forward(self, x):
        """forwarding roi feature"""
        c4 = self.layer4(x)
        c4 = self.globalavgpool(c4)
        f_cls = self.conv_cls(c4)
        f_reg = self.conv_reg(c4)
        return f_cls, f_reg

    @staticmethod
    def rcnn_prediction(rois, scaling_factor, img_height, img_width, class_scores,
                        bbox_deltas, bbox_reg_weights=(10.0, 10.0, 5.0, 5.0)):
        boxes = rois / scaling_factor
        pred_boxes = rpn.bbox_transform(boxes, bbox_deltas, bbox_reg_weights)
        pred_boxes = rpn.bbox_clip(pred_boxes, img_height, img_width)
        scores, boxes, box_ids, cls_boxes = rcnn_nms(class_scores, pred_boxes)
        return scores, boxes, box_ids, cls_boxes


def rcnn_nms(rcnn_cls, bbox_pred, overlap_thresh=0.5, score_thresh=0.05, topK=100):
    """RCNN NMS"""
    nclass = rcnn_cls.shape[1]
    np_scores = rcnn_cls.asnumpy()
    cls_boxes = [[] for _ in range(nclass)]
    # pre-thresh using numpy for now
    for j in range(1, nclass):
        inds = np.where(np_scores[:, j] > score_thresh)[0]
        if len(inds) == 0:
            continue
        scores_j = rcnn_cls[inds, j]
        boxes_j = bbox_pred[inds, j * 4:(j + 1) * 4]
        data = F.concat(scores_j.expand_dims(1), boxes_j, dim=1)
        nms_pred = F.contrib.box_nms(data, overlap_thresh, -1, coord_start=1,
                                     score_index=0, id_index=-1, force_suppress=True)
        effect = int(F.sum(nms_pred[:, 0] >= 0).asscalar())
        cls_boxes[j] = nms_pred[:effect, :]

    if topK > 0:
        image_scores = F.concat(*[cls_boxes[j][:, 0] for j in range(1, nclass)
                                  if len(cls_boxes[j]) != 0], dim=0)
        if image_scores.size > topK:
            image_thresh = np.sort(image_scores.asnumpy())[-topK]
            for j in range(1, nclass):
                if len(cls_boxes[j]) == 0:
                    continue
                keep = np.where(cls_boxes[j][:, 0].asnumpy() >= image_thresh)[0]
                cls_boxes[j] = cls_boxes[j][keep, :]

    im_results = F.concat(*[cls_boxes[j] for j in range(1, nclass)
                            if len(cls_boxes[j]) != 0], dim=0)
    box_ids = F.concat(*[F.ones_like(cls_boxes[j][:, 0]) * (j-1) for j in range(1, nclass)
                         if len(cls_boxes[j]) != 0], dim=0)
    boxes = im_results[:, 1:]
    scores = im_results[:, 0]
    print('scores.shape', scores.shape)
    print('box_ids.shape', box_ids.shape)
    return scores, boxes, box_ids, cls_boxes
    """
    scores = rcnn_cls#.asnumpy()
    boxes = bbox_pred#.asnumpy()
    cls_boxes = [[] for _ in range(nclass)]
    # skip 0 for background
    from gluoncv.utils.nms import nms
    for j in range(1, nclass):
        inds = np.where(scores[:, j] > score_thresh)[0]
        scores_j = scores[inds, j]
        boxes_j = boxes[inds, j * 4:(j + 1) * 4]
        dets_j = np.hstack((boxes_j, scores_j[:, np.newaxis])).astype(
            np.float32, copy=False
        )
        keep = nms(dets_j, overlap_thresh)
        cls_boxes[j] = dets_j[keep, :]

    # Limit to max_per_image detections **over all classes**
    if topK > 0:
        image_scores = np.hstack(
            [cls_boxes[j][:, -1] for j in range(1, nclass)]
        )
        if len(image_scores) > topK:
            image_thresh = np.sort(image_scores)[-topK]
            for j in range(1, nclass):
                keep = np.where(cls_boxes[j][:, -1] >= image_thresh)[0]
                cls_boxes[j] = cls_boxes[j][keep, :]

    im_results = np.vstack([cls_boxes[j] for j in range(1, nclass)])
    boxes = im_results[:, :-1]
    scores = im_results[:, -1]
    # Predict ids
    return scores, boxes, cls_boxes
    """
