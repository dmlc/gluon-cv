"""Faster RCNN Model."""
from __future__ import absolute_import

import os
import mxnet as mx
from mxnet import autograd
from mxnet.gluon import nn
from .rcnn_target import RCNNTargetSampler, RCNNTargetGenerator
from ..rcnn import RCNN
from ..rpn import RPN

__all__ = ['FasterRCNN', 'get_faster_rcnn',
           'faster_rcnn_resnet50_v1b_voc',
           'faster_rcnn_resnet50_v1b_coco']


class FasterRCNN(RCNN):
    def __init__(self, features, top_features, scales, ratios, classes, roi_mode, roi_size,
                 stride=16, rpn_channel=1024, nms_thresh=0.3, nms_topk=400,
                 num_sample=128, pos_iou_thresh=0.5, neg_iou_thresh_high=0.5,
                 neg_iou_thresh_low=0.0, pos_ratio=0.25, max_batch=1, max_roi=100000, **kwargs):
        super(FasterRCNN, self).__init__(
            features, top_features, classes, roi_mode, roi_size, **kwargs)
        self.stride = stride
        self._max_batch = max_batch
        self._max_roi = max_roi
        self._target_generator = set([RCNNTargetGenerator(self.num_class)])
        with self.name_scope():
            self.rpn = RPN(rpn_channel, stride, scales=scales, ratios=ratios)
            self.sampler = RCNNTargetSampler(num_sample, pos_iou_thresh, neg_iou_thresh_high,
                                             neg_iou_thresh_low, pos_ratio)

    @property
    def target_generator(self):
        return list(self._target_generator)[0]

    def hybrid_forward(self, F, x, gt_box=None):
        feat = self.features(x)
        # RPN proposals
        if autograd.is_training():
            rpn_score, rpn_box, raw_rpn_score, raw_rpn_box, anchors = self.rpn(
                feat, F.zeros_like(x))
            # sample 128 roi
            assert gt_box is not None
            rpn_box, samples, matches = self.sampler(rpn_box, gt_box)
        else:
            rpn_score, rpn_box = self.rpn(feat, F.zeros_like(x))

        # create batchid for roi
        with autograd.pause():
            roi_batchid = F.arange(
                0, self._max_batch, repeat=self._max_roi).reshape(
                    (-1, self._max_roi))
            roi_batchid = F.slice_like(roi_batchid, rpn_box * 0, axes=(0, 1))
            rpn_roi = F.concat(*[roi_batchid.reshape((-1, 1)), rpn_box.reshape((-1, 4))], dim=-1)

        # ROI features
        if self._roi_mode == 'pool':
            pooled_feat = F.ROIPooling(feat, rpn_roi, self._roi_size, 1. / self.stride)
        elif self._roi_mode == 'align':
            pooled_feat = F.contrib.ROIAlign(feat, rpn_roi, self._roi_size, 1. / self.stride)
        else:
            raise ValueError("Invalid roi mode: {}".format(self._roi_mode))
        # RCNN prediction
        top_feat = self.top_features(pooled_feat)
        # top_feat = F.Pooling(top_feat, global_pool=True, pool_type='avg', kernel=self._roi_size)
        top_feat = self.global_avg_pool(top_feat)
        cls_pred = self.class_predictor(top_feat)
        box_pred = self.box_predictor(top_feat).reshape((-1, self.num_class, 4)).transpose((1, 0, 2))

        # no need to convert bounding boxes in training, just return
        if autograd.is_training():
            box_pred = box_pred.transpose((1, 0, 2))
            return (cls_pred, box_pred, rpn_box, samples, matches,
                    raw_rpn_score, raw_rpn_box, anchors)

        # translate bboxes
        bboxes = self.box_decoder(box_pred, self.box_to_center(rpn_box)).split(
            axis=0, num_outputs=self.num_class, squeeze_axis=True)
        cls_ids, scores = self.cls_decoder(F.softmax(cls_pred, axis=-1))
        results = []
        for i in range(self.num_class):
            cls_id = cls_ids.slice_axis(axis=-1, begin=i, end=i+1)
            score = scores.slice_axis(axis=-1, begin=i, end=i+1)
            # per class results
            per_result = F.concat(*[cls_id, score, bboxes[i]], dim=-1)

            results.append(per_result)
        result = F.concat(*results, dim=0).expand_dims(0)
        if self.nms_thresh > 0 and self.nms_thresh < 1:
            result = F.contrib.box_nms(
                result, overlap_thresh=self.nms_thresh, topk=self.nms_topk,
                id_index=0, score_index=1, coord_start=2)
            if self.nms_topk > 0:
                result = result.slice_axis(axis=1, begin=0, end=100).squeeze(axis=0)
        ids = F.slice_axis(result, axis=-1, begin=0, end=1)
        scores = F.slice_axis(result, axis=-1, begin=1, end=2)
        bboxes = F.slice_axis(result, axis=-1, begin=2, end=6)
        return ids, scores, bboxes

def get_faster_rcnn(name, features, top_features, scales, ratios, classes,
                    roi_mode, roi_size, dataset, stride=16,
                    rpn_channel=1024, pretrained=False, pretrained_base=True, ctx=mx.cpu(),
                    root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    net = FasterRCNN(features, top_features, scales, ratios, classes, roi_mode, roi_size,
                     stride=stride, rpn_channel=rpn_channel, **kwargs)
    if pretrained:
        from ..model_store import get_model_file
        full_name = '_'.join(('faster_rcnn', name, dataset))
        net.load_params(get_model_file(full_name, root=root), ctx=ctx)
    return net

def faster_rcnn_resnet50_v1b_voc(pretrained_base=True, **kwargs):
    r"""Faster RCNN model from the paper
    "Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster r-cnn: Towards
    real-time object detection with region proposal networks"

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.

    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    Examples
    --------
    >>> model = get_faster_rcnn_resnet50_voc(pretrained=True)
    >>> print(model)
    """
    from ..resnetv1b import resnet50_v1b
    from ...data import VOCDetection
    classes = VOCDetection.CLASSES
    base_network = resnet50_v1b(pretrained=pretrained_base, dilated=False)
    features = nn.HybridSequential()
    top_features = nn.HybridSequential()
    for layer in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']:
        features.add(getattr(base_network, layer))
    for layer in ['layer4']:
        top_features.add(getattr(base_network, layer))
    train_patterns = '|'.join(['.*dense', '.*rpn', '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv'])
    return get_faster_rcnn('resnet50_v1b', features, top_features, scales=(2, 4, 8, 16, 32),
                           ratios=(0.5, 1, 2), classes=classes, dataset='voc',
                           roi_mode='align', roi_size=(14, 14), stride=16,
                           rpn_channel=1024, train_patterns=train_patterns, **kwargs)

def faster_rcnn_resnet50_v1b_coco(pretrained_base=True, **kwargs):
    from ..resnetv1b import resnet50_v1b
    from ...data import COCODetection
    classes = COCODetection.CLASSES
    base_network = resnet50_v1b(pretrained=pretrained_base, dilated=False)
    features = nn.HybridSequential()
    top_features = nn.HybridSequential()
    for layer in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']:
        features.add(getattr(base_network, layer))
    for layer in ['layer4']:
        top_features.add(getattr(base_network, layer))
    train_patterns = '|'.join(['.*dense', '.*rpn', '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv'])
    return get_faster_rcnn('resnet50_v1b', features, top_features, scales=(2, 4, 8, 16, 32),
                           ratios=(0.5, 1, 2), classes=classes, dataset='coco',
                           roi_mode='align', roi_size=(14, 14), stride=16,
                           rpn_channel=1024, train_patterns=train_patterns, **kwargs)
