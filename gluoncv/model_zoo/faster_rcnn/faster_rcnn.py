"""Faster RCNN Model."""
from __future__ import absolute_import

import os
import mxnet as mx
from mxnet import autograd
from mxnet.gluon import nn
from ..rcnn import RCNN
from ..rpn import RPN

__all__ = ['FasterRCNN', 'get_faster_rcnn',
           'get_faster_rcnn_resnet50_v1b_voc',
           'get_faster_rcnn_resnet50_v1b_coco']


class FasterRCNN(RCNN):
    def __init__(self, features, top_features, scales, ratios, classes, roi_mode, roi_size,
                 stride=16, rpn_channel=1024, nms_thresh=0.3, nms_topk=400, **kwargs):
        super(FasterRCNN, self).__init__(
            features, top_features, classes, roi_mode, roi_size, **kwargs)
        self._stride = stride
        with self.name_scope():
            self.rpn = RPN(rpn_channel, stride, scales=scales, ratios=ratios)

    def hybrid_forward(self, F, x, width, height):
        feat = self.features(x)
        # RPN proposals
        rpn_score, rpn_box, rpn_roi, roi = self.rpn(feat, width, height)
        # ROI features
        if self._roi_mode == 'pool':
            pooled_feat = F.ROIPooling(feat, rpn_roi, self._roi_size, 1. / self._stride)
        elif self._roi_mode == 'align':
            #TODO(zhreshold): use ROIAlign
            pooled_feat = F.ROIPooling(feat, rpn_roi, self._roi_size, 1. / self._stride)
        else:
            raise ValueError("Invalid roi mode: {}".format(self._roi_mode))
        # RCNN prediction
        top_feat = self.top_features(pooled_feat)
        # top_feat = F.Pooling(top_feat, global_pool=True, pool_type='avg', kernel=self._roi_size)
        top_feat = self.global_avg_pool(top_feat)
        cls_pred = self.class_predictor(top_feat)
        box_pred = self.box_predictor(top_feat).reshape(-1, self.num_class, 4).transpose((1, 0, 2))

        # no need to convert bounding boxes in training, just return
        if autograd.is_training():
            return cls_pred, box_pred, roi

        # translate bboxes
        bboxes = self.box_decoder(box_pred, roi).split(
            axis=0, num_outputs=self.num_class, squeeze_axis=True)
        cls_ids, scores = self.cls_decoder(F.softmax(cls_pred, axis=-1))
        results = []
        for i in range(self.num_class):
            cls_id = cls_ids.slice_axis(axis=-1, begin=i, end=i+1)
            score = scores.slice_axis(axis=-1, begin=i, end=i+1)
            # per class results
            per_result = F.concat(*[cls_id, score, bboxes[i]], dim=-1)
            if self.nms_thresh > 0 and self.nms_thresh < 1:
                per_result = F.contrib.box_nms(
                    per_result, overlap_thresh=self.nms_thresh, topk=self.nms_topk,
                    id_index=0, score_index=1, coord_start=2)
                if self.nms_topk > 0:
                    per_result = per_result.slice_axis(axis=0, begin=0, end=self.nms_topk)
            results.append(per_result)
        result = F.concat(*results, dim=0)
        ids = F.slice_axis(result, axis=-1, begin=0, end=1)
        scores = F.slice_axis(result, axis=-1, begin=1, end=2)
        bboxes = F.slice_axis(result, axis=-1, begin=2, end=6)
        return ids, scores, bboxes

def get_faster_rcnn(features, top_features, scales, ratios, classes,
                    roi_mode, roi_size, dataset, stride=16,
                    rpn_channel=1024, pretrained=False, pretrained_base=True, ctx=mx.cpu(),
                    root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    net = FasterRCNN(features, top_features, scales, ratios, classes, roi_mode, roi_size,
                     stride=stride, rpn_channel=rpn_channel)
    if pretrained:
        from ..model_store import get_model_file
        full_name = '_'.join(('faster_rcnn', name, dataset))
        net.load_params(get_model_file(full_name, root=root), ctx=ctx)
    return net

def get_faster_rcnn_resnet50_v1b_voc(pretrained=False, pretrained_base=True, **kwargs):
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
    return get_faster_rcnn(features, top_features, scales=(32, 64, 128, 256, 512),
                           ratios=(0.5, 1, 2), classes=classes, dataset='voc',
                           roi_mode='align', roi_size=(14, 14), stride=16,
                           rpn_channel=1024, **kwargs)

def get_faster_rcnn_resnet50_v1b_coco(pretrained=False, pretrained_base=True, **kwargs):
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
    return get_faster_rcnn(features, top_features, scales=(32, 64, 128, 256, 512),
                           ratios=(0.5, 1, 2), classes=classes, dataset='coco',
                           roi_mode='align', roi_size=(14, 14), stride=16,
                           rpn_channel=1024, **kwargs)
