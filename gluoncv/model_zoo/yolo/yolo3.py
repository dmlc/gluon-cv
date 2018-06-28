"""You Only Look Once Object Detection v3"""
from __future__ import absolute_import

import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet import autograd
from mxnet.gluon import nn
from .darknet import _conv2d

__all__ = []

# TODO(zhreshold) upsampling mode


class YOLOOutputV3(gluon.HybridBlock):
    def __init__(self, index, classes, anchors, stride, alloc_size=(128, 128), **kwargs):
        super(YOLOOutputV3, self).__init__(**kwargs)
        self._classes = classes
        self._num_pred = 1 + 4 + classes  # 1 objness + 4 box + classes
        self._num_anchors = len(anchors)
        self._stride = stride
        with self.name_scope():
            all_pred = self._num_pred * self._num_anchors
            self.prediction = nn.Conv2D(all_pred, kernel_size=1, padding=0, strides=1)
            # anchors will be multiplied to predictions
            anchors = np.array(anchors).reshape(1, 1, -1, 2) / stride
            self.anchors = self.params.get_constant('anchor_%d'%(index), anchors)
            # offsets will be added to predictions
            grid_x = np.arange(alloc_size[1])
            grid_y = np.arange(alloc_size[0])
            grid_x, grid_y = np.meshgrid(grid_x, grid_y)
            # stack to (n, n, 2)
            offsets = np.concatenate((grid_x[:, :, np.newaxis], grid_y[:, :, np.newaxis]), axis=-1)
            # expand dims to (1, 1, n, n, 2) so it's easier for broadcasting
            offsets = np.expand_dims(np.expand_dims(offsets, 0))
            self.offsets = self.params.get_constant('offset_%d'%(index), offsets)


    def hybrid_forward(self, F, x, anchors, offsets):
        # prediction flat to (batch, pred per pixel, height * width)
        pred = self.prediction(x).reshape((0, self._num_anchors * self._num_pred, -1))
        # transpose to (batch, height * width, num_anchor, num_pred)
        pred = pred.transpose(axes=(0, 2, 1)).reshape((0, -1, self._num_anchors, self._num_pred))
        # components
        box_centers = pred.slice_axis(axis=-1, begin=0, end=2)
        box_scales = pred.slice_axis(axis=-1, begin=2, end=4)
        objness = pred.slice_axis(axis=-1, begin=4, end=5)
        class_pred = pred.slice_axis(axis=-1, begin=5, end=None)

        if autograd.is_training():
            return box_centers, box_scales, objness, class_pred

        # valid offsets, (1, 1, height, width, 2)
        offsets = F.slice_like(offsets, x * 0, axes=(2, 3))
        # reshape to (1, height*width, 1, 2)
        offsets = offsets.reshape((1, -1, 1, 2))

        box_centers = F.broadcast_add(F.sigmoid(box_centers), offsets) * self._stride
        box_scales = F.broadcast_mul(F.exp(box_scales), anchors) * self._stride
        confidence = F.sigmoid(objness)
        class_score = F.broadcast_mul(F.sigmoid(class_pred), confidence)
        wh = box_scales / 2
        bbox = F.concat(box_centers - wh, centers + wh, axis=-1)

        # apply nms per class
        bboxes = F.tile(bbox, reps=(self._classes, 1, 1, 1, 1))
        scores = F.transpose(class_score, axes=(3, 0, 1, 2)).expand_dims(axis=-1)
        ids = F.broadcast_add(bboxes * 0, F.arange(0, self._classes).reshape((0, 1, 1, 1, 1)))
        detections = F.concat(ids, scores, bboxes, axis=-1)
        result = F.contrib.box_nms(detections, overlap_thresh=self.nms_thresh, topk=self.nms_topk,
            id_index=0, score_index=1, coord_start=2, force_suppress=False)






class YOLODetectionBlockV3(gluon.HybridBlock):
    """
    Add a few conv layers, return the output, and have a branch that do yolo detection.
    """
    def __init__(self, channel, **kwargs):
        super(YOLODetectionV3, self).__init__(**kwargs)
        assert channel % 2 == 0, "channel {} cannot be divided by 2".format(channel)
        self.body = nn.HybridSequential(prefix='')
        for _ in range(2):
            # 1x1 reduce
            self.body.add(_conv2d(channel, 1, 0, 1))
            # 3x3 expand
            self.body.add(_conv2d(channel * 2, 3, 1, 1))
        self.body.add(_conv2d(channel, 1, 0, 1))
        self.tip = _conv2d(channel * 2, 3, 1, 1)

    def hybrid_forward(self, F, x):
        route = self.body(x)
        tip = self.tip(route)
        return route, tip
