"""You Only Look Once Object Detection v3"""
from __future__ import absolute_import
from __future__ import division

import os
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet import autograd
from mxnet.gluon import nn
from .darknet import _conv2d, darknet53

__all__ = ['YOLOV3', 'yolo3_416_darknet53_voc', 'yolo3_416_darknet53_coco']

# TODO(zhreshold) upsampling mode
def _upsample(x, stride=2):
    return x.repeat(axis=-1, repeats=stride).repeat(axis=-2, repeats=stride)


class YOLOOutputV3(gluon.HybridBlock):
    def __init__(self, index, num_classes, anchors, stride, alloc_size=(128, 128), **kwargs):
        super(YOLOOutputV3, self).__init__(**kwargs)
        anchors = np.array(anchors).astype('float32')
        self._classes = num_classes
        self._num_pred = 1 + 4 + num_classes  # 1 objness + 4 box + num_classes
        self._num_anchors = anchors.size // 2
        self._stride = stride
        with self.name_scope():
            all_pred = self._num_pred * self._num_anchors
            self.prediction = nn.Conv2D(all_pred, kernel_size=1, padding=0, strides=1)
            # anchors will be multiplied to predictions
            anchors = anchors.reshape(1, 1, -1, 2)
            self.anchors = self.params.get_constant('anchor_%d'%(index), anchors)
            # offsets will be added to predictions
            grid_x = np.arange(alloc_size[1])
            grid_y = np.arange(alloc_size[0])
            grid_x, grid_y = np.meshgrid(grid_x, grid_y)
            # stack to (n, n, 2)
            offsets = np.concatenate((grid_x[:, :, np.newaxis], grid_y[:, :, np.newaxis]), axis=-1)
            # expand dims to (1, 1, n, n, 2) so it's easier for broadcasting
            offsets = np.expand_dims(np.expand_dims(offsets, axis=0), axis=0)
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
            return box_centers, box_scales, objness, class_pred, anchors, offsets

        # valid offsets, (1, 1, height, width, 2)
        offsets = F.slice_like(offsets, x * 0, axes=(2, 3))
        # reshape to (1, height*width, 1, 2)
        offsets = offsets.reshape((1, -1, 1, 2))

        box_centers = F.broadcast_add(F.sigmoid(box_centers), offsets) * self._stride
        box_scales = F.broadcast_mul(F.exp(box_scales), anchors)
        confidence = F.sigmoid(objness)
        class_score = F.broadcast_mul(F.sigmoid(class_pred), confidence)
        wh = box_scales / 2.0
        bbox = F.concat(box_centers - wh, box_centers + wh, dim=-1)

        # prediction per class
        bboxes = F.tile(bbox, reps=(self._classes, 1, 1, 1, 1))
        scores = F.transpose(class_score, axes=(3, 0, 1, 2)).expand_dims(axis=-1)
        ids = F.broadcast_add(scores * 0, F.arange(0, self._classes).reshape((0, 1, 1, 1, 1)))
        detections = F.concat(ids, scores, bboxes, dim=-1)
        # reshape to (B, xx, 6)
        detections = F.reshape(detections.transpose(axes=(1, 0, 2, 3, 4)), (0, -1, 6))
        return detections


class YOLODetectionBlockV3(gluon.HybridBlock):
    """
    Add a few conv layers, return the output, and have a branch that do yolo detection.
    """
    def __init__(self, channel, **kwargs):
        super(YOLODetectionBlockV3, self).__init__(**kwargs)
        assert channel % 2 == 0, "channel {} cannot be divided by 2".format(channel)
        with self.name_scope():
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


class YOLOV3(gluon.HybridBlock):
    def __init__(self, stages, channels, anchors, strides, classes, alloc_size=(128, 128),
                 nms_thresh=0.45, nms_topk=400, post_nms=100, **kwargs):
        super(YOLOV3, self).__init__(**kwargs)
        self.classes = classes
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms
        with self.name_scope():
            self.stages = nn.HybridSequential()
            self.transitions = nn.HybridSequential()
            self.yolo_blocks = nn.HybridSequential()
            self.yolo_outputs = nn.HybridSequential()
            # note that anchors and strides should be used in reverse order
            for i, stage, channel, anchor, stride in zip(range(len(stages)), stages, channels, anchors[::-1], strides[::-1]):
                self.stages.add(stage)
                block = YOLODetectionBlockV3(channel)
                self.yolo_blocks.add(block)
                output = YOLOOutputV3(i, len(classes), anchor, stride, alloc_size=alloc_size)
                self.yolo_outputs.add(output)
                if i > 0:
                    self.transitions.add(_conv2d(channel, 1, 0, 1))

    def set_nms(self, nms_thresh=0.45, nms_topk=400, post_nms=100):
        """Set non-maximum suppression parameters.

        Parameters
        ----------
        nms_thresh : float, default is 0.45.
            Non-maximum suppression threshold. You can speficy < 0 or > 1 to disable NMS.
        nms_topk : int, default is 400
            Apply NMS to top k detection results, use -1 to disable so that every Detection
             result is used in NMS.
        post_nms : int, default is 100
            Only return top `post_nms` detection results, the rest is discarded. The number is
            based on COCO dataset which has maximum 100 objects per image. You can adjust this
            number if expecting more objects. You can use -1 to return all detections.

        Returns
        -------
        None

        """
        self._clear_cached_op()
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms

    def hybrid_forward(self, F, x):
        all_box_centers = []
        all_box_scales = []
        all_objectness = []
        all_class_pred = []
        all_anchors = []
        all_offsets = []
        all_detections = []
        routes = []
        for stage, block, output in zip(self.stages, self.yolo_blocks, self.yolo_outputs):
            x = stage(x)
            routes.append(x)

        for i, block, output in zip(range(len(routes)), self.yolo_blocks, self.yolo_outputs):
            x, tip = block(x)
            if autograd.is_training():
                box_centers, box_scales, objness, class_pred, anchors, offsets = output(tip)
                all_box_centers.append(box_centers)
                all_box_scales.append(box_scales)
                all_objectness.append(objness)
                all_class_pred.append(class_pred)
                all_anchors.append(anchors)
                all_offsets.append(offsets)
            else:
                detections = output(tip)
                all_detections.append(detections)
            if i >= len(routes) - 1:
                break
            x = self.transitions[i](x)
            upsample = _upsample(x, stride=2)
            x = F.concat(upsample, routes[::-1][i + 1], dim=1)

        if autograd.is_training():
            # return raw predictions
            return (
                F.concat(*all_box_centers, dim=-2),
                F.concat(*all_box_scales, dim=-2),
                F.concat(*all_objectness, dim=-2),
                F.concat(*all_class_pred, dim=-2))

        result = F.concat(*all_detections, dim=1)
        # apply nms per class
        if self.nms_thresh > 0 and self.nms_thresh < 1:
            result = F.contrib.box_nms(result, overlap_thresh=self.nms_thresh,
                topk=self.nms_topk, id_index=0, score_index=1, coord_start=2, force_suppress=False)
            if self.post_nms > 0:
                result = result.slice_axis(axis=1, begin=0, end=self.post_nms)
        ids = result.slice_axis(axis=-1, begin=0, end=1)
        scores = result.slice_axis(axis=-1, begin=1, end=2)
        bboxes = result.slice_axis(axis=-1, begin=2, end=None)
        return ids, scores, bboxes

def get_yolov3(name, base_size, stages, filters, anchors, strides, classes,
               dataset, pretrained=False, ctx=mx.cpu(),
               root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    """Get YOLOV3 models.

    Parameters
    ----------
    name : str or None
        Model name, if `None` is used, you must specify `features` to be a `HybridBlock`.
    base_size : int
        Base image size for training, this is fixed once training is assigned.
        A fixed base size still allows you to have variable input size during test.
    features : iterable of str or `HybridBlock`
        List of network internal output names, in order to specify which layers are
        used for predicting bbox values.
        If `name` is `None`, `features` must be a `HybridBlock` which generate mutliple
        outputs for prediction.
    filters : iterable of float or None
        List of convolution layer channels which is going to be appended to the base
        network feature extractor. If `name` is `None`, this is ignored.
    sizes : iterable fo float
        Sizes of anchor boxes, this should be a list of floats, in incremental order.
        The length of `sizes` must be len(layers) + 1. For example, a two stage SSD
        model can have ``sizes = [30, 60, 90]``, and it converts to `[30, 60]` and
        `[60, 90]` for the two stages, respectively. For more details, please refer
        to original paper.
    ratios : iterable of list
        Aspect ratios of anchors in each output layer. Its length must be equals
        to the number of SSD output layers.
    steps : list of int
        Step size of anchor boxes in each output layer.
    classes : iterable of str
        Names of categories.
    dataset : str
        Name of dataset. This is used to identify model name because models trained on
        differnet datasets are going to be very different.
    pretrained : bool, optional, default is False
        Load pretrained weights.
    pretrained_base : bool, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `Ture`, this has no effect.
    ctx : mxnet.Context
        Context such as mx.cpu(), mx.gpu(0).
    root : str
        Model weights storing path.

    Returns
    -------
    HybridBlock
        A YOLOV3 detection network.
    """
    net = YOLOV3(stages, filters, anchors, strides, classes=classes, **kwargs)
    if pretrained:
        from ..model_store import get_model_file
        full_name = '_'.join(('yolo3', str(base_size), name, dataset))
        net.load_params(get_model_file(full_name, root=root), ctx=ctx)
    return net

def yolo3_416_darknet53_voc(pretrained_base=True, pretrained=False, **kwargs):
    from ...data import VOCDetection
    pretrained_base = False if pretrained else pretrained_base
    base_net = darknet53(pretrained=pretrained_base)
    stages = [base_net.features[:15], base_net.features[15:24], base_net.features[24:]]
    anchors = [[10,13,  16,30,  33,23],  [30,61,  62,45,  59,119],  [116,90,  156,198,  373,326]]
    strides = [8, 16, 32]
    classes = VOCDetection.CLASSES
    return get_yolov3('darknet53', 416, stages, [512, 256, 128], anchors, strides, classes, 'voc', pretrained=pretrained, **kwargs)

def yolo3_416_darknet53_coco(pretrained_base=True, pretrained=False, **kwargs):
    from ...data import COCODetection
    pretrained_base = False if pretrained else pretrained_base
    base_net = darknet53(pretrained=pretrained_base)
    stages = [base_net.features[:15], base_net.features[15:24], base_net.features[24:]]
    anchors = [[10,13,  16,30,  33,23],  [30,61,  62,45,  59,119],  [116,90,  156,198,  373,326]]
    strides = [8, 16, 32]
    classes = COCODetection.CLASSES
    return get_yolov3('darknet53', 416, stages, [512, 256, 128], anchors, strides, classes, 'coco', pretrained=pretrained, **kwargs)
