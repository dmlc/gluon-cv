"""Target generators for YOLOs."""
from __future__ import absolute_import
from __future__ import division

import numpy as np
from mxnet import gluon
from mxnet import nd
from mxnet import autograd
from ...nn.bbox import BBoxCornerToCenter, BBoxCenterToCorner
from ...nn.coder import GridBoxCenterEncoder


class YOLOPrefetchTargetGeneratorV3(gluon.Block):
    def __init__(self, num_class, **kwargs):
        super(YOLOPrefetchTargetGeneratorV3, self).__init__(**kwargs)
        self._num_class = num_class
        self.bbox2center = BBoxCornerToCenter(axis=-1, split=True)
        self.bbox2corner = BBoxCenterToCorner(axis=-1, split=False)


    def forward(self, img, xs, anchors, offsets, gt_boxes, gt_ids):
        """Generating training targets that do not require network predictions.


        """
        assert isinstance(anchors, (list, tuple))
        all_anchors = nd.concat(*[a.reshape(-1, 2) for a in anchors], dim=0)
        assert isinstance(offsets, (list, tuple))
        all_offsets = nd.concat(*[o.reshape(-1, 2) for o in offsets], dim=0)
        num_anchors = np.cumsum([a.size // 2 for a in anchors])
        num_offsets = np.cumsum([o.size // 2 for o in offsets])
        assert isinstance(xs, (list, tuple))
        assert len(xs) == len(anchors) == len(offsets)

        # orig image size
        orig_height = img.shape[2]
        orig_width = img.shape[3]
        with autograd.pause():
            # outputs
            shape_like = all_anchors.reshape((1, -1, 2)) * all_offsets.reshape(
                (-1, 1, 2)).expand_dims(0).repeat(repeats=gt_ids.shape[0], axis=0)
            center_targets = nd.zeros_like(shape_like)
            scale_targets = nd.zeros_like(center_targets)
            weights = nd.zeros_like(center_targets.split(axis=-1, num_outputs=2)[0])
            objectness = nd.zeros_like(weights)
            class_targets = nd.one_hot(weights.squeeze(axis=-1), depth=self._num_class)
            class_targets[:] = -1  # prefill -1 for ignores

            # for each ground-truth, find the best matching anchor within the particular grid
            # for instance, center of object 1 reside in grid (3, 4) in (16, 16) feature map
            # then only the anchor in (3, 4) is going to be matched
            gtx, gty, gtw, gth = self.bbox2center(gt_boxes)
            shift_gt_boxes = nd.concat(gtx * 0, gty * 0, gtw, gth, dim=-1)
            anchor_boxes = nd.concat(0 * all_anchors, all_anchors, dim=-1)  # zero center anchor boxes
            shift_anchor_boxes = self.bbox2corner(anchor_boxes)
            ious = nd.contrib.box_iou(shift_anchor_boxes, shift_gt_boxes).transpose((1, 0, 2))
            # real value is required to process, convert to Numpy
            matches = ious.argmax(axis=1).asnumpy()  # (B, M)
            valid_gts = (gt_boxes >= 0).prod(axis=-1).asnumpy()  # (B, M)
            np_gtx, np_gty, np_gtw, np_gth = [x.asnumpy() for x in [gtx, gty, gtw, gth]]
            np_anchors = all_anchors.asnumpy()
            np_gt_ids = gt_ids.asnumpy()
            # TODO(zhreshold): the number of valid gt is not a big number, therefore for loop
            # should not be a problem right now. Switch to better solution is needed.
            for b in range(matches.shape[0]):
                for m in range(matches.shape[1]):
                    if valid_gts[b, m] < 1:
                        break
                    match = int(matches[b, m])
                    nlayer = np.nonzero(num_anchors > match)[0][0]
                    height = xs[nlayer].shape[2]
                    width = xs[nlayer].shape[3]
                    gtx, gty, gtw, gth = np_gtx[b, m, 0], np_gty[b, m, 0], np_gtw[b, m, 0], np_gth[b, m, 0]
                    # compute the location of the gt centers
                    loc_x = int(gtx / orig_width * width)
                    loc_y = int(gty / orig_height * height)
                    # write back to targets
                    index = np.sum(num_offsets[:nlayer]) + loc_y * width + loc_x
                    center_targets[b, index, match, 0] = gtx / orig_width * width - loc_x  # tx
                    center_targets[b, index, match, 1] = gty / orig_height * height - loc_y  # ty
                    scale_targets[b, index, match, 0] = np.log(gtw / np_anchors[match, 0])
                    scale_targets[b, index, match, 1] = np.log(gth / np_anchors[match, 1])
                    weights[b, index, match, 0] = 2.0 - gtw * gth / orig_width / orig_height
                    objectness[b, index, match, 0] = 1
                    class_targets[b, index, match, int(np_gt_ids[b, m, 0])] = 1
                    # print(loc_x, loc_y, nlayer, index, num_offsets, 'gt', gtx, gty, gtw, gth,
                    # 'targets', center_targets[b, index, match, 0].asscalar(),
                    # center_targets[b, index, match, 1].asscalar(), scale_targets[b, index, match, 0].asscalar(),
                    # scale_targets[b, index, match, 1].asscalar(), weights[b, index, match, 0].asscalar(), 'id', int(np_gt_ids[b, m, 0]))
        return center_targets, scale_targets, weights, objectness, class_targets


class YOLODynamicTargetGeneratorV3(gluon.Block):
    def __init__(self, num_class, pos_iou_thresh, ignore_iou_thresh, **kwargs):
        super(YOLOTargetGeneratorV3, self).__init__(**kwargs)
        self._num_class = num_class
        self._pos_iou_thresh = pos_iou_thresh
        self._ignore_iou_thresh = ignore_iou_thresh
        self.bbox_c2c = BBoxCornerToCenter(axis=-1, split=True)

    def forward(self, img, xs, anchors, offsets, box_preds, gt_boxes, gt_ids):
        """Generate dynamic training targets based on current predictions.
        anchors : list of anchors in each yolo output layer
        gt_boxes
        gt_ids
        """
        assert isinstance(anchors, (list, tuple))
        assert isinstance(box_preds, (list, tuple))
        assert isinstance(xs, (list, tuple))
        assert len(anchors) == len(box_preds) == len(xs)

        # orig image size
        orig_height = img.shape[2]
        orig_width = img.shape[3]


        with autograd.pause():
            # for each yolo stage
            all_obj_targets = []
            all_center_targets = []
            all_scale_targets = []
            all_weights = []
            all_cls_targets = []
            for anchor, box_pred, offset, x in zip(anchors, box_preds, offsets, xs):
                # feature map size
                height = x.shape[2]
                width = x.shape[3]
                # process each batch separately
                obj_targets = []
                center_targets = []
                scale_targets = []
                weights = []
                cls_targets = []
                for b in range(box_pred.shape[0]):
                    # for each prediction box, find besting matching
                    ious = nd.contrib.box_iou(box_preds[b], gt_boxes[b])
                    best_index = ious.argmax(axis=-1)
                    best_ious = ious.pick(best_index, axis=-1)
                    # fill with ignore -1
                    obj_target = nd.ones_like(best_ious) * -1
                    # positive: 1
                    obj_target = nd.where(best_ious > self._pos_iou_thresh, nd.ones_like(best_ious), obj_target)
                    # negative: 0
                    obj_target = nd.where(best_ious <= self._ignore_iou_thresh, best_ious * 0, obj_target)
                    # compute box targets
                    gts = gt_boxes[b].take(best_index)  # (N, 4)
                    gtx, gty, gtw, gth = self.bbox_c2c(gts)
                    x_offset, y_offset = offset.reshape(-1, 2).split(aixs=-1, num_outputs=2)
                    x_anchor, y_anchor = anchor.reshape(-1, 2).split(axis=-1, num_outputs=2)
                    tx = gtx / orig_width * width - x_offset
                    ty = gty / orig_height * height - y_offset
                    tw = nd.log(gtw / orig_width * width / x_anchor)
                    th = nd.log(gth / orig_height * height / y_anchor)
                    center_target = nd.stack(tx, ty, axis=-1)
                    scale_target = nd.stack(tw, th, axis=-1)
                    weight = 2 - gtw * gth / orig_width / orig_height
                    cls_target = nd.one_hot(gt_ids.take(best_index).reshape(-1), self._num_class)

                    obj_targets.append(obj_target)
                    center_targets.append(center_target)
                    scale_targets.append(scale_target)
                    weights.append(weight)
                    cls_targets.append(cls_target)
                all_obj_targets.append(nd.stack(*obj_targets))
                all_center_targets.append(nd.stack(*center_targets))
                all_scale_targets.append(nd.stack(*scale_targets))
                all_weights.append(nd.stack(*weights))
                all_cls_targets.append(nd.stack(*cls_targets))
            all_obj_targets = nd.concat(*all_obj_targets, dim=0)
            all_center_targets = nd.concat(*all_center_targets, dim=0)
            all_scale_targets = nd.concat(*all_scale_targets, dim=0)
            all_weights = nd.conat(*all_weights, dim=0)
            all_cls_targets = nd.concat(*all_cls_targets, dim=0)

        return all_obj_targets, all_center_targets, all_scale_targets, all_weights, all_cls_targets
