"""Target generators for YOLOs."""
from __future__ import absolute_import
from __future__ import division

from mxnet import gluon
from mxnet import nd
from mxnet import autograd


class YOLOTargetGeneratorV3(gluon.Block):
    def __init__(self, num_class, pos_iou_thresh, ignore_iou_thresh, **kwargs):
        super(YOLOTargetGeneratorV3, self).__init__(**kwargs)
        self._num_class = num_class
        self._pos_iou_thresh = pos_iou_thresh
        self._ignore_iou_thresh = ignore_iou_thresh
        self.bbox_c2c = BBoxCornerToCenter(axis=-1, split=True)

    def forward(self, img, x, anchors, offsets, box_preds, gt_boxes, gt_ids):
        """
        anchors : list of anchors in each yolo output layer
        gt_boxes
        gt_ids
        """
        assert isinstance(anchors, (list, tuple))
        assert isinstance(box_preds, (list, tuple))
        assert len(anchors) == len(box_preds)

        # orig image size
        orig_height = img.shape[2]
        orig_width = img.shape[3]
        # feature map size
        height = x.shape[2]
        width = x.shape[3]

        with autograd.pause():
            # for each yolo stage
            all_obj_targets = []
            all_center_targets = []
            all_scale_targets = []
            all_cls_targets = []
            for anchor, box_pred, offset in zip(anchors, box_preds, offsets):
                # process each batch separately
                obj_targets = []
                center_targets = []
                scale_targets = []
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
                    cls_target = nd.one_hot(gt_ids.take(best_index).reshape(-1), self._num_class)

                    obj_targets.append(obj_target)
                    center_targets.append(center_target)
                    scale_targets.append(scale_target)
                    cls_targets.append(cls_target)
                all_obj_targets.append(nd.stack(*obj_targets))
                all_center_targets.append(nd.stack(*center_targets))
                all_scale_targets.append(nd.stack(*scale_targets))
                all_cls_targets.append(nd.stack(*cls_targets))
            all_obj_targets = nd.concat(*all_obj_targets, dim=0)
            all_center_targets = nd.concat(*all_center_targets, dim=0)
            all_scale_targets = nd.concat(*all_scale_targets, dim=0)
            all_cls_targets = nd.concat(*all_cls_targets, dim=0)

            # for each ground-truth, find matching anchors
            shift_gtx, shift_gty, shift_gtw, shift_gth = self.bbox_c2c(gt_boxes)
            # the grid that gt center located in
            loc_x = nd.floor(shift_gtx / orig_width * width)
            loc_y = nd.floor(shift_gty / orig_height * height)
            # make center 0
            half_w = shift_gtw / 2
            half_h = shift_gth / 2
            shift_gt_box = nd.stack(-half_w, -half_h, half_w, half_h, axis=-1)
            all_anchors = nd.conat(*[a.reshape(-1, 2) for a in anchors], dim=0)
            all_anchors_x, all_anchors_y = all_anchors.split(axis=-1, num_outputs=2)
            shift_boxes = nd.stack(half_w * 0, half_h * 0, all_anchors_x / width, all_anchors_y / height, axis=-1)
            shift_ious = nd.contrib.box_iou(shift_boxes, shift_gt_box).transpose(1, 0, 2)
            best_shift_index = shift_ious.argmax(axis=-1)

            # TODO(zhreshold): modify targets in specific grids
