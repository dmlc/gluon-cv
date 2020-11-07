"""RCNN framewark Training Metrics."""

import mxnet as mx
try:
    from mxnet.metric import EvalMetric
except ImportError:
    from mxnet.gluon.metric import EvalMetric


class RPNAccMetric(EvalMetric):
    """ RPN accuracy. """

    def __init__(self):
        super(RPNAccMetric, self).__init__('RPNAcc')

    def update(self, labels, preds):
        """ Updates the internal evaluation result. """
        # label: [rpn_label, rpn_weight]
        # preds: [rpn_cls_logits]
        rpn_label, rpn_weight = labels
        rpn_cls_logits = preds[0]

        # calculate num_inst (average on those fg anchors)
        num_inst = mx.nd.sum(rpn_weight)

        # cls_logits (b, c, h, w) red_label (b, 1, h, w)
        pred_label = mx.nd.sigmoid(rpn_cls_logits) >= 0.5
        # label (b, 1, h, w)
        num_acc = mx.nd.sum((pred_label == rpn_label) * rpn_weight)

        self.sum_metric += num_acc.asscalar()
        self.num_inst += num_inst.asscalar()


class RPNL1LossMetric(mx.metric.EvalMetric):
    """ RPN L1 loss. """

    def __init__(self):
        super(RPNL1LossMetric, self).__init__('RPNL1Loss')

    def update(self, labels, preds):
        """ Updates the internal evaluation result. """
        # label = [rpn_bbox_target, rpn_bbox_weight]
        # pred = [rpn_bbox_reg]
        rpn_bbox_target, rpn_bbox_weight = labels
        rpn_bbox_reg = preds[0]

        # calculate num_inst (average on those fg anchors)
        num_inst = mx.nd.sum(rpn_bbox_weight) / 4

        # calculate smooth_l1
        loss = mx.nd.sum(
            rpn_bbox_weight * mx.nd.smooth_l1(rpn_bbox_reg - rpn_bbox_target, scalar=3))

        self.sum_metric += loss.asscalar()
        self.num_inst += num_inst.asscalar()


class RCNNAccMetric(mx.metric.EvalMetric):
    """ RCNN accuracy. """

    def __init__(self):
        super(RCNNAccMetric, self).__init__('RCNNAcc')

    def update(self, labels, preds):
        """ Updates the internal evaluation result. """
        # label = [rcnn_label]
        # pred = [rcnn_cls]
        rcnn_label = labels[0]
        rcnn_cls = preds[0]

        # calculate num_acc
        pred_label = mx.nd.argmax(rcnn_cls, axis=-1)
        num_acc = mx.nd.sum(pred_label == rcnn_label)

        self.sum_metric += num_acc.asscalar()
        self.num_inst += rcnn_label.size


class RCNNL1LossMetric(mx.metric.EvalMetric):
    """ RCNN L1 loss. """

    def __init__(self):
        super(RCNNL1LossMetric, self).__init__('RCNNL1Loss')

    def update(self, labels, preds):
        """ Updates the internal evaluation result. """
        # label = [rcnn_bbox_target, rcnn_bbox_weight]
        # pred = [rcnn_reg]
        rcnn_bbox_target, rcnn_bbox_weight = labels
        rcnn_bbox_reg = preds[0]

        # calculate num_inst
        num_inst = mx.nd.sum(rcnn_bbox_weight) / 4

        # calculate smooth_l1
        loss = mx.nd.sum(
            rcnn_bbox_weight * mx.nd.smooth_l1(rcnn_bbox_reg - rcnn_bbox_target, scalar=1))

        self.sum_metric += loss.asscalar()
        self.num_inst += num_inst.asscalar()


class MaskAccMetric(mx.metric.EvalMetric):
    """ RCNN mask branch accuracy. """

    def __init__(self):
        super(MaskAccMetric, self).__init__('MaskAcc')

    def update(self, labels, preds):
        """ Updates the internal evaluation result. """
        # label = [rcnn_mask_target, rcnn_mask_weight]
        # pred = [rcnn_mask]
        rcnn_mask_target, rcnn_mask_weight = labels
        rcnn_mask = preds[0]

        # calculate num_inst
        num_inst = mx.nd.sum(rcnn_mask_weight)

        # rcnn_mask (b, n, c, h, w)
        pred_label = mx.nd.sigmoid(rcnn_mask) >= 0.5
        label = rcnn_mask_target >= 0.5
        # label (b, n, c, h, w)
        num_acc = mx.nd.sum((pred_label == label) * rcnn_mask_weight)

        self.sum_metric += num_acc.asscalar()
        self.num_inst += num_inst.asscalar()


class MaskFGAccMetric(mx.metric.EvalMetric):
    """ RCNN mask branch foreground accuracy. """

    def __init__(self):
        super(MaskFGAccMetric, self).__init__('MaskFGAcc')

    def update(self, labels, preds):
        """ Updates the internal evaluation result. """
        # label = [rcnn_mask_target, rcnn_mask_weight]
        # pred = [rcnn_mask]
        rcnn_mask_target, _ = labels
        rcnn_mask = preds[0]

        # calculate num_inst
        num_inst = mx.nd.sum(rcnn_mask_target)

        # rcnn_mask (b, n, c, h, w)
        pred_label = mx.nd.sigmoid(rcnn_mask) >= 0.5
        label = rcnn_mask_target >= 0.5
        # label (b, n, c, h, w)
        num_acc = mx.nd.sum((pred_label == label) * label)

        self.sum_metric += num_acc.asscalar()
        self.num_inst += num_inst.asscalar()
