"""Data parallel task for Faster RCNN Model."""

from mxnet import autograd

from gluoncv.utils.parallel import Parallelizable


class ForwardBackwardTask(Parallelizable):
    """ Faster R-CNN training task that can be scheduled concurrently using Parallel.
    Parameters
    ----------
    net : gluon.HybridBlock
        Faster R-CNN network.
    optimizer : gluon.Trainer
        Optimizer for the training.
    rpn_cls_loss : gluon.loss
        RPN box classification loss.
    rpn_box_loss : gluon.loss
        RPN box regression loss.
    rcnn_cls_loss : gluon.loss
        R-CNN box head classification loss.
    rcnn_box_loss : gluon.loss
        R-CNN box head regression loss.
    mix_ratio : int
        Object detection mixup ratio.
    amp_enabled : bool
        Whether to enable Automatic Mixed Precision.
    """

    def __init__(self, net, optimizer, rpn_cls_loss, rpn_box_loss, rcnn_cls_loss, rcnn_box_loss,
                 mix_ratio, amp_enabled):
        super(ForwardBackwardTask, self).__init__()
        self.net = net
        self._optimizer = optimizer
        self.rpn_cls_loss = rpn_cls_loss
        self.rpn_box_loss = rpn_box_loss
        self.rcnn_cls_loss = rcnn_cls_loss
        self.rcnn_box_loss = rcnn_box_loss
        self.mix_ratio = mix_ratio
        self.amp_enabled = amp_enabled

    def forward_backward(self, x):
        data, label, rpn_cls_targets, rpn_box_targets, rpn_box_masks = x
        with autograd.record():
            gt_label = label[:, :, 4:5]
            gt_box = label[:, :, :4]
            cls_pred, box_pred, _, _, _Z, rpn_score, rpn_box, _, cls_targets, \
                box_targets, box_masks, _ = self.net(data, gt_box, gt_label)
            # losses of rpn
            rpn_score = rpn_score.squeeze(axis=-1)
            num_rpn_pos = (rpn_cls_targets >= 0).sum()
            rpn_loss1 = self.rpn_cls_loss(rpn_score, rpn_cls_targets,
                                          rpn_cls_targets >= 0) * rpn_cls_targets.size / num_rpn_pos
            rpn_loss2 = self.rpn_box_loss(rpn_box, rpn_box_targets,
                                          rpn_box_masks) * rpn_box.size / num_rpn_pos
            # rpn overall loss, use sum rather than average
            rpn_loss = rpn_loss1 + rpn_loss2
            # losses of rcnn
            num_rcnn_pos = (cls_targets >= 0).sum()
            rcnn_loss1 = self.rcnn_cls_loss(
                cls_pred, cls_targets, cls_targets.expand_dims(-1) >= 0) * cls_targets.size / \
                         num_rcnn_pos
            rcnn_loss2 = self.rcnn_box_loss(box_pred, box_targets, box_masks) * box_pred.size / \
                         num_rcnn_pos
            rcnn_loss = rcnn_loss1 + rcnn_loss2
            # overall losses
            total_loss = rpn_loss.sum() * self.mix_ratio + rcnn_loss.sum() * self.mix_ratio

            rpn_loss1_metric = rpn_loss1.mean() * self.mix_ratio
            rpn_loss2_metric = rpn_loss2.mean() * self.mix_ratio
            rcnn_loss1_metric = rcnn_loss1.mean() * self.mix_ratio
            rcnn_loss2_metric = rcnn_loss2.mean() * self.mix_ratio
            rpn_acc_metric = [[rpn_cls_targets, rpn_cls_targets >= 0], [rpn_score]]
            rpn_l1_loss_metric = [[rpn_box_targets, rpn_box_masks], [rpn_box]]
            rcnn_acc_metric = [[cls_targets], [cls_pred]]
            rcnn_l1_loss_metric = [[box_targets, box_masks], [box_pred]]

            if self.amp_enabled:
                from mxnet.contrib import amp
                with amp.scale_loss(total_loss, self._optimizer) as scaled_losses:
                    autograd.backward(scaled_losses)
            else:
                total_loss.backward()

        return rpn_loss1_metric, rpn_loss2_metric, rcnn_loss1_metric, rcnn_loss2_metric, \
               rpn_acc_metric, rpn_l1_loss_metric, rcnn_acc_metric, rcnn_l1_loss_metric
