"""Mask R-CNN Model."""
from __future__ import absolute_import

import os
import mxnet as mx
from mxnet import autograd
from mxnet.gluon import nn
from ..faster_rcnn.faster_rcnn import FasterRCNN
from .rcnn_target import MaskTargetGenerator

__all__ = ['MaskRCNN', 'get_mask_rcnn',
           'mask_rcnn_resnet50_v1b_coco']


class Mask(nn.HybridBlock):
    r"""Mask predictor head

    Parameters
    ----------
    batch_images : int
        Used to reshape output
    num_classes : int
        Used to determine number of output channels
    mask_channels : int
        Used to determine number of hidden channels

    """
    def __init__(self, batch_images, num_classes, mask_channels, **kwargs):
        super(Mask, self).__init__(**kwargs)
        self._batch_images = batch_images
        init = mx.init.Xavier(rnd_type='gaussian', factor_type='out', magnitude=2)
        with self.name_scope():
            self.deconv = nn.Conv2DTranspose(mask_channels, kernel_size=(2, 2), strides=(2, 2),
                                             padding=(0, 0), weight_initializer=init)
            self.mask = nn.Conv2D(num_classes, kernel_size=(1, 1), strides=(1, 1), padding=(0, 0),
                                  weight_initializer=init)

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, x):
        """Forward Mask Head.

        The behavior during traing and inference is different.

        Parameters
        ----------
        x : mxnet.nd.NDArray or mxnet.symbol
            The network input tensor of shape (B * N, fC, fH, fW).

        Returns
        -------
        x : mxnet.nd.NDArray or mxnet.symbol
            Mask prediction of shape (B, N, C, MS, MS)

        """
        # (B * N, mask_channels, MS, MS)
        x = F.relu(self.deconv(x))
        # (B * N, C, MS, MS)
        x = self.mask(x)
        # (B * N, C, MS, MS) -> (B, N, C, MS, MS)
        x = x.reshape((-4, self._batch_images, -1, 0, 0, 0))
        return x


class MaskRCNN(FasterRCNN):
    r"""Mask RCNN network.

    Parameters
    ----------
    features : gluon.HybridBlock
        Base feature extractor before feature pooling layer.
    top_features : gluon.HybridBlock
        Tail feature extractor after feature pooling layer.
    classes : iterable of str
        Names of categories, its length is ``num_class``.
    mask_channels : int, default is 256
        Number of channels in mask prediction

    """
    def __init__(self, features, top_features, classes,
                 mask_channels=256, rcnn_max_dets=1000, **kwargs):
        super(MaskRCNN, self).__init__(features, top_features, classes,
                                       additional_output=True, **kwargs)
        self._rcnn_max_dets = rcnn_max_dets
        with self.name_scope():
            self.mask = Mask(self._max_batch, self.num_class, mask_channels)
            self.mask_target = MaskTargetGenerator(
                self._max_batch, self._num_sample, self.num_class, self._roi_size)

    def hybrid_forward(self, F, x, gt_box=None):
        """Forward Mask RCNN network.

        The behavior during traing and inference is different.

        Parameters
        ----------
        x : mxnet.nd.NDArray or mxnet.symbol
            The network input tensor.
        gt_box : type, only required during training
            The ground-truth bbox tensor with shape (1, N, 4).

        Returns
        -------
        (ids, scores, bboxes, masks)
            During inference, returns final class id, confidence scores, bounding
            boxes, segmentation masks.

        """
        if autograd.is_training():
            cls_pred, box_pred, rpn_box, samples, matches, \
                raw_rpn_score, raw_rpn_box, anchors, top_feat = \
                super(MaskRCNN, self).hybrid_forward(F, x, gt_box)
            mask_pred = self.mask(top_feat)
            return cls_pred, box_pred, mask_pred, rpn_box, samples, matches, \
                 raw_rpn_score, raw_rpn_box, anchors
        else:
            ids, scores, boxes, feat = \
                super(MaskRCNN, self).hybrid_forward(F, x)

            # (B, N * (C - 1), 1) -> (B, N * (C - 1)) -> (B, topk)
            num_rois = self._rcnn_max_dets
            order = F.argsort(scores.squeeze(axis=-1), axis=1, is_ascend=False)
            topk = F.slice_axis(order, axis=1, begin=0, end=num_rois)

            # pick from (B, N * (C - 1), X) to (B * topk, X) -> (B, topk, X)
            roi_batch_id = F.arange(0, self._max_batch, repeat=num_rois)
            indices = F.stack(roi_batch_id, topk.reshape((-1,)), axis=0)
            ids = F.gather_nd(ids, indices).reshape((-4, self._max_batch, num_rois, 1))
            scores = F.gather_nd(scores, indices).reshape((-4, self._max_batch, num_rois, 1))
            boxes = F.gather_nd(boxes, indices).reshape((-4, self._max_batch, num_rois, 4))

            # create batch id and reshape for roi pooling
            padded_rois = F.concat(roi_batch_id.reshape((-1, 1)), boxes.reshape((-3, 0)), dim=-1)
            padded_rois = F.stop_gradient(padded_rois)

            # pool to roi features
            if self._roi_mode == 'pool':
                pooled_feat = F.ROIPooling(
                    feat, padded_rois, self._roi_size, 1. / self._stride)
            elif self._roi_mode == 'align':
                pooled_feat = F.contrib.ROIAlign(
                    feat, padded_rois, self._roi_size, 1. / self._stride, sample_ratio=2)
            else:
                raise ValueError("Invalid roi mode: {}".format(self._roi_mode))

            # run top_features again
            top_feat = self.top_features(pooled_feat)
            # (B, N, C, pooled_size * 2, pooled_size * 2)
            rcnn_mask = self.mask(top_feat)
            # index the B dimension (B * N,)
            batch_ids = F.arange(0, self._max_batch, repeat=num_rois)
            # index the N dimension (B * N,)
            roi_ids = F.tile(F.arange(0, num_rois), reps=self._max_batch)
            # index the C dimension (B * N,)
            class_ids = ids.reshape((-1,))
            # clip to 0 to max class
            class_ids = F.clip(class_ids, 0, self.num_class)
            # pick from (B, N, C, PS*2, PS*2) -> (B * N, PS*2, PS*2)
            indices = F.stack(batch_ids, roi_ids, class_ids, axis=0)
            masks = F.gather_nd(rcnn_mask, indices)
            # (B * N, PS*2, PS*2) -> (B, N, PS*2, PS*2)
            masks = masks.reshape((-4, self._max_batch, num_rois, 0, 0))
            # output prob
            masks = F.sigmoid(masks)

            # ids (B, N, 1), scores (B, N, 1), boxes (B, N, 4), masks (B, N, PS*2, PS*2)
            return ids, scores, boxes, masks

def get_mask_rcnn(name, dataset, pretrained=False, ctx=mx.cpu(),
                  root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    r"""Utility function to return mask rcnn networks.

    Parameters
    ----------
    name : str
        Model name.
    dataset : str
        The name of dataset.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : mxnet.Context
        Context such as mx.cpu(), mx.gpu(0).
    root : str
        Model weights storing path.

    Returns
    -------
    mxnet.gluon.HybridBlock
        The Mask RCNN network.

    """
    net = MaskRCNN(**kwargs)
    if pretrained:
        from ..model_store import get_model_file
        full_name = '_'.join(('mask_rcnn', name, dataset))
        net.load_parameters(get_model_file(full_name, tag=pretrained, root=root), ctx=ctx)
    return net

def mask_rcnn_resnet50_v1b_coco(pretrained=False, pretrained_base=True, **kwargs):
    r"""Mask RCNN model from the paper
    "He, K., Gkioxari, G., Doll&ar, P., & Girshick, R. (2017). Mask R-CNN"

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `Ture`, this has no effect.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = mask_rcnn_resnet50_v1b_coco(pretrained=True)
    >>> print(model)
    """
    from ..resnetv1b import resnet50_v1b
    from ...data import COCODetection
    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet50_v1b(pretrained=pretrained_base, dilated=False, use_global_stats=True)
    features = nn.HybridSequential()
    top_features = nn.HybridSequential()
    for layer in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']:
        features.add(getattr(base_network, layer))
    for layer in ['layer4']:
        top_features.add(getattr(base_network, layer))
    train_patterns = '|'.join(['.*dense', '.*rpn', '.*mask',
                               '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv'])
    return get_mask_rcnn(
        name='resnet50_v1b', dataset='coco', pretrained=pretrained,
        features=features, top_features=top_features, classes=classes,
        mask_channels=256, rcnn_max_dets=1000,
        short=800, max_size=1333, train_patterns=train_patterns,
        nms_thresh=0.5, nms_topk=-1, post_nms=-1,
        roi_mode='align', roi_size=(14, 14), stride=16, clip=4.42,
        rpn_channel=1024, base_size=16, scales=(2, 4, 8, 16, 32),
        ratios=(0.5, 1, 2), alloc_size=(128, 128), rpn_nms_thresh=0.7,
        rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
        rpn_test_pre_nms=6000, rpn_test_post_nms=1000, rpn_min_size=0,
        num_sample=128, pos_iou_thresh=0.5, pos_ratio=0.25,
        **kwargs)
