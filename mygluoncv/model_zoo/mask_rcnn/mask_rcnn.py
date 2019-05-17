"""Mask R-CNN Model."""
from __future__ import absolute_import

import os
import warnings

import mxnet as mx
from mxnet import autograd
from mxnet.gluon import nn

from .rcnn_target import MaskTargetGenerator
from ..faster_rcnn.faster_rcnn import FasterRCNN
from ...nn.feature import FPNFeatureExpander

__all__ = ['MaskRCNN', 'get_mask_rcnn',
           'mask_rcnn_resnet50_v1b_coco',
           'mask_rcnn_fpn_resnet50_v1b_coco',
           'mask_rcnn_resnet101_v1d_coco',
           'mask_rcnn_fpn_resnet101_v1d_coco']


class Mask(nn.HybridBlock):
    r"""Mask predictor head

    Parameters
    ----------
    batch_images : int
        Used to reshape output
    classes : iterable of str
        Used to determine number of output channels, and store class names
    mask_channels : int
        Used to determine number of hidden channels
    deep_fcn : boolean, default False
        Whether to use deep mask branch (4 convs)

    """

    def __init__(self, batch_images, classes, mask_channels, deep_fcn=False, **kwargs):
        super(Mask, self).__init__(**kwargs)
        self._batch_images = batch_images
        self.classes = classes
        init = mx.init.Xavier(rnd_type='gaussian', factor_type='out', magnitude=2)
        with self.name_scope():
            if deep_fcn:
                self.deconv = nn.HybridSequential()
                for _ in range(4):
                    self.deconv.add(
                        nn.Conv2D(mask_channels, kernel_size=(3, 3), strides=(1, 1), padding=(1, 1),
                                  weight_initializer=init),
                        nn.Activation('relu'))
                self.deconv.add(
                    nn.Conv2DTranspose(mask_channels, kernel_size=(2, 2), strides=(2, 2),
                                       padding=(0, 0), weight_initializer=init))
            else:
                self.deconv = nn.Conv2DTranspose(mask_channels, kernel_size=(2, 2), strides=(2, 2),
                                                 padding=(0, 0), weight_initializer=init)
            self.mask = nn.Conv2D(len(classes), kernel_size=(1, 1), strides=(1, 1), padding=(0, 0),
                                  weight_initializer=init)

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, x):
        """Forward Mask Head.

        The behavior during training and inference is different.

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

    def reset_class(self, classes, reuse_weights=None):
        """Reset class for mask branch."""
        if reuse_weights:
            assert hasattr(self, 'classes'), "require old classes to reuse weights"
        old_classes = getattr(self, 'classes', [])
        self.classes = classes
        if isinstance(reuse_weights, (dict, list)):
            if isinstance(reuse_weights, dict):
                # trying to replace str with indices
                for k, v in reuse_weights.items():
                    if isinstance(v, str):
                        try:
                            v = old_classes.index(v)  # raise ValueError if not found
                        except ValueError:
                            raise ValueError(
                                "{} not found in old class names {}".format(v, old_classes))
                        reuse_weights[k] = v
                    if isinstance(k, str):
                        try:
                            new_idx = self.classes.index(k)  # raise ValueError if not found
                        except ValueError:
                            raise ValueError(
                                "{} not found in new class names {}".format(k, self.classes))
                        reuse_weights.pop(k)
                        reuse_weights[new_idx] = v
            else:
                new_map = {}
                for x in reuse_weights:
                    try:
                        new_idx = self.classes.index(x)
                        old_idx = old_classes.index(x)
                        new_map[new_idx] = old_idx
                    except ValueError:
                        warnings.warn("{} not found in old: {} or new class names: {}".format(
                            x, old_classes, self.classes))
                reuse_weights = new_map
        with self.name_scope():
            old_mask = self.mask
            ctx = list(old_mask.params.values())[0].list_ctx()
            # to avoid deferred init, number of in_channels must be defined
            in_channels = list(old_mask.params.values())[0].shape[1]
            init = mx.init.Xavier(rnd_type='gaussian', factor_type='out', magnitude=2)
            self.mask = nn.Conv2D(len(classes), kernel_size=(1, 1), strides=(1, 1), padding=(0, 0),
                                  weight_initializer=init, in_channels=in_channels)
            self.mask.initialize(ctx=ctx)
            if reuse_weights:
                assert isinstance(reuse_weights, dict)
                for old_params, new_params in zip(old_mask.params.values(),
                                                  self.mask.params.values()):
                    # slice and copy weights
                    old_data = old_params.data()
                    new_data = new_params.data()

                    for k, v in reuse_weights.items():
                        if k >= len(self.classes) or v >= len(old_classes):
                            warnings.warn("reuse mapping {}/{} -> {}/{} out of range".format(
                                k, self.classes, v, old_classes))
                            continue
                        new_data[k:k + 1] = old_data[v:v + 1]
                    # set data to new conv layers
                    new_params.set_data(new_data)


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
    deep_fcn : boolean, default False
            Whether to use deep mask branch (4 convs)
    """

    def __init__(self, features, top_features, classes, mask_channels=256, rcnn_max_dets=1000,
                 deep_fcn=False, **kwargs):
        super(MaskRCNN, self).__init__(features, top_features, classes,
                                       additional_output=True, **kwargs)
        self._rcnn_max_dets = rcnn_max_dets
        with self.name_scope():
            self.mask = Mask(self._max_batch, classes, mask_channels, deep_fcn=deep_fcn)
            if deep_fcn:
                roi_size = (self._roi_size[0] * 2, self._roi_size[1] * 2)
            else:
                roi_size = self._roi_size
            self._target_roi_size = roi_size
            self.mask_target = MaskTargetGenerator(
                self._max_batch, self._num_sample, self.num_class, self._target_roi_size)

    def hybrid_forward(self, F, x, gt_box=None):
        """Forward Mask RCNN network.

        The behavior during training and inference is different.

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
            # roi_batch_id = F.arange(0, self._max_batch, repeat=num_rois)
            roi_batch_id = F.arange(0, self._max_batch)
            roi_batch_id = F.repeat(roi_batch_id, num_rois)
            indices = F.stack(roi_batch_id, topk.reshape((-1,)), axis=0)
            ids = F.gather_nd(ids, indices).reshape((-4, self._max_batch, num_rois, 1))
            scores = F.gather_nd(scores, indices).reshape((-4, self._max_batch, num_rois, 1))
            boxes = F.gather_nd(boxes, indices).reshape((-4, self._max_batch, num_rois, 4))

            # create batch id and reshape for roi pooling
            padded_rois = F.concat(roi_batch_id.reshape((-1, 1)), boxes.reshape((-3, 0)), dim=-1)
            padded_rois = F.stop_gradient(padded_rois)

            # pool to roi features
            if self.num_stages > 1:
                # using FPN
                pooled_feat = self._pyramid_roi_feats(F, feat, padded_rois, self._roi_size,
                                                      self._strides, roi_mode=self._roi_mode)
            else:
                if self._roi_mode == 'pool':
                    pooled_feat = F.ROIPooling(
                        feat[0], padded_rois, self._roi_size, 1. / self._strides)
                elif self._roi_mode == 'align':
                    pooled_feat = F.contrib.ROIAlign(
                        feat[0], padded_rois, self._roi_size, 1. / self._strides, sample_ratio=2)
                else:
                    raise ValueError("Invalid roi mode: {}".format(self._roi_mode))

            # run top_features again
            if self.top_features is not None:
                top_feat = self.top_features(pooled_feat)
            else:
                top_feat = pooled_feat
            # (B, N, C, pooled_size * 2, pooled_size * 2)
            rcnn_mask = self.mask(top_feat)
            # index the B dimension (B * N,)
            # batch_ids = F.arange(0, self._max_batch, repeat=num_rois)
            batch_ids = F.arange(0, self._max_batch)
            batch_ids = F.repeat(batch_ids, num_rois)
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

    def reset_class(self, classes, reuse_weights=None):
        """Reset class categories and class predictors.

        Parameters
        ----------
        classes : iterable of str
            The new categories. ['apple', 'orange'] for example.
        reuse_weights : dict
            A {new_integer : old_integer} or mapping dict or {new_name : old_name} mapping dict,
            or a list of [name0, name1,...] if class names don't change.
            This allows the new predictor to reuse the
            previously trained weights specified.

        Example
        -------
        >>> net = gluoncv.model_zoo.get_model('mask_rcnn_resnet50_v1b_voc', pretrained=True)
        >>> # use direct name to name mapping to reuse weights
        >>> net.reset_class(classes=['person'], reuse_weights={'person':'person'})
        >>> # or use interger mapping, person is the first category in COCO
        >>> net.reset_class(classes=['person'], reuse_weights={0:0})
        >>> # you can even mix them
        >>> net.reset_class(classes=['person'], reuse_weights={'person':0})
        >>> # or use a list of string if class name don't change
        >>> net.reset_class(classes=['person'], reuse_weights=['person'])

        """
        self._clear_cached_op()
        super(MaskRCNN, self).reset_class(classes=classes, reuse_weights=reuse_weights)
        self.mask.reset_class(classes=classes, reuse_weights=reuse_weights)
        self.mask_target = MaskTargetGenerator(
            self._max_batch, self._num_sample, self.num_class, self._target_roi_size)

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
        if pretrained is `True`, this has no effect.
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
        roi_mode='align', roi_size=(14, 14), strides=16, clip=4.42,
        rpn_channel=1024, base_size=16, scales=(2, 4, 8, 16, 32),
        ratios=(0.5, 1, 2), alloc_size=(128, 128), rpn_nms_thresh=0.7,
        rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
        rpn_test_pre_nms=6000, rpn_test_post_nms=1000, rpn_min_size=0,
        num_sample=128, pos_iou_thresh=0.5, pos_ratio=0.25,
        **kwargs)


def mask_rcnn_fpn_resnet50_v1b_coco(pretrained=False, pretrained_base=True, **kwargs):
    r"""Mask RCNN model from the paper
    "He, K., Gkioxari, G., Doll&ar, P., & Girshick, R. (2017). Mask R-CNN"

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
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
    features = FPNFeatureExpander(
        network=base_network,
        outputs=['layers1_relu8_fwd', 'layers2_relu11_fwd', 'layers3_relu17_fwd',
                 'layers4_relu8_fwd'], num_filters=[256, 256, 256, 256], use_1x1=True,
        use_upsample=True, use_elewadd=True, use_p6=True, no_bias=False, pretrained=pretrained_base)
    top_features = None
    box_features = nn.HybridSequential()
    box_features.add(nn.AvgPool2D(pool_size=(3, 3), strides=2, padding=1))  # reduce to 7x7
    for _ in range(2):
        box_features.add(nn.Dense(1024, weight_initializer=mx.init.Normal(0.01)),
                         nn.Activation('relu'))
    train_patterns = '|'.join(['.*dense', '.*rpn', '.*mask', 'P',
                               '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv'])
    return get_mask_rcnn(
        name='fpn_resnet50_v1b', dataset='coco', pretrained=pretrained,
        features=features, top_features=top_features, classes=classes,
        box_features=box_features, mask_channels=256, rcnn_max_dets=1000,
        short=800, max_size=1333, min_stage=2, max_stage=6,
        train_patterns=train_patterns, nms_thresh=0.5, nms_topk=-1,
        post_nms=-1, roi_mode='align', roi_size=(14, 14),
        strides=(4, 8, 16, 32, 64), clip=4.42, rpn_channel=1024, base_size=16,
        scales=(2, 4, 8, 16, 32), ratios=(0.5, 1, 2), alloc_size=(384, 384),
        rpn_nms_thresh=0.7, rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
        rpn_test_pre_nms=6000, rpn_test_post_nms=1000, rpn_min_size=0,
        num_sample=512, pos_iou_thresh=0.5, pos_ratio=0.25, deep_fcn=True,
        **kwargs)


def mask_rcnn_resnet101_v1d_coco(pretrained=False, pretrained_base=True, **kwargs):
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
    >>> model = mask_rcnn_resnet101_v1d_coco(pretrained=True)
    >>> print(model)
    """
    from ..resnetv1b import resnet101_v1d
    from ...data import COCODetection
    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet101_v1d(pretrained=pretrained_base, dilated=False, use_global_stats=True)
    features = nn.HybridSequential()
    top_features = nn.HybridSequential()
    for layer in ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3']:
        features.add(getattr(base_network, layer))
    for layer in ['layer4']:
        top_features.add(getattr(base_network, layer))
    train_patterns = '|'.join(['.*dense', '.*rpn', '.*mask',
                               '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv'])
    return get_mask_rcnn(
        name='resnet101_v1d', dataset='coco', pretrained=pretrained,
        features=features, top_features=top_features, classes=classes,
        mask_channels=256, rcnn_max_dets=1000,
        short=800, max_size=1333, train_patterns=train_patterns,
        nms_thresh=0.5, nms_topk=-1, post_nms=-1,
        roi_mode='align', roi_size=(14, 14), strides=16, clip=4.42,
        rpn_channel=1024, base_size=16, scales=(2, 4, 8, 16, 32),
        ratios=(0.5, 1, 2), alloc_size=(128, 128), rpn_nms_thresh=0.7,
        rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
        rpn_test_pre_nms=6000, rpn_test_post_nms=1000, rpn_min_size=0,
        num_sample=128, pos_iou_thresh=0.5, pos_ratio=0.25,
        **kwargs)


def mask_rcnn_fpn_resnet101_v1d_coco(pretrained=False, pretrained_base=True, **kwargs):
    r"""Mask RCNN model from the paper
    "He, K., Gkioxari, G., Doll&ar, P., & Girshick, R. (2017). Mask R-CNN"

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = mask_rcnn_fpn_resnet101_v1d_coco(pretrained=True)
    >>> print(model)
    """
    from ..resnetv1b import resnet101_v1d
    from ...data import COCODetection
    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet101_v1d(pretrained=pretrained_base, dilated=False, use_global_stats=True)
    features = FPNFeatureExpander(
        network=base_network,
        outputs=['layers1_relu8_fwd', 'layers2_relu11_fwd', 'layers3_relu68_fwd',
                 'layers4_relu8_fwd'], num_filters=[256, 256, 256, 256], use_1x1=True,
        use_upsample=True, use_elewadd=True, use_p6=True, no_bias=False, pretrained=pretrained_base)
    top_features = None
    box_features = nn.HybridSequential()
    box_features.add(nn.AvgPool2D(pool_size=(3, 3), strides=2, padding=1))  # reduce to 7x7
    for _ in range(2):
        box_features.add(nn.Dense(1024, weight_initializer=mx.init.Normal(0.01)),
                         nn.Activation('relu'))
    train_patterns = '|'.join(['.*dense', '.*rpn', '.*mask', 'P',
                               '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv'])
    return get_mask_rcnn(
        name='fpn_resnet101_v1d', dataset='coco', pretrained=pretrained,
        features=features, top_features=top_features, classes=classes,
        box_features=box_features, mask_channels=256, rcnn_max_dets=1000,
        short=800, max_size=1333, min_stage=2, max_stage=6,
        train_patterns=train_patterns, nms_thresh=0.5, nms_topk=-1,
        post_nms=-1, roi_mode='align', roi_size=(14, 14),
        strides=(4, 8, 16, 32, 64), clip=4.42, rpn_channel=1024, base_size=16,
        scales=(2, 4, 8, 16, 32), ratios=(0.5, 1, 2), alloc_size=(384, 384),
        rpn_nms_thresh=0.7, rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
        rpn_test_pre_nms=6000, rpn_test_post_nms=1000, rpn_min_size=0,
        num_sample=512, pos_iou_thresh=0.5, pos_ratio=0.25, deep_fcn=True,
        **kwargs)
