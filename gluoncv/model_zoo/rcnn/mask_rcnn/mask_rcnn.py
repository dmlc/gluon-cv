"""Mask R-CNN Model."""
from __future__ import absolute_import

import os
import warnings

import mxnet as mx
from mxnet import autograd
from mxnet.gluon import nn
from mxnet.gluon.contrib.nn import SyncBatchNorm

from .rcnn_target import MaskTargetGenerator
from ..faster_rcnn import FasterRCNN
from ..rcnn import custom_rcnn_fpn

__all__ = ['MaskRCNN', 'get_mask_rcnn', 'custom_mask_rcnn_fpn']


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
    num_fcn_convs : int, default 0
        number of convolution blocks before deconv layer. For FPN network this is typically 4.

    """

    def __init__(self, batch_images, classes, mask_channels, num_fcn_convs=0, norm_layer=None,
                 norm_kwargs=None, **kwargs):
        super(Mask, self).__init__(**kwargs)
        self._batch_images = batch_images
        self.classes = classes
        init = mx.init.Xavier(rnd_type='gaussian', factor_type='out', magnitude=2)
        with self.name_scope():
            if num_fcn_convs > 0:
                self.deconv = nn.HybridSequential()
                for _ in range(num_fcn_convs):
                    self.deconv.add(
                        nn.Conv2D(mask_channels, kernel_size=(3, 3), strides=(1, 1),
                                  padding=(1, 1), weight_initializer=init))
                    if norm_layer is not None and norm_layer is SyncBatchNorm:
                        self.deconv.add(norm_layer(**norm_kwargs))
                    self.deconv.add(nn.Activation('relu'))
                self.deconv.add(
                    nn.Conv2DTranspose(mask_channels, kernel_size=(2, 2), strides=(2, 2),
                                       padding=(0, 0), weight_initializer=init))
                if norm_layer is not None and norm_layer is SyncBatchNorm:
                    self.deconv.add(norm_layer(**norm_kwargs))
            else:
                # this is for compatibility of older models.
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
        if autograd.is_training():
            x = x.reshape((-4, self._batch_images, -1, 0, 0, 0))
        else:
            # always use batch_size = 1 for inference
            x = x.reshape((-4, 1, -1, 0, 0, 0))
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
    rcnn_max_dets : int, default is 1000
        Number of rois to retain in RCNN.
        Upper bounded by min of rpn_test_pre_nms and rpn_test_post_nms.
    rpn_test_pre_nms : int, default is 6000
        Filter top proposals before NMS in testing of RPN.
    rpn_test_post_nms : int, default is 1000
        Return top proposal results after NMS in testing of RPN.
        Will be set to rpn_test_pre_nms if it is larger than rpn_test_pre_nms.
    target_roi_scale : int, default 1
        Ratio of mask output roi / input roi. For model with FPN, this is typically 2.
    num_fcn_convs : int, default 0
        number of convolution blocks before deconv layer. For FPN network this is typically 4.
    """

    def __init__(self, features, top_features, classes, mask_channels=256, rcnn_max_dets=1000,
                 rpn_test_pre_nms=6000, rpn_test_post_nms=1000, target_roi_scale=1, num_fcn_convs=0,
                 norm_layer=None, norm_kwargs=None, **kwargs):
        super(MaskRCNN, self).__init__(features, top_features, classes,
                                       rpn_test_pre_nms=rpn_test_pre_nms,
                                       rpn_test_post_nms=rpn_test_post_nms,
                                       additional_output=True, **kwargs)
        if min(rpn_test_pre_nms, rpn_test_post_nms) < rcnn_max_dets:
            rcnn_max_dets = min(rpn_test_pre_nms, rpn_test_post_nms)
        self._rcnn_max_dets = rcnn_max_dets
        with self.name_scope():
            self.mask = Mask(self._batch_size, classes, mask_channels, num_fcn_convs=num_fcn_convs,
                             norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            roi_size = (self._roi_size[0] * target_roi_scale, self._roi_size[1] * target_roi_scale)
            self._target_roi_size = roi_size
            self.mask_target = MaskTargetGenerator(
                self._batch_size, self._num_sample, self.num_class, self._target_roi_size)

    def hybrid_forward(self, F, x, gt_box=None, gt_label=None):
        """Forward Mask RCNN network.

        The behavior during training and inference is different.

        Parameters
        ----------
        x : mxnet.nd.NDArray or mxnet.symbol
            The network input tensor.
        gt_box : type, only required during training
            The ground-truth bbox tensor with shape (1, N, 4).
        gt_label : type, only required during training
            The ground-truth label tensor with shape (B, 1, 4).

        Returns
        -------
        (ids, scores, bboxes, masks)
            During inference, returns final class id, confidence scores, bounding
            boxes, segmentation masks.

        """
        if autograd.is_training():
            cls_pred, box_pred, rpn_box, samples, matches, raw_rpn_score, raw_rpn_box, anchors, \
            cls_targets, box_targets, box_masks, top_feat, indices = \
                super(MaskRCNN, self).hybrid_forward(F, x, gt_box, gt_label)
            top_feat = F.reshape(top_feat.expand_dims(0), (self._batch_size, -1, 0, 0, 0))
            top_feat = F.concat(
                *[F.take(F.slice_axis(top_feat, axis=0, begin=i, end=i + 1).squeeze(),
                         F.slice_axis(indices, axis=0, begin=i, end=i + 1).squeeze())
                  for i in range(self._batch_size)], dim=0)
            mask_pred = self.mask(top_feat)

            return cls_pred, box_pred, mask_pred, rpn_box, samples, matches, raw_rpn_score, \
                   raw_rpn_box, anchors, cls_targets, box_targets, box_masks, indices
        else:
            batch_size = 1
            ids, scores, boxes, feat = \
                super(MaskRCNN, self).hybrid_forward(F, x)

            # (B, N * (C - 1), 1) -> (B, N * (C - 1)) -> (B, topk)
            num_rois = self._rcnn_max_dets
            order = F.argsort(scores.squeeze(axis=-1), axis=1, is_ascend=False)
            topk = F.slice_axis(order, axis=1, begin=0, end=num_rois)

            # pick from (B, N * (C - 1), X) to (B * topk, X) -> (B, topk, X)
            # roi_batch_id = F.arange(0, self._max_batch, repeat=num_rois)
            roi_batch_id = F.arange(0, batch_size)
            roi_batch_id = F.repeat(roi_batch_id, num_rois)
            indices = F.stack(roi_batch_id, topk.reshape((-1,)), axis=0)
            ids = F.gather_nd(ids, indices).reshape((-4, batch_size, num_rois, 1))
            scores = F.gather_nd(scores, indices).reshape((-4, batch_size, num_rois, 1))
            boxes = F.gather_nd(boxes, indices).reshape((-4, batch_size, num_rois, 4))

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
            batch_ids = F.arange(0, batch_size)
            batch_ids = F.repeat(batch_ids, num_rois)
            # index the N dimension (B * N,)
            roi_ids = F.tile(F.arange(0, num_rois), reps=batch_size)
            # index the C dimension (B * N,)
            class_ids = ids.reshape((-1,))
            # clip to 0 to max class
            class_ids = F.clip(class_ids, 0, self.num_class)
            # pick from (B, N, C, PS*2, PS*2) -> (B * N, PS*2, PS*2)
            indices = F.stack(batch_ids, roi_ids, class_ids, axis=0)
            masks = F.gather_nd(rcnn_mask, indices)
            # (B * N, PS*2, PS*2) -> (B, N, PS*2, PS*2)
            masks = masks.reshape((-4, batch_size, num_rois, 0, 0))
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
            self._batch_size, self._num_sample, self.num_class, self._target_roi_size)


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
    net = MaskRCNN(minimal_opset=pretrained, **kwargs)
    if pretrained:
        from ....model_zoo.model_store import get_model_file
        full_name = '_'.join(('mask_rcnn', name, dataset))
        net.load_parameters(get_model_file(full_name, tag=pretrained, root=root), ctx=ctx,
                            ignore_extra=True, allow_missing=True)
    else:
        for v in net.collect_params().values():
            try:
                v.reset_ctx(ctx)
            except ValueError:
                pass
    return net


def custom_mask_rcnn_fpn(classes, transfer=None, dataset='custom', pretrained_base=True,
                         base_network_name='resnet18_v1b', norm_layer=nn.BatchNorm,
                         norm_kwargs=None, sym_norm_layer=None, sym_norm_kwargs=None,
                         num_fpn_filters=256, num_box_head_conv=4, num_box_head_conv_filters=256,
                         num_box_head_dense_filters=1024, **kwargs):
    r"""Mask RCNN model with resnet base network and FPN on custom dataset.

    Parameters
    ----------
    classes : iterable of str
        Names of custom foreground classes. `len(classes)` is the number of foreground classes.
    transfer : str or None
        Dataset from witch to transfer from. If not `None`, will try to reuse pre-trained weights
        from faster RCNN networks trained on other dataset, specified by the parameter.
    dataset : str, default 'custom'
        Dataset name attached to the network name
    pretrained_base : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    base_network_name : str, default 'resnet18_v1b'
        base network for mask RCNN. Currently support: 'resnet18_v1b', 'resnet50_v1b',
        and 'resnet101_v1d'
    norm_layer : nn.HybridBlock, default nn.BatchNorm
        Gluon normalization layer to use. Default is frozen batch normalization layer.
    norm_kwargs : dict
        Keyword arguments for gluon normalization layer
    sym_norm_layer : nn.SymbolBlock, default `None`
        Symbol normalization layer to use in FPN. This is due to FPN being implemented using
        SymbolBlock. Default is `None`, meaning no normalization layer will be used in FPN.
    sym_norm_kwargs : dict
        Keyword arguments for symbol normalization layer used in FPN.
    num_fpn_filters : int, default 256
        Number of filters for FPN output layers.
    num_box_head_conv : int, default 4
        Number of convolution layers to use in box head if batch normalization is not frozen.
    num_box_head_conv_filters : int, default 256
        Number of filters for convolution layers in box head.
        Only applicable if batch normalization is not frozen.
    num_box_head_dense_filters : int, default 1024
        Number of hidden units for the last fully connected layer in box head.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Returns
    -------
    mxnet.gluon.HybridBlock
        Hybrid faster RCNN network.
    """
    use_global_stats = norm_layer is nn.BatchNorm
    train_patterns = '|'.join(['.*dense', '.*rpn', '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv',
                               '.*mask', 'P']) if use_global_stats \
        else '(?!.*moving)'  # excluding symbol bn moving mean and var'''

    if transfer is None:
        features, top_features, box_features = \
            custom_rcnn_fpn(pretrained_base, base_network_name, norm_layer, norm_kwargs,
                            sym_norm_layer, sym_norm_kwargs, num_fpn_filters, num_box_head_conv,
                            num_box_head_conv_filters, num_box_head_dense_filters)
        return get_mask_rcnn(
            name='fpn_' + base_network_name, dataset=dataset, features=features,
            top_features=top_features, classes=classes, box_features=box_features,
            train_patterns=train_patterns, **kwargs)
    else:
        from ....model_zoo import get_model
        module_list = ['fpn']
        if norm_layer is not None:
            module_list.append(norm_layer)
        net = get_model('_'.join(['mask_rcnn'] + module_list + [base_network_name, str(transfer)]),
                        pretrained=True, **kwargs)
        reuse_classes = [x for x in classes if x in net.classes]
        net.reset_class(classes, reuse_weights=reuse_classes)
    return net
