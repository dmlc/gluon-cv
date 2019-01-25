"""Feature Pyramid Network Detector."""
from __future__ import absolute_import

import os
import mxnet as mx
from mxnet import autograd
from mxnet.gluon import nn
from ..rpn import RPN
from ..rcnn import RCNN
from ..faster_rcnn import RCNNTargetGenerator, RCNNTargetSampler
from ...nn.feature import FPNFeatureExpander

__all__ = ['FPN', 'get_faster_rcnn_fpn',
           'faster_rcnn_fpn_resnet50_v1b_voc',
           'faster_rcnn_fpn_resnet50_v1b_coco']


class FPN(RCNN):
    r"""FPN network.
    Parameters
    ----------
    network : string or None
        Name of the base network, if `None` is used, will instantiate the
        base network from `features` directly instead of composing.
    features : gluon.HybridBlock
        Base feature extractor before feature pooling layer.
    num_filters : list
        Output channels for each FPN stage.
    use_1x1 : bool, default is true.
        1x1 Convolution transition for ``lateral connection``.
    use_upsample : bool, default is true. 
        Nearest upsample strategy for ``lateral connection``.
    use_elewadd : bool, default is true. 
        Element-wise add operation for ``lateral connection``.
    use_p6 : bool, default is true.
        Use ``P6`` for RPN network. P6 will be discarded in RCNN stage.
    no_bias : bool, default is true 
        Whether use bias for Convolution operation.
    pretrained_base : true 
        Load pretrained base network, the extra layers are randomized. 
    top_features : gluon.HybridBlock
        TODO(Angzz) Whether FPN need top features?
        Tail feature extractor after feature pooling layer.
    classes : iterable of str
        Names of categories, its length is ``num_class``.
    short : int, default is 600.
        Input image short side size.
    max_size : int, default is 1000.
        Maximum size of input image long side.
    min_stage : int, default is 2
        Minimum stage NO. for FPN stages.
    max_stage : int, default is 5
        Maximum stage NO. for FPN stages.
    train_patterns : str, default is None.
        Matching pattern for trainable parameters.
    nms_thresh : float, default is 0.3.
        Non-maximum suppression threshold. You can speficy < 0 or > 1 to disable NMS.
    nms_topk : int, default is 400
        Apply NMS to top k detection results, use -1 to disable so that every Detection
         result is used in NMS.
    post_nms : int, default is 100
        Only return top `post_nms` detection results, the rest is discarded. The number is
        based on COCO dataset which has maximum 100 objects per image. You can adjust this
        number if expecting more objects. You can use -1 to return all detections.
    roi_mode : str, default is align
        ROI pooling mode. Currently support 'pool' and 'align'.
    roi_size : tuple of int, length 2, default is (14, 14)
        (height, width) of the ROI region.
    strides : list or tuple
        Feature map stride with respect to original image.
        This is usually the ratio between original image size and feature map size.
    clip : float, default is None
        Clip bounding box target to this value.
    rpn_channel : int, default is 1024
        Channel number used in RPN convolutional layers.
    base_size : int
        The width(and height) of reference anchor box.
    scales : iterable of float, default is (8, 16, 32)
        The areas of anchor boxes.
        We use the following form to compute the shapes of anchors:
        .. math::
            width_{anchor} = size_{base} \times scale \times \sqrt{ 1 / ratio}
            height_{anchor} = size_{base} \times scale \times \sqrt{ratio}
    ratios : iterable of float, default is (0.5, 1, 2)
        The aspect ratios of anchor boxes. We expect it to be a list or tuple.
    alloc_size : tuple of int
        Allocate size for the anchor boxes as (H, W).
        Usually we generate enough anchors for large feature map, e.g. 128x128.
        Later in inference we can have variable input sizes,
        at which time we can crop corresponding anchors from this large
        anchor map so we can skip re-generating anchors for each input.
    rpn_train_pre_nms : int, default is 12000
        Filter top proposals before NMS in training of RPN.
    rpn_train_post_nms : int, default is 2000
        Return top proposal results after NMS in training of RPN.
    rpn_test_pre_nms : int, default is 6000
        Filter top proposals before NMS in testing of RPN.
    rpn_test_post_nms : int, default is 300
        Return top proposal results after NMS in testing of RPN.
    rpn_nms_thresh : float, default is 0.7
        IOU threshold for NMS. It is used to remove overlapping proposals.
    train_pre_nms : int, default is 12000
        Filter top proposals before NMS in training.
    train_post_nms : int, default is 2000
        Return top proposal results after NMS in training.
    test_pre_nms : int, default is 6000
        Filter top proposals before NMS in testing.
    test_post_nms : int, default is 300
        Return top proposal results after NMS in testing.
    rpn_min_size : int, default is 16
        Proposals whose size is smaller than ``min_size`` will be discarded.
    num_sample : int, default is 128
        Number of samples for RCNN targets.
    pos_iou_thresh : float, default is 0.5
        Proposal whose IOU larger than ``pos_iou_thresh`` is regarded as positive samples.
    pos_ratio : float, default is 0.25
        ``pos_ratio`` defines how many positive samples (``pos_ratio * num_sample``) is
        to be sampled.
    max_num_gt : int, default is 300
        Maximum ground-truth number in whole training dataset. This is only an upper bound, not
        necessarily very precise. However, using a very big number may impact the training speed.
    additional_output : boolean, default is False
        ``additional_output`` is only used for Mask R-CNN to get internal outputs.
    Attributes
    ----------
    classes : iterable of str
        Names of categories, its length is ``num_class``.
    num_class : int
        Number of positive categories.
    short : int
        Input image short side size.
    max_size : int
        Maximum size of input image long side.
    train_patterns : str
        Matching pattern for trainable parameters.
    nms_thresh : float
        Non-maximum suppression threshold. You can speficy < 0 or > 1 to disable NMS.
    nms_topk : int
        Apply NMS to top k detection results, use -1 to disable so that every Detection
         result is used in NMS.
    post_nms : int
        Only return top `post_nms` detection results, the rest is discarded. The number is
        based on COCO dataset which has maximum 100 objects per image. You can adjust this
        number if expecting more objects. You can use -1 to return all detections.
    target_generator : gluon.Block
        Generate training targets with boxes, samples, matches, gt_label and gt_box.
    """

    def __init__(self, network, features, num_filters, use_1x1, use_upsample,
                 use_elewadd, use_p6, no_bias, pretrained_base, top_features, classes,
                 short=600, max_size=1000, min_stage=2, max_stage=5, train_patterns=None,
                 nms_thresh=0.3, nms_topk=400, post_nms=100, roi_mode='align',
                 roi_size=(14, 14), strides=(4, 8, 16, 32), clip=None,
                 rpn_channel=1024, base_size=16, scales=(2, 4, 8, 16),
                 ratios=(0.5, 1, 2), alloc_size=(128, 128), rpn_nms_thresh=0.7,
                 rpn_train_pre_nms=12000, rpn_train_post_nms=2000,
                 rpn_test_pre_nms=6000, rpn_test_post_nms=300, rpn_min_size=16,
                 num_sample=128, pos_iou_thresh=0.5, pos_ratio=0.25,
                 max_num_gt=300, additional_output=False, ctx=mx.cpu(), **kwargs
                 ):
        # 2 FC layer before RCNN cls and reg
        fchead = nn.HybridSequential()
        for _ in range(2):
            fchead.add(nn.Dense(1024, weight_initializer=mx.init.Normal(0.01)))
            fchead.add(nn.Activation('relu'))
        super(FPN, self).__init__(
            features=features, top_features=top_features, classes=classes, box_features=fchead,
            short=short, max_size=max_size, train_patterns=train_patterns,
            nms_thresh=nms_thresh, nms_topk=nms_topk, post_nms=post_nms,
            roi_mode=roi_mode, roi_size=roi_size, clip=clip, stride=None, **kwargs)

        assert len(scales) >= 4, "The num_stages in FPN must over 4."
        self.num_stages = num_stages = len(scales)
        self._use_p6 = use_p6
        if self._use_p6:
            assert num_stages == 5, "If use_p6 for RPN proposal in FPN, " \
                                    "the num_stages must be 5, usually [P2, P3, P4, P5, P6]."
        self.ashape = alloc_size[0]  # ashape is used for rpn target generation
        self._max_batch = 1  # currently only support batch size = 1
        self._num_sample = num_sample
        self._min_stage = min_stage
        self._max_stage = max_stage
        self._roi_size = roi_size
        self._pool_strides = strides
        self._rpn_nms_thresh = rpn_nms_thresh
        self._additional_output = additional_output
        self._target_generator = {RCNNTargetGenerator(self.num_class)}

        with self.name_scope():
            self.rpn = RPN(
                channels=rpn_channel, strides=strides, base_size=base_size,
                scales=scales, ratios=ratios, alloc_size=alloc_size,
                clip=clip, nms_thresh=rpn_nms_thresh, train_pre_nms=rpn_train_pre_nms,
                train_post_nms=rpn_train_post_nms, test_pre_nms=rpn_test_pre_nms,
                test_post_nms=rpn_test_post_nms, min_size=rpn_min_size, multi_level=True)

            # Feature symbols
            self.features = FPNFeatureExpander(
                network=network, outputs=features, num_filters=num_filters,
                use_1x1=use_1x1, use_upsample=use_upsample, use_elewadd=use_elewadd,
                use_p6=use_p6, no_bias=no_bias, pretrained=pretrained_base, ctx=ctx)
            # Sample RCNN target
            self.sampler = RCNNTargetSampler(
                num_image=self._max_batch, num_proposal=rpn_train_post_nms,
                num_sample=num_sample, pos_iou_thresh=pos_iou_thresh,
                pos_ratio=pos_ratio, max_num_gt=max_num_gt)

    @property
    def target_generator(self):
        """Returns stored target generator
        Returns
        -------
        mxnet.gluon.HybridBlock
            The RCNN target generator
        """
        return list(self._target_generator)[0]

    def reset_class(self, classes):
        super(FPN, self).reset_class(classes)
        self._target_generator = {RCNNTargetGenerator(self.num_class)}

    def _pyramid_roi_feats(self, F, features, rpn_rois, roi_size, strides, roi_mode='pool',
                           eps=1e-6):
        """Assign rpn_rois to specific FPN layers according to its area
           and then perform `ROIPooling` or `ROIAlign` to generate final
           region proposals aggregated features.
        Parameters
        ----------
        features : list of mx.ndarray or mx.symbol
            Features extracted from FPN base network
        rpn_rois : mx.ndarray or mx.symbol
            (N, 5) with [[batch_index, x1, y1, x2, y2], ...] like
        roi_size : tuple
            The size of each roi with regard to ROI-Wise operation
            each region proposal will be roi_size spatial shape.
        strides : tuple e.g. [4, 8, 16, 32]
            Define the gap that ori image and feature map have
        roi_mode : str, default is align
            ROI pooling mode. Currently support 'pool' and 'align'.
        Returns
        -------
        Pooled roi features aggregated according to its roi_level
        """
        max_stage = self._max_stage
        if self._use_p6:  # do not use p6 for RCNN
            max_stage = self._max_stage - 1
        _, x1, y1, x2, y2 = F.split(rpn_rois, axis=-1, num_outputs=5)
        h = y2 - y1 + 1
        w = x2 - x1 + 1
        roi_level = F.floor(4 + F.log2(F.sqrt(w * h) / 224.0 + eps))
        roi_level = F.squeeze(F.clip(roi_level, self._min_stage, max_stage))
        # [2,2,..,3,3,...,4,4,...,5,5,...] ``Prohibit swap order here``
        # roi_level_sorted_args = F.argsort(roi_level, is_ascend=True) 
        # roi_level = F.sort(roi_level, is_ascend=True)
        # rpn_rois = F.take(rpn_rois, roi_level_sorted_args, axis=0)
        pooled_roi_feats = []
        for i, l in enumerate(range(self._min_stage, max_stage + 1)):
            # Pool features with all rois first, and then set invalid pooled features to zero,
            # at last ele-wise add together to aggregate all features.
            if roi_mode == 'pool':
                pooled_feature = F.ROIPooling(features[i], rpn_rois, roi_size, 1. / strides[i])
            elif roi_mode == 'align':
                pooled_feature = F.contrib.ROIAlign(features[i], rpn_rois, roi_size,
                                                    1. / strides[i],
                                                    sample_ratio=2)
            else:
                raise ValueError("Invalid roi mode: {}".format(roi_mode))
            pooled_feature = F.where(roi_level == l, pooled_feature, F.zeros_like(pooled_feature))
            pooled_roi_feats.append(pooled_feature)
        # Ele-wise add to aggregate all pooled features
        pooled_roi_feats = F.ElementWiseSum(*pooled_roi_feats)
        # Sort all pooled features by asceding order
        # [2,2,..,3,3,...,4,4,...,5,5,...]
        # pooled_roi_feats = F.take(pooled_roi_feats, roi_level_sorted_args)
        # pooled roi feats (B*N, C, 7, 7), N = N2 + N3 + N4 + N5 = num_roi, C=256 in ori paper
        return pooled_roi_feats

    def hybrid_forward(self, F, x, gt_box=None):
        """Forward FPN network.
        The behavior during traing and inference is different.
        Parameters
        ----------
        x : mxnet.nd.NDArray or mxnet.symbol
          The network input tensor. The shape is the same with ori_img
        gt_box : type, only required during training
          The ground-truth bbox tensor with shape (1, N, 4).
        Returns
        -------
        (ids, scores, bboxes)
          During inference, returns final class id, confidence scores, bounding
          boxes.
        """

        def _split(x, axis, num_outputs, squeeze_axis):
            x = F.split(x, axis=axis, num_outputs=num_outputs, squeeze_axis=squeeze_axis)
            if isinstance(x, list):
                return x
            else:
                return [x]

        # Aggregate all pre proposals 
        anchors = []
        rpn_pre_nms_proposals = []
        raw_rpn_scores = []
        raw_rpn_boxes = []
        # Extract Features from [P2, P3, P4, P5, P6] 
        features = self.features(x)

        # Sample proposals for training
        # Output self._num_sample boxes 
        if autograd.is_training():
            rpn_scores, rpn_boxes, raw_rpn_scores, raw_rpn_boxes, anchors = \
                self.rpn(F.zeros_like(x), *features)
            rpn_boxes, samples, matches = self.sampler(rpn_boxes, rpn_scores, gt_box)
        else:
            _, rpn_boxes = self.rpn(F.zeros_like(x), *features)
        # Create batchid for roi
        # Note : the `ROIPooing` or `ROIAlign` Operation need the rois as a 2D
        # array of [[batch_inde x, x1, y1, x2, y2], ...]
        num_roi = self._num_sample if autograd.is_training() else self.rpn.get_test_post_nms()
        with autograd.pause():
            roi_batchid = F.arange(0, self._max_batch, repeat=num_roi)
            # remove batch dim because ROIPooling require 2d input
            rpn_rois = F.concat(*[roi_batchid.reshape((-1, 1)), rpn_boxes.reshape((-1, 4))], dim=-1)
            rpn_rois = F.stop_gradient(rpn_rois)

            # Note : the rpn_rois will be [[0, x1, y1, x2, y2], ..., [1, x1, y1, x2, y2], ...,
        # [B, x1, y1, x2, y2], ...] like, and same number for each batch_size

        # Get pyramid pooled ROI features and distribute those proposals 
        # to their appropriate FPN levels, An anchor at one FPN level may 
        # predict an RoI that will map to another level, hence the need 
        # to redistribute the proposals.
        pooled_feat = self._pyramid_roi_feats(F, features, rpn_rois, self._roi_size,
                                              self._pool_strides, roi_mode=self._roi_mode)

        # RCNN predictions
        """TODO(Angzz) Whether or not use top features in FPN"""
        # top_feat = self.top_features(pooled_feat)  # (B*N. 256, 7, 7)
        fc_pooled_feature = self.box_features(pooled_feat)  # 2fc layers
        cls_pred = self.class_predictor(fc_pooled_feature)  # (B*N, C+1) 
        box_pred = self.box_predictor(fc_pooled_feature)  # (B*N, C*4) 
        # cls_pred (B * N, C) -> (B, N, C)
        cls_pred = cls_pred.reshape((self._max_batch, num_roi, self.num_class + 1))
        # box_pred (B * N, C * 4) -> (B, N, C, 4)
        box_pred = box_pred.reshape((self._max_batch, num_roi, self.num_class, 4))

        # no need to convert bounding boxes in training, just return
        if autograd.is_training():
            return (cls_pred, box_pred, rpn_boxes, samples, matches,
                    raw_rpn_scores, raw_rpn_boxes, anchors)

        # cls_ids (B, N, C), scores (B, N, C)
        cls_ids, scores = self.cls_decoder(F.softmax(cls_pred, axis=-1))
        # cls_ids, scores (B, N, C) -> (B, C, N) -> (B, C, N, 1)
        cls_ids = cls_ids.transpose((0, 2, 1)).reshape((0, 0, 0, 1))
        scores = scores.transpose((0, 2, 1)).reshape((0, 0, 0, 1))
        # box_pred (B, N, C, 4) -> (B, C, N, 4)
        box_pred = box_pred.transpose((0, 2, 1, 3))

        # rpn_boxes (B, N, 4) -> B * (1, N, 4)
        rpn_boxes = _split(rpn_boxes, axis=0, num_outputs=self._max_batch, squeeze_axis=False)
        # cls_ids, scores (B, C, N, 1) -> B * (C, N, 1)
        cls_ids = _split(cls_ids, axis=0, num_outputs=self._max_batch, squeeze_axis=False)
        scores = _split(scores, axis=0, num_outputs=self._max_batch, squeeze_axis=False)
        # box_preds (B, C, N, 4) -> B * (C, N, 4)
        box_preds = _split(box_pred, axis=0, num_outputs=self._max_batch, squeeze_axis=False)

        # per batch predict, nms, each class has topk outputs
        results = []
        for rpn_box, cls_id, score, box_pred in zip(rpn_boxes, cls_ids, scores, box_preds):
            # box_pred (C, N, 4) rpn_box (1, N, 4) -> bbox (C, N, 4)
            bbox = self.box_decoder(box_pred, self.box_to_center(rpn_box))
            # res (C, N, 6)
            res = F.concat(*[cls_id, score, bbox], dim=-1)
            # res (C, self.nms_topk, 6)
            res = F.contrib.box_nms(
                res, overlap_thresh=self.nms_thresh, topk=self.nms_topk, valid_thresh=0.0001,
                id_index=0, score_index=1, coord_start=2, force_suppress=True)
            # res (C * self.nms_topk, 6)
            res = F.squeeze(res, axis=0).reshape((-3, 0))
            results.append(res)

        # result B * (C * topk, 6) -> (B, C * topk, 6)
        result = F.stack(*results, axis=0)
        # return final self.post_nms dets, usually 100 in coco
        if self.post_nms > 0:
            result = F.slice_axis(result, axis=1, begin=0, end=self.post_nms)
        ids = F.slice_axis(result, axis=-1, begin=0, end=1)
        scores = F.slice_axis(result, axis=-1, begin=1, end=2)
        bboxes = F.slice_axis(result, axis=-1, begin=2, end=6)
        if self._additional_output:
            return ids, scores, bboxes, features
        # ids (B, C * topk, 1) scores (B, C * topk, 1) bboxes (B, C * topk, 4)
        return ids, scores, bboxes


def get_faster_rcnn_fpn(name, dataset, pretrained=False, pretrained_base=True, ctx=mx.cpu(),
                        root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    r"""Utility function to return fpn networks.
    Parameters
    ----------
    name : str
        Model name.
    dataset : str
        The name of dataset.
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
    mxnet.gluon.HybridBlock
        The FPN network.
    """
    pretrained_base = False if pretrained else pretrained_base
    net = FPN(pretrained_base=pretrained_base, **kwargs)
    if pretrained:
        from ..model_store import get_model_file
        full_name = '_'.join(('fpn', name, dataset))
        net.load_parameters(get_model_file(full_name, root=root), ctx=ctx)
    return net


def faster_rcnn_fpn_resnet50_v1b_voc(pretrained=False, pretrained_base=True, **kwargs):
    r"""FPN model from the paper
    "Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan,
    Serge Belongie (2017). Feature Pyramid Networks for Object Detection"
    Parameters
    ----------
    pretrained : bool, optional, default is False
        Load pretrained weights.
    pretrained_base : bool, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `Ture`, this has no effect.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    Examples
    --------
    >>> model = get_faster_rcnn_fpn_resnet50_v1b_voc(pretrained=True)
    >>> print(model)
    """
    from ..resnetv1b import resnet50_v1b
    from ...data import VOCDetection
    classes = VOCDetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet50_v1b(pretrained=pretrained_base, dilated=False, use_global_stats=True)
    train_patterns = '|'.join(['.*dense', '.*rpn', '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv'])
    # Note : rpn_train_pre_nms means per level, e.g. 2000 X 4 lvl.
    # And the same with rpn_test_post_nms, e.g. 1000 X 4 lvl.
    # However, rpn_train_post_nms and rpn_test_post_nms means all 
    # fpn levels 
    return get_faster_rcnn_fpn(
        name='resnet50_v1b', dataset='voc', pretrained=pretrained, network=base_network,
        features=['layers1_relu8_fwd', 'layers2_relu11_fwd', 'layers3_relu17_fwd',
                  'layers4_relu8_fwd'],
        num_filters=[256, 256, 256, 256], use_1x1=True, use_upsample=True,
        use_elewadd=True, use_p6=True, no_bias=False, top_features=None,
        classes=classes, short=600, max_size=1000, min_stage=2, max_stage=6,
        train_patterns=train_patterns, nms_thresh=0.3, nms_topk=400, post_nms=-1,
        roi_mode='align', roi_size=(14, 14), strides=(4, 8, 16, 32, 64), clip=None,
        rpn_channel=1024, base_size=16, scales=(2, 4, 8, 16, 32), ratios=(0.5, 1, 2),
        alloc_size=(512, 512), rpn_nms_thresh=0.7, rpn_train_pre_nms=12000,
        rpn_train_post_nms=2000, rpn_test_pre_nms=6000, rpn_test_post_nms=300,
        rpn_min_size=16, num_sample=128, pos_iou_thresh=0.5, pos_ratio=0.25,
        max_num_gt=100, **kwargs)


def faster_rcnn_fpn_resnet50_v1b_coco(pretrained=False, pretrained_base=True, **kwargs):
    r"""FPN model from the paper
    "Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan,
    Serge Belongie (2017). Feature Pyramid Networks for Object Detection"
    Parameters
    ---------- 
    pretrained : bool, optional, default is False
        Load pretrained weights.
    pretrained_base : bool, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `Ture`, this has no effect.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    Examples
    --------
    >>> model = get_faster_rcnn_fpn_resnet50_v1b_coco(pretrained=True)
    >>> print(model)
    """
    from ..resnetv1b import resnet50_v1b
    from ...data import COCODetection
    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet50_v1b(pretrained=pretrained_base, dilated=False, use_global_stats=True)
    train_patterns = '|'.join(['.*dense', '.*rpn', '.*down(2|3|4)_conv', '.*layers(2|3|4)_conv'])
    # Note : rpn_train_pre_nms means per level, e.g. 2000 X 4 lvl.
    # And the same with rpn_test_post_nms, e.g. 1000 X 4 lvl.
    # However, rpn_train_post_nms and rpn_test_post_nms means all 
    # fpn levels 
    return get_faster_rcnn_fpn(
        name='resnet50_v1b', dataset='coco', pretrained=pretrained, network=base_network,
        features=['layers1_relu8_fwd', 'layers2_relu11_fwd', 'layers3_relu17_fwd',
                  'layers4_relu8_fwd'],
        num_filters=[256, 256, 256, 256], use_1x1=True, use_upsample=True,
        use_elewadd=True, use_p6=True, no_bias=False, top_features=None,
        classes=classes, short=800, max_size=1333, min_stage=2, max_stage=6,
        train_patterns=train_patterns, nms_thresh=0.5, nms_topk=-1, post_nms=-1,
        roi_mode='align', roi_size=(14, 14), strides=(4, 8, 16, 32, 64), clip=4.42,
        rpn_channel=1024, base_size=16, scales=(2, 4, 8, 16, 32), ratios=(0.5, 1, 2),
        alloc_size=(384, 384), rpn_nms_thresh=0.7, rpn_train_pre_nms=12000,
        rpn_train_post_nms=2000, rpn_test_pre_nms=6000, rpn_test_post_nms=1000,
        rpn_min_size=0, num_sample=512, pos_iou_thresh=0.5, pos_ratio=0.25,
        max_num_gt=100, **kwargs)
