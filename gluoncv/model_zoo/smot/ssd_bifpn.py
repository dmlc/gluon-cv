"""
MXNet implementation of tracktor in SMOT: Single-Shot Multi Object Tracking
https://arxiv.org/abs/2010.16031
"""
# pylint: disable=unused-argument,unused-variable,ungrouped-imports
from __future__ import absolute_import

import os
import mxnet as mx
from mxnet.gluon import nn, HybridBlock
from gluoncv.nn.feature import FeatureExpander
from gluoncv.nn.predictor import ConvPredictor
from gluoncv.nn.coder import MultiPerClassDecoder, NormalizedBoxCenterDecoder
from mxnet.gluon.contrib.nn import SyncBatchNorm

from .anchor import SSDAnchorGenerator
from .feature_bifpn import FPNFeatureExpander
from .decoders import NormalizedLandmarkCenterDecoder, GeneralNormalizedKeyPointsDecoder


__all__ = ['SSD', 'get_ssd']


class SSDDetectorHead(HybridBlock):
    """SSD detector head"""
    def __init__(self, num_layers, base_size, sizes, ratios,
                 steps, classes,
                 stds=(0.1, 0.1, 0.2, 0.2), nms_thresh=0.3, nms_topk=10000, post_nms=3000,
                 anchor_alloc_size=640, is_multitask=False, use_pose=False,
                 use_keypoints=False, num_keypoints=1,
                 use_embedding=False, embedding_dim=128, return_intermediate_features=False,
                 **kwargs):
        super(SSDDetectorHead, self).__init__(**kwargs)

        self._num_layers = num_layers
        self.classes = classes
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms
        self._use_pose = use_pose
        if self._use_pose:
            self._is_multitask = True
        else:
            self._is_multitask = is_multitask

        self._use_keypoints = use_keypoints
        self._keypoint_size = num_keypoints * 2

        self._use_emebdding = use_embedding
        self._embedding_dim = embedding_dim

        self._return_int_feat = return_intermediate_features

        with self.name_scope():
            self.class_predictors = nn.HybridSequential()
            self.box_predictors = nn.HybridSequential()
            self.anchor_generators = nn.HybridSequential()
            if self._is_multitask:
                self.landmark_predictors = nn.HybridSequential()
            if self._use_pose:
                self.pose_predictors = nn.HybridSequential()
            if self._use_keypoints:
                self.keypoint_predictors = nn.HybridSequential()
            if self._use_emebdding:
                self.embedding_predictors = nn.HybridSequential()
            asz = anchor_alloc_size
            im_size = (base_size, base_size)
            for i, s, r, st in zip(range(num_layers), sizes, ratios, steps):
                anchor_generator = SSDAnchorGenerator(i, im_size, s, r, st, (asz, asz))
                self.anchor_generators.add(anchor_generator)
                asz = max(asz // 2, 16)  # pre-compute larger than 16x16 anchor map
                num_anchors = anchor_generator.num_depth
                self.class_predictors.add(ConvPredictor(num_anchors * (len(self.classes) + 1)))
                self.box_predictors.add(ConvPredictor(num_anchors * 4))
                if self._is_multitask:
                    self.landmark_predictors.add(ConvPredictor(num_anchors * 10))
                if self._use_pose:
                    self.pose_predictors.add(ConvPredictor(num_anchors * 6))
                if self._use_keypoints:
                    self.keypoint_predictors.add(ConvPredictor(num_anchors * self._keypoint_size))
                if self._use_emebdding:
                    local_seq = nn.HybridSequential()
                    local_seq.add(ConvPredictor(num_anchors * self._embedding_dim * len(self.classes)))
                    local_seq.add(nn.BatchNorm(prefix='embedding_norm_{}_'.format(i)))
                    local_seq.add(nn.LeakyReLU(alpha=0.25))
                    local_seq.add(nn.Conv2D(num_anchors * self._embedding_dim * len(self.classes), (1, 1),
                                            weight_initializer=mx.init.Xavier(magnitude=2),
                                            bias_initializer='zeros', groups=num_anchors * len(self.classes)))
                    self.embedding_predictors.add(local_seq)

            self.bbox_decoder = NormalizedBoxCenterDecoder(stds)
            self.cls_decoder = MultiPerClassDecoder(len(self.classes) + 1, thresh=0.01)
            if self._is_multitask:
                self.landmark_decoder = NormalizedLandmarkCenterDecoder(stds)
            if self._use_keypoints:
                self.keypoint_decoder = GeneralNormalizedKeyPointsDecoder(1)

    def hybrid_forward(self, F, *inputs, **kwargs):
        """forward of both detection and tracking"""
        offset = 0
        features = inputs[offset:offset + self._num_layers]
        offset += self._num_layers

        raw_features = inputs[offset:offset + self._num_layers]
        offset += self._num_layers

        if len(inputs) == 2 * self._num_layers + 3:
            # tracking mode
            tracking_inputs = inputs[offset + 1: offset + 3]
        else:
            tracking_inputs = None

        cls_preds = [F.flatten(F.transpose(cp(feat), (0, 2, 3, 1)))
                     for feat, cp in zip(features, self.class_predictors)]
        raw_box_preds = [bp(feat) for feat, bp in zip(features, self.box_predictors)]
        box_preds = [F.flatten(F.transpose(pred, (0, 2, 3, 1)))
                     for pred in raw_box_preds]
        if self._is_multitask:
            landmark_preds = [F.flatten(F.transpose(lp(feat), (0, 2, 3, 1)))
                              for feat, lp in zip(features, self.landmark_predictors)]
        if self._use_pose:
            pose_preds = [F.flatten(F.transpose(lp(feat), (0, 2, 3, 1)))
                          for feat, lp in zip(features, self.pose_predictors)]

        if self._use_keypoints:
            keypoint_preds = [F.flatten(F.transpose(lp(feat), (0, 2, 3, 1)))
                              for feat, lp in zip(features, self.keypoint_predictors)]
        if self._use_emebdding:
            embedding_preds = [F.flatten(F.transpose(lp(feat), (0, 2, 3, 1)))
                               for feat, lp in zip(features, self.embedding_predictors)]

        anchors = [F.reshape(ag(feat), shape=(1, -1))
                   for feat, ag in zip(features, self.anchor_generators)]
        cls_preds = F.concat(*cls_preds, dim=1).reshape((0, -1, self.num_classes + 1))
        box_preds = F.concat(*box_preds, dim=1).reshape((0, -1, 4))
        if self._is_multitask:
            landmark_preds = F.concat(*landmark_preds, dim=1).reshape((0, -1, 10))
        if self._use_pose:
            pose_preds = F.concat(*pose_preds, dim=1).reshape((0, -1, 6))
        if self._use_keypoints:
            keypoint_preds = F.concat(*keypoint_preds, dim=1).reshape((0, -1, self._keypoint_size))
        if self._use_emebdding:
            embedding_preds = F.concat(*embedding_preds, dim=1).reshape((0, -1, self._embedding_dim * self.num_classes))

        anchors = F.concat(*anchors, dim=1).reshape((1, -1, 4))

        bboxes = self.bbox_decoder(box_preds, anchors)
        cls_ids, scores = self.cls_decoder(F.softmax(cls_preds, axis=-1))
        if self._is_multitask:
            landmarks = self.landmark_decoder(landmark_preds, anchors)
        if self._use_keypoints:
            keypoints = self.keypoint_decoder(keypoint_preds, anchors)

        anchor_indices = F.contrib.arange_like(cls_ids.slice_axis(axis=-1, begin=0, end=1))

        results = []
        for i in range(self.num_classes):
            cls_id = cls_ids.slice_axis(axis=-1, begin=i, end=i + 1)
            score = scores.slice_axis(axis=-1, begin=i, end=i + 1)
            cls_outputs = [cls_id, score, bboxes]
            # per class results
            if self._is_multitask:
                cls_outputs.append(landmarks)

            if self._use_pose:
                cls_outputs.append(pose_preds)

            if self._use_keypoints:
                cls_outputs.append(keypoints)

            if self._use_emebdding:
                cls_outputs.append(embedding_preds.slice_axis(axis=-1,
                                                              begin=i * self._embedding_dim,
                                                              end=(i + 1) * self._embedding_dim))

            cls_outputs.append(anchor_indices)
            # per class results
            per_result = F.concat(*cls_outputs, dim=-1)
            results.append(per_result)

        result = F.concat(*results, dim=1)

        if tracking_inputs is not None:
            tracking_anchor_indices, tracking_anchor_weights = tracking_inputs

            # pickup the tracking results before NMS
            tracking_results = F.take(result, tracking_anchor_indices, axis=1)
            tracking_results = F.squeeze(tracking_results, axis=0)
            tracking_weights = F.expand_dims(tracking_anchor_weights, axis=2)

            tracking_results = F.broadcast_mul(tracking_results, tracking_weights)
            tracking_results = F.sum(tracking_results, axis=1)
        else:
            tracking_results = None

        if self.nms_thresh > 0 and self.nms_thresh < 1:
            result = F.contrib.box_nms(
                result, overlap_thresh=self.nms_thresh, topk=self.nms_topk, valid_thresh=0.01,
                id_index=0, score_index=1, coord_start=2, force_suppress=False)
            if self.post_nms > 0:
                result = result.slice_axis(axis=1, begin=0, end=self.post_nms)
        ids = F.slice_axis(result, axis=2, begin=0, end=1)
        scores = F.slice_axis(result, axis=2, begin=1, end=2)
        bboxes = F.slice_axis(result, axis=2, begin=2, end=6)

        outputs = [ids, scores, bboxes]

        offset = 6
        if self._is_multitask:
            landmarks = F.slice_axis(result, axis=2, begin=offset, end=offset + 10)
            offset += 10
            outputs.append(landmarks)
        if self._use_pose:
            poses = F.slice_axis(result, axis=2, begin=offset, end=offset + 6)
            offset += 6
            outputs.append(poses)

        if self._use_keypoints:
            keypoints = F.slice_axis(result, axis=2, begin=offset, end=offset + self._keypoint_size)
            offset += self._keypoint_size
            outputs.append(keypoints)

        if self._use_emebdding:
            embeddings = F.slice_axis(result, axis=2, begin=offset, end=offset + self._embedding_dim)
            offset += self._embedding_dim
            outputs.append(embeddings)

        anchor_indices = F.slice_axis(result, axis=2, begin=offset, end=offset + 1)
        offset += 1
        outputs.extend([anchor_indices])

        if tracking_results is not None:
            outputs.append(tracking_results)

        outputs.append(anchors)

        return tuple(outputs)

    @property
    def num_classes(self):
        """Return number of foreground classes.

        Returns
        -------
        int
            Number of foreground classes

        """
        return len(self.classes)

    def set_nms(self, nms_thresh=0.45, nms_topk=10000, post_nms=2000):
        """Set non-maximum suppression parameters.

        Parameters
        ----------
        nms_thresh : float, default is 0.45.
            Non-maximum suppression threshold. You can specify < 0 or > 1 to disable NMS.
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


class SSD(HybridBlock):
    """Single-shot Object Detection Network: https://arxiv.org/abs/1512.02325.

    Parameters
    ----------
    network : string or None
        Name of the base network, if `None` is used, will instantiate the
        base network from `features` directly instead of composing.
    base_size : int
        Base input size, it is speficied so SSD can support dynamic input shapes.
    features : list of str or mxnet.gluon.HybridBlock
        Intermediate features to be extracted or a network with multi-output.
        If `network` is `None`, `features` is expected to be a multi-output network.
    num_filters : list of int
        Number of channels for the appended layers, ignored if `network`is `None`.
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
        Names of all categories.
    use_1x1_transition : bool
        Whether to use 1x1 convolution as transition layer between attached layers,
        it is effective reducing model capacity.
    use_bn : bool
        Whether to use BatchNorm layer after each attached convolutional layer.
    reduce_ratio : float
        Channel reduce ratio (0, 1) of the transition layer.
    min_depth : int
        Minimum channels for the transition layers.
    global_pool : bool
        Whether to attach a global average pooling layer as the last output layer.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    stds : tuple of float, default is (0.1, 0.1, 0.2, 0.2)
        Std values to be divided/multiplied to box encoded values.
    nms_thresh : float, default is 0.45.
        Non-maximum suppression threshold. You can specify < 0 or > 1 to disable NMS.
    nms_topk : int, default is 400
        Apply NMS to top k detection results, use -1 to disable so that every Detection
         result is used in NMS.
    post_nms : int, default is 100
        Only return top `post_nms` detection results, the rest is discarded. The number is
        based on COCO dataset which has maximum 100 objects per image. You can adjust this
        number if expecting more objects. You can use -1 to return all detections.
    anchor_alloc_size : tuple of int, default is (128, 128)
        For advanced users. Define `anchor_alloc_size` to generate large enough anchor
        maps, which will later saved in parameters. During inference, we support arbitrary
        input image by cropping corresponding area of the anchor map. This allow us
        to export to symbol so we can run it in c++, scalar, etc.
    ctx : mx.Context
        Network context.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
        This will only apply to base networks that has `norm_layer` specified, will ignore if the
        base network (e.g. VGG) don't accept this argument.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.

    """
    def __init__(self, network, base_size, features, ssd_filters, sizes, ratios,
                 steps, classes, use_1x1_transition=True, use_bn=True,
                 reduce_ratio=1.0, min_depth=128, global_pool=False, pretrained=False,
                 stds=(0.1, 0.1, 0.2, 0.2), nms_thresh=0.3, nms_topk=10000, post_nms=3000,
                 anchor_alloc_size=640, ctx=mx.cpu(),
                 norm_layer=SyncBatchNorm, norm_kwargs=None, is_fpn=False,
                 fpn_filters=256, is_multitask=False, use_pose=False,
                 use_keypoints=False, num_keypoints=1,
                 use_embedding=False, embedding_dim=128, return_intermediate_features=False,
                 use_mish=False,
                 **kwargs):
        super(SSD, self).__init__(**kwargs)
        if norm_kwargs is None:
            norm_kwargs = {}
        if network is None:
            num_layers = len(ratios)
        else:
            num_layers = len(features) + len(ssd_filters) + int(global_pool)
        assert len(sizes) == len(ratios)
        assert isinstance(ratios, list), "Must provide ratios as list or list of list"
        if not isinstance(ratios[0], (tuple, list)):
            ratios = ratios * num_layers  # propagate to all layers if use same ratio
        assert num_layers == len(sizes) == len(ratios), \
            "Mismatched (number of layers) vs (sizes) vs (ratios): {}, {}, {}".format(
                num_layers, len(sizes), len(ratios))
        assert num_layers > 0, "SSD require at least one layer, suggest multiple."

        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms

        self._two_phase_run = return_intermediate_features


        with self.name_scope():
            if network is None:
                # use fine-grained manually designed block as features
                try:
                    self.features = features(pretrained=pretrained, ctx=ctx,
                                             norm_layer=norm_layer, norm_kwargs=norm_kwargs)
                except TypeError:
                    self.features = features(pretrained=pretrained, ctx=ctx)
            else:
                if is_fpn:
                    fpn_filters_list = [fpn_filters for i in range(num_layers)]
                    try:
                        self.features = FPNFeatureExpander(
                            network=network, outputs=features, ssd_filters=ssd_filters, fpn_filters=fpn_filters_list,
                            use_bn=use_bn, reduce_ratio=reduce_ratio, min_depth=min_depth,
                            global_pool=global_pool, pretrained=pretrained, ctx=ctx,
                            norm_layer=norm_layer, norm_kwargs=norm_kwargs, use_mish=use_mish)

                    except TypeError:
                        self.features = FPNFeatureExpander(
                            network=network, outputs=features, ssd_filters=ssd_filters, fpn_filters=fpn_filters_list,
                            use_bn=use_bn, reduce_ratio=reduce_ratio, min_depth=min_depth,
                            global_pool=global_pool, pretrained=pretrained, ctx=ctx, use_mish=use_mish)
                else:
                    try:
                        self.features = FeatureExpander(
                            network=network, outputs=features, num_filters=ssd_filters,
                            use_1x1_transition=use_1x1_transition,
                            use_bn=use_bn, reduce_ratio=reduce_ratio, min_depth=min_depth,
                            global_pool=global_pool, pretrained=pretrained, ctx=ctx,
                            norm_layer=norm_layer, norm_kwargs=norm_kwargs)
                    except TypeError:
                        self.features = FeatureExpander(
                            network=network, outputs=features, num_filters=ssd_filters,
                            use_1x1_transition=use_1x1_transition,
                            use_bn=use_bn, reduce_ratio=reduce_ratio, min_depth=min_depth,
                            global_pool=global_pool, pretrained=pretrained, ctx=ctx)

            self.detector_head = SSDDetectorHead(num_layers, base_size, sizes, ratios, steps, classes,
                                                 stds, nms_thresh, nms_topk, post_nms,
                                                 anchor_alloc_size, is_multitask, use_pose,
                                                 use_keypoints, num_keypoints, use_embedding, embedding_dim,
                                                 return_intermediate_features, **kwargs)

    @property
    def num_classes(self):
        """Return number of foreground classes.

        Returns
        -------
        int
            Number of foreground classes

        """
        return self.detector_head.num_classes

    def set_nms(self, nms_thresh=0.45, nms_topk=10000, post_nms=2000):
        """Set non-maximum suppression parameters.

        Parameters
        ----------
        nms_thresh : float, default is 0.45.
            Non-maximum suppression threshold. You can specify < 0 or > 1 to disable NMS.
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
        self.detector_head.set_nms(nms_thresh, nms_topk, post_nms)

    @property
    def anchor_generators(self):
        return self.detector_head.anchor_generators

    def forward_features(self, F, x):
        features, _ = self.features(x)
        anchors = [F.reshape(ag(feat), shape=(1, -1))
                   for feat, ag in zip(features, self.anchor_generators)]
        anchors = F.concat(*anchors, dim=1).reshape((1, -1, 4))
        raw_box_preds = [bp(feat) for feat, bp in zip(features, self.detector_head.box_predictors)]
        return features, anchors, raw_box_preds

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, x, *args):
        """Hybrid forward"""
        if not self._two_phase_run:
            features, raw_features = self.features(x)
        else:
            features = raw_features = x

        head_input = tuple(features) + tuple(raw_features) + args

        return self.detector_head(*head_input)


def get_ssd(name, base_size, features, ssd_filters, sizes, ratios, steps, classes,
            dataset, pretrained=False, pretrained_dir='', pretrained_base=True, ctx=mx.cpu(),
            root=os.path.join('~', '.mxnet', 'models'), is_fpn=False, fpn_filters=256,
            is_multitask=False, use_pose=False, use_keypoints=False, num_keypoints=1,
            use_embedding=False, embedding_dim=128, return_features=False,
            use_mish=False, **kwargs):
    """Get SSD models.

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
        If `name` is `None`, `features` must be a `HybridBlock` which generate multiple
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
        different datasets are going to be very different.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    ctx : mxnet.Context
        Context such as mx.cpu(), mx.gpu(0).
    root : str
        Model weights storing path.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.

    Returns
    -------
    HybridBlock
        A SSD detection network.
    """
    pretrained_base = False if pretrained else pretrained_base
    base_name = None if callable(features) else name
    net = SSD(base_name, base_size, features, ssd_filters, sizes, ratios, steps,
              pretrained=pretrained_base, classes=classes, ctx=ctx, is_fpn=is_fpn, fpn_filters=fpn_filters,
              is_multitask=is_multitask, use_pose=use_pose, use_keypoints=use_keypoints, num_keypoints=num_keypoints,
              use_embedding=use_embedding, embedding_dim=embedding_dim, return_intermediate_features=return_features,
              use_mish=use_mish,
              **kwargs)
    if pretrained:
        net.load_parameters(pretrained_dir, ctx=ctx)
    return net
