"""Single-shot Multi-box Detector."""
from __future__ import absolute_import

import os
import warnings
import mxnet as mx
from mxnet import autograd
from mxnet.gluon import nn
from mxnet.gluon import HybridBlock
from ...nn.feature import FeatureExpander
from .anchor import SSDAnchorGenerator
from ...nn.predictor import ConvPredictor
from ...nn.coder import MultiPerClassDecoder, NormalizedBoxCenterDecoder
from .vgg_atrous import vgg16_atrous_300, vgg16_atrous_512

__all__ = ['SSD', 'get_ssd', 'custom_ssd']


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
    root : str
        The root path for model storage, default is '~/.mxnet/models'
    minimal_opset : bool
        We sometimes add special operators to accelerate training/inference, however, for exporting
        to third party compilers we want to utilize most widely used operators.
        If `minimal_opset` is `True`, the network will use a minimal set of operators good
        for e.g., `TVM`.
    predictor_kernel: tuple of int. default is (3,3)
        Dimension of predictor kernel
    predictor_pad: tuple of int. default is (1,1)
        Padding of the predictor kenrel conv.
    anchor_generator: default is SSDAnchorGenerator
        Anchor Generator to be used. The default it SSDAnchorGenerator corresponding
        to SSD published article. This argument can be used for other custom
        anchor generators. Like LiteAnchorGenerator.

    """
    def __init__(self, network, base_size, features, num_filters, sizes, ratios,
                 steps, classes, use_1x1_transition=True, use_bn=True,
                 reduce_ratio=1.0, min_depth=128, global_pool=False, pretrained=False,
                 stds=(0.1, 0.1, 0.2, 0.2), nms_thresh=0.45, nms_topk=400, post_nms=100,
                 anchor_alloc_size=128, ctx=mx.cpu(),
                 norm_layer=nn.BatchNorm, norm_kwargs=None,
                 root=os.path.join('~', '.mxnet', 'models'), minimal_opset=False,
                 predictors_kernel=(3, 3), predictors_pad=(1, 1),
                 anchor_generator=SSDAnchorGenerator, **kwargs):
        super(SSD, self).__init__(**kwargs)
        if norm_kwargs is None:
            norm_kwargs = {}
        if network is None:
            num_layers = len(ratios)
        else:
            num_layers = len(features) + len(num_filters) + int(global_pool)
        assert len(sizes) == num_layers + 1
        sizes = list(zip(sizes[:-1], sizes[1:]))
        assert isinstance(ratios, (tuple, list)), "Must provide ratios as tuple or list"
        if not isinstance(ratios[0], (tuple, list)):
            ratios = ratios * num_layers  # propagate to all layers if use same ratio
        assert num_layers == len(sizes) == len(ratios), \
            "Mismatched (number of layers) vs (sizes) vs (ratios): {}, {}, {}".format(
                num_layers, len(sizes), len(ratios))
        assert num_layers > 0, "SSD require at least one layer, suggest multiple."
        self._num_layers = num_layers
        self.classes = classes
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms

        with self.name_scope():
            if network is None:
                # use fine-grained manually designed block as features
                try:
                    self.features = features(pretrained=pretrained, ctx=ctx, root=root,
                                             norm_layer=norm_layer, norm_kwargs=norm_kwargs)
                except TypeError:
                    self.features = features(pretrained=pretrained, ctx=ctx, root=root)
            else:
                try:
                    self.features = FeatureExpander(
                        network=network, outputs=features, num_filters=num_filters,
                        use_1x1_transition=use_1x1_transition,
                        use_bn=use_bn, reduce_ratio=reduce_ratio, min_depth=min_depth,
                        global_pool=global_pool, pretrained=pretrained, ctx=ctx,
                        norm_layer=norm_layer, norm_kwargs=norm_kwargs, root=root)
                except TypeError:
                    self.features = FeatureExpander(
                        network=network, outputs=features, num_filters=num_filters,
                        use_1x1_transition=use_1x1_transition,
                        use_bn=use_bn, reduce_ratio=reduce_ratio, min_depth=min_depth,
                        global_pool=global_pool, pretrained=pretrained, ctx=ctx, root=root)
            self.class_predictors = nn.HybridSequential()
            self.box_predictors = nn.HybridSequential()
            self.anchor_generators = nn.HybridSequential()
            asz = anchor_alloc_size
            im_size = (base_size, base_size)
            for i, s, r, st in zip(range(num_layers), sizes, ratios, steps):
                branch_anchor_generator = anchor_generator(i, im_size, s, r, st, (asz, asz))
                self.anchor_generators.add(branch_anchor_generator)
                asz = max(asz // 2, 16)  # pre-compute larger than 16x16 anchor map
                num_anchors = branch_anchor_generator.num_depth
                self.class_predictors.add(ConvPredictor(num_anchors * (len(self.classes) + 1),
                                                        kernel=predictors_kernel,
                                                        pad=predictors_pad))
                self.box_predictors.add(ConvPredictor(num_anchors * 4,
                                                      kernel=predictors_kernel,
                                                      pad=predictors_pad))
            self.bbox_decoder = NormalizedBoxCenterDecoder(stds, minimal_opset=minimal_opset)
            self.cls_decoder = MultiPerClassDecoder(len(self.classes) + 1, thresh=0.01)

    @property
    def num_classes(self):
        """Return number of foreground classes.

        Returns
        -------
        int
            Number of foreground classes

        """
        return len(self.classes)

    def set_nms(self, nms_thresh=0.45, nms_topk=400, post_nms=100):
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

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, x):
        """Hybrid forward"""
        features = self.features(x)
        cls_preds = [F.flatten(F.transpose(cp(feat), (0, 2, 3, 1)))
                     for feat, cp in zip(features, self.class_predictors)]
        box_preds = [F.flatten(F.transpose(bp(feat), (0, 2, 3, 1)))
                     for feat, bp in zip(features, self.box_predictors)]
        anchors = [F.reshape(ag(feat), shape=(1, -1))
                   for feat, ag in zip(features, self.anchor_generators)]
        cls_preds = F.concat(*cls_preds, dim=1).reshape((0, -1, self.num_classes + 1))
        box_preds = F.concat(*box_preds, dim=1).reshape((0, -1, 4))
        anchors = F.concat(*anchors, dim=1).reshape((1, -1, 4))
        if autograd.is_training():
            return [cls_preds, box_preds, anchors]
        bboxes = self.bbox_decoder(box_preds, anchors)
        cls_ids, scores = self.cls_decoder(F.softmax(cls_preds, axis=-1))
        results = []
        for i in range(self.num_classes):
            cls_id = cls_ids.slice_axis(axis=-1, begin=i, end=i+1)
            score = scores.slice_axis(axis=-1, begin=i, end=i+1)
            # per class results
            per_result = F.concat(*[cls_id, score, bboxes], dim=-1)
            results.append(per_result)
        result = F.concat(*results, dim=1)
        if self.nms_thresh > 0 and self.nms_thresh < 1:
            result = F.contrib.box_nms(
                result, overlap_thresh=self.nms_thresh, topk=self.nms_topk, valid_thresh=0.01,
                id_index=0, score_index=1, coord_start=2, force_suppress=False)
            if self.post_nms > 0:
                result = result.slice_axis(axis=1, begin=0, end=self.post_nms)
        ids = F.slice_axis(result, axis=2, begin=0, end=1)
        scores = F.slice_axis(result, axis=2, begin=1, end=2)
        bboxes = F.slice_axis(result, axis=2, begin=2, end=6)
        return ids, scores, bboxes

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
        >>> net = gluoncv.model_zoo.get_model('ssd_512_resnet50_v1_voc', pretrained=True)
        >>> # use direct name to name mapping to reuse weights
        >>> net.reset_class(classes=['person'], reuse_weights={'person':'person'})
        >>> # or use interger mapping, person is the 14th category in VOC
        >>> net.reset_class(classes=['person'], reuse_weights={0:14})
        >>> # you can even mix them
        >>> net.reset_class(classes=['person'], reuse_weights={'person':14})
        >>> # or use a list of string if class name don't change
        >>> net.reset_class(classes=['person'], reuse_weights=['person'])

        """
        self._clear_cached_op()
        old_classes = self.classes
        self.classes = classes
        # trying to reuse weights by mapping old and new classes
        if isinstance(reuse_weights, (dict, list)):
            if isinstance(reuse_weights, dict):
                # trying to replace str with indices
                new_keys = []
                new_vals = []
                for k, v in reuse_weights.items():
                    if isinstance(v, str):
                        try:
                            new_vals.append(old_classes.index(v))  # raise ValueError if not found
                        except ValueError:
                            raise ValueError(
                                "{} not found in old class names {}".format(v, old_classes))
                    else:
                        if v < 0 or v >= len(old_classes):
                            raise ValueError(
                                "Index {} out of bounds for old class names".format(v))
                        new_vals.append(v)
                    if isinstance(k, str):
                        try:
                            new_keys.append(self.classes.index(k))  # raise ValueError if not found
                        except ValueError:
                            raise ValueError(
                                "{} not found in new class names {}".format(k, self.classes))
                    else:
                        if k < 0 or k >= len(self.classes):
                            raise ValueError(
                                "Index {} out of bounds for new class names".format(k))
                        new_keys.append(k)
                reuse_weights = dict(zip(new_keys, new_vals))
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
        # replace class predictors
        with self.name_scope():
            class_predictors = nn.HybridSequential(prefix=self.class_predictors.prefix)
            for i, ag in zip(range(len(self.class_predictors)), self.anchor_generators):
                # Re-use the same prefix and ctx_list as used by the current ConvPredictor
                prefix = self.class_predictors[i].prefix
                old_pred = self.class_predictors[i].predictor
                ctx = list(old_pred.params.values())[0].list_ctx()
                # to avoid deferred init, number of in_channels must be defined
                in_channels = list(old_pred.params.values())[0].shape[1]
                new_cp = ConvPredictor(ag.num_depth * (self.num_classes + 1),
                                       in_channels=in_channels, prefix=prefix)
                new_cp.collect_params().initialize(ctx=ctx)
                if reuse_weights:
                    assert isinstance(reuse_weights, dict)
                    for old_params, new_params in zip(old_pred.params.values(),
                                                      new_cp.predictor.params.values()):
                        old_data = old_params.data()
                        new_data = new_params.data()

                        for k, v in reuse_weights.items():
                            if k >= len(self.classes) or v >= len(old_classes):
                                warnings.warn("reuse mapping {}/{} -> {}/{} out of range".format(
                                    k, self.classes, v, old_classes))
                                continue
                            # always increment k and v (background is always the 0th)
                            new_data[k+1::len(self.classes)+1] = old_data[v+1::len(old_classes)+1]
                        # reuse background weights as well
                        new_data[0::len(self.classes)+1] = old_data[0::len(old_classes)+1]
                        # set data to new conv layers
                        new_params.set_data(new_data)
                class_predictors.add(new_cp)
            self.class_predictors = class_predictors
            self.cls_decoder = MultiPerClassDecoder(len(self.classes) + 1, thresh=0.01)

def get_ssd(name, base_size, features, filters, sizes, ratios, steps, classes,
            dataset, pretrained=False, pretrained_base=True, ctx=mx.cpu(),
            root=os.path.join('~', '.mxnet', 'models'),
            anchor_generator=SSDAnchorGenerator, **kwargs):
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
    net = SSD(base_name, base_size, features, filters, sizes, ratios, steps,
              pretrained=pretrained_base, classes=classes, ctx=ctx, root=root,
              minimal_opset=pretrained, anchor_generator=anchor_generator, **kwargs)
    if pretrained:
        from ..model_store import get_model_file
        full_name = '_'.join(('ssd', str(base_size), name, dataset))
        net.load_parameters(get_model_file(full_name, tag=pretrained, root=root), ctx=ctx)
    return net

def custom_ssd(base_network_name, base_size, filters, sizes, ratios, steps,
               classes, dataset, pretrained_base, **kwargs):
    """Custom SSD models.
    """
    if base_network_name == 'vgg16_atrous' and base_size == 300:
        features = vgg16_atrous_300
    elif base_network_name == 'vgg16_atrous' and base_size == 512:
        features = vgg16_atrous_512
    elif base_network_name == 'resnet18_v1' and base_size == 512:
        features = ['stage3_activation1', 'stage4_activation1']
    elif base_network_name == 'resnet50_v1' and base_size == 512:
        features = ['stage3_activation5', 'stage4_activation2']
    elif base_network_name == 'resnet101_v2' and base_size == 512:
        features = ['stage3_activation22', 'stage4_activation2']
    elif base_network_name == 'resnet152_v2' and base_size == 512:
        features = ['stage2_activation7', 'stage3_activation35', 'stage4_activation2']
    elif base_network_name == 'mobilenet1.0' and base_size == 512:
        features = ['relu22_fwd', 'relu26_fwd']
    elif base_network_name == 'mobilenet1.0' and base_size == 300:
        features = ['relu22_fwd', 'relu26_fwd']
    elif base_network_name == 'mobilenet0.25' and base_size == 300:
        features = ['relu22_fwd', 'relu26_fwd']
    else:
        raise NotImplementedError('Unsupported network', base_network_name)

    net = get_ssd(name=base_network_name,
                  base_size=base_size,
                  features=features,
                  filters=filters,
                  sizes=sizes,
                  ratios=ratios,
                  steps=steps,
                  classes=classes,
                  dataset=dataset,
                  pretrained_base=pretrained_base,
                  **kwargs)

    return net
