"""RCNN Model."""
from __future__ import absolute_import

import warnings

import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn

from ...nn.coder import NormalizedBoxCenterDecoder, MultiPerClassDecoder
from ...nn.feature import FPNFeatureExpander


class RCNN(gluon.HybridBlock):
    """RCNN network.

    Parameters
    ----------
    features : gluon.HybridBlock
        Base feature extractor before feature pooling layer.
    top_features : gluon.HybridBlock
        Tail feature extractor after feature pooling layer.
    classes : iterable of str
        Names of categories, its length is ``num_class``.
    box_features : gluon.HybridBlock
        feature head for transforming roi output for box prediction.
    short : int
        Input image short side size.
    max_size : int
        Maximum size of input image long side.
    train_patterns : str
        Matching pattern for trainable parameters.
    nms_thresh : float
        Non-maximum suppression threshold. You can specify < 0 or > 1 to disable NMS.
    nms_topk : int
        Apply NMS to top k detection results, use -1 to disable so that every Detection
         result is used in NMS.
    roi_mode : str
        ROI pooling mode. Currently support 'pool' and 'align'.
    roi_size : tuple of int, length 2
        (height, width) of the ROI region.
    strides : int/tuple of ints
        Stride(s) of network features. Tuple for FPN.
    clip: float
        Clip bounding box target to this value.

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
        Non-maximum suppression threshold. You can specify < 0 or > 1 to disable NMS.
    nms_topk : int
        Apply NMS to top k detection results, use -1 to disable so that every Detection
         result is used in NMS.
    minimal_opset : bool
        We sometimes add special operators to accelerate training/inference, however, for exporting
        to third party compilers we want to utilize most widely used operators.
        If `minimal_opset` is `True`, the network will use a minimal set of operators good
        for e.g., `TVM`.
    """

    def __init__(self, features, top_features, classes, box_features, short, max_size,
                 train_patterns, nms_thresh, nms_topk, post_nms, roi_mode, roi_size, strides, clip,
                 force_nms=False, minimal_opset=False, **kwargs):
        super(RCNN, self).__init__(**kwargs)
        self.classes = classes
        self.num_class = len(classes)
        self.short = short
        self.max_size = max_size
        self.train_patterns = train_patterns
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms
        self.force_nms = force_nms

        assert self.num_class > 0, "Invalid number of class : {}".format(self.num_class)
        assert roi_mode.lower() in ['align', 'pool'], "Invalid roi_mode: {}".format(roi_mode)
        self._roi_mode = roi_mode.lower()
        assert len(roi_size) == 2, "Require (h, w) as roi_size, given {}".format(roi_size)
        self._roi_size = roi_size
        self._strides = strides

        with self.name_scope():
            self.features = features
            self.top_features = top_features
            self.box_features = box_features
            self.class_predictor = nn.Dense(
                self.num_class + 1, weight_initializer=mx.init.Normal(0.01))
            self.box_predictor = nn.Dense(
                self.num_class * 4, weight_initializer=mx.init.Normal(0.001))
            self.cls_decoder = MultiPerClassDecoder(num_class=self.num_class + 1)
            self.box_decoder = NormalizedBoxCenterDecoder(clip=clip, convert_anchor=True, minimal_opset=minimal_opset)

    def collect_train_params(self, select=None):
        """Collect trainable params.

        This function serves as a help utility function to return only
        trainable parameters if predefined by experienced developer/researcher.
        For example, if cross-device BatchNorm is not enabled, we will definitely
        want to fix BatchNorm statistics to avoid scaling problem because RCNN training
        batch size is usually very small.

        Parameters
        ----------
        select : select : str
            Regular expressions for parameter match pattern

        Returns
        -------
        The selected :py:class:`mxnet.gluon.ParameterDict`

        """
        if select is None:
            return self.collect_params(self.train_patterns)
        return self.collect_params(select)

    def set_nms(self, nms_thresh=0.3, nms_topk=400, force_nms=False, post_nms=100):
        """Set NMS parameters to the network.

        .. Note::
            If you are using hybrid mode, make sure you re-hybridize after calling
            ``set_nms``.

        Parameters
        ----------
        nms_thresh : float, default is 0.3.
            Non-maximum suppression threshold. You can specify < 0 or > 1 to disable NMS.
        nms_topk : int, default is 400
            Apply NMS to top k detection results, use -1 to disable so that every Detection
             result is used in NMS.
        force_nms : bool, default is False
            Appy NMS to all categories, this is to avoid overlapping detection results
            from different categories.

        Returns
        -------
        None

        """
        self._clear_cached_op()
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.force_nms = force_nms
        self.post_nms = post_nms

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

        """
        self._clear_cached_op()
        if reuse_weights:
            assert hasattr(self, 'classes'), "require old classes to reuse weights"
        old_classes = getattr(self, 'classes', [])
        self.classes = classes
        self.num_class = len(classes)
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
            old_class_pred = self.class_predictor
            old_box_pred = self.box_predictor
            ctx = list(old_class_pred.params.values())[0].list_ctx()
            # to avoid deferred init, number of in_channels must be defined
            in_units = list(old_class_pred.params.values())[0].shape[1]
            self.class_predictor = nn.Dense(
                self.num_class + 1, weight_initializer=mx.init.Normal(0.01),
                prefix=self.class_predictor.prefix, in_units=in_units)
            self.box_predictor = nn.Dense(
                self.num_class * 4, weight_initializer=mx.init.Normal(0.001),
                prefix=self.box_predictor.prefix, in_units=in_units)
            self.cls_decoder = MultiPerClassDecoder(num_class=self.num_class + 1)
            # initialize
            self.class_predictor.initialize(ctx=ctx)
            self.box_predictor.initialize(ctx=ctx)
            if reuse_weights:
                assert isinstance(reuse_weights, dict)
                # class predictors
                srcs = (old_class_pred, old_box_pred)
                dsts = (self.class_predictor, self.box_predictor)
                offsets = (1, 0)  # class predictor has bg, box don't
                lens = (1, 4)  # class predictor length=1, box length=4
                for src, dst, offset, l in zip(srcs, dsts, offsets, lens):
                    for old_params, new_params in zip(src.params.values(),
                                                      dst.params.values()):
                        # slice and copy weights
                        old_data = old_params.data()
                        new_data = new_params.data()

                        for k, v in reuse_weights.items():
                            if k >= len(self.classes) or v >= len(old_classes):
                                warnings.warn("reuse mapping {}/{} -> {}/{} out of range".format(
                                    k, self.classes, v, old_classes))
                                continue
                            new_data[(k + offset) * l:(k + offset + 1) * l] = \
                                old_data[(v + offset) * l:(v + offset + 1) * l]
                        # reuse background weights as well
                        if offset > 0:
                            new_data[0:l] = old_data[0:l]
                        # set data to new conv layers
                        new_params.set_data(new_data)

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, x, width, height):
        """Not implemented yet."""
        raise NotImplementedError


def custom_rcnn_fpn(pretrained_base=True, base_network_name='resnet18_v1b', norm_layer=nn.BatchNorm,
                    norm_kwargs=None, sym_norm_layer=None, sym_norm_kwargs=None,
                    num_fpn_filters=256, num_box_head_conv=4, num_box_head_conv_filters=256,
                    num_box_head_dense_filters=1024):
    r"""Generate custom RCNN model with resnet base network w/FPN.

    Parameters
    ----------
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

    Returns
    -------
    SymbolBlock or HybridBlock
        Base feature extractor eg. resnet w/ FPN.
    None or HybridBlock
        R-CNN feature before each task heads.
    HybridBlock
        Box feature extractor
    """
    use_global_stats = norm_layer is nn.BatchNorm
    if base_network_name == 'resnet18_v1b':
        from ...model_zoo.resnetv1b import resnet18_v1b
        base_network = resnet18_v1b(pretrained=pretrained_base, dilated=False,
                                    use_global_stats=use_global_stats, norm_layer=norm_layer,
                                    norm_kwargs=norm_kwargs)
        fpn_inputs_names = ['layers1_relu3_fwd', 'layers2_relu3_fwd', 'layers3_relu3_fwd',
                            'layers4_relu3_fwd']
    elif base_network_name == 'resnet50_v1b':
        from ...model_zoo.resnetv1b import resnet50_v1b
        base_network = resnet50_v1b(pretrained=pretrained_base, dilated=False,
                                    use_global_stats=use_global_stats, norm_layer=norm_layer,
                                    norm_kwargs=norm_kwargs)
        fpn_inputs_names = ['layers1_relu8_fwd', 'layers2_relu11_fwd', 'layers3_relu17_fwd',
                            'layers4_relu8_fwd']
    elif base_network_name == 'resnet101_v1d':
        from ...model_zoo.resnetv1b import resnet101_v1d
        base_network = resnet101_v1d(pretrained=pretrained_base, dilated=False,
                                     use_global_stats=use_global_stats, norm_layer=norm_layer,
                                     norm_kwargs=norm_kwargs)
        fpn_inputs_names = ['layers1_relu8_fwd', 'layers2_relu11_fwd', 'layers3_relu68_fwd',
                            'layers4_relu8_fwd']
    elif base_network_name == 'resnest50':
        from ...model_zoo.resnest import resnest50
        base_network = resnest50(pretrained=pretrained_base, dilated=False,
                                 use_global_stats=use_global_stats, norm_layer=norm_layer,
                                 norm_kwargs=norm_kwargs)
        fpn_inputs_names = ['layers1_relu11_fwd', 'layers2_relu15_fwd', 'layers3_relu23_fwd',
                            'layers4_relu11_fwd']
    elif base_network_name == 'resnest101':
        from ...model_zoo.resnest import resnest101
        base_network = resnest101(pretrained=pretrained_base, dilated=False,
                                  use_global_stats=use_global_stats, norm_layer=norm_layer,
                                  norm_kwargs=norm_kwargs)
        fpn_inputs_names = ['layers1_relu11_fwd', 'layers2_relu15_fwd', 'layers3_relu91_fwd',
                            'layers4_relu11_fwd']
    else:
        raise NotImplementedError('Unsupported network', base_network_name)
    features = FPNFeatureExpander(
        network=base_network, outputs=fpn_inputs_names,
        num_filters=[num_fpn_filters] * len(fpn_inputs_names), use_1x1=True,
        use_upsample=True, use_elewadd=True, use_p6=True, no_bias=not use_global_stats,
        pretrained=pretrained_base, norm_layer=sym_norm_layer, norm_kwargs=sym_norm_kwargs)
    top_features = None
    box_features = nn.HybridSequential()
    box_features.add(nn.AvgPool2D(pool_size=(3, 3), strides=2, padding=1))  # reduce to 7x7
    if use_global_stats:
        box_features.add(
            nn.Dense(num_box_head_dense_filters, weight_initializer=mx.init.Normal(0.01)),
            nn.Activation('relu'))
    else:
        for _ in range(num_box_head_conv):
            box_features.add(nn.Conv2D(num_box_head_conv_filters, 3, padding=1, use_bias=False),
                             norm_layer(**norm_kwargs),
                             nn.Activation('relu'))
    box_features.add(
        nn.Dense(num_box_head_dense_filters, weight_initializer=mx.init.Normal(0.01)),
        nn.Activation('relu'))
    return features, top_features, box_features
