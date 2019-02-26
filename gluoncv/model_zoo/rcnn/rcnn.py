"""RCNN Model."""
from __future__ import absolute_import

import warnings
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from ...nn.bbox import BBoxCornerToCenter
from ...nn.coder import NormalizedBoxCenterDecoder, MultiPerClassDecoder


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
    post_nms : int
        Only return top `post_nms` detection results, the rest is discarded. The number is
        based on COCO dataset which has maximum 100 objects per image. You can adjust this
        number if expecting more objects. You can use -1 to return all detections.
    roi_mode : str
        ROI pooling mode. Currently support 'pool' and 'align'.
    roi_size : tuple of int, length 2
        (height, width) of the ROI region.
    stride : int
        Stride of network features.
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
    post_nms : int
        Only return top `post_nms` detection results, the rest is discarded. The number is
        based on COCO dataset which has maximum 100 objects per image. You can adjust this
        number if expecting more objects. You can use -1 to return all detections.

    """
    def __init__(self, features, top_features, classes,
                 short, max_size, train_patterns,
                 nms_thresh, nms_topk, post_nms,
                 roi_mode, roi_size, stride, clip, **kwargs):
        super(RCNN, self).__init__(**kwargs)
        self.classes = classes
        self.num_class = len(classes)
        self.short = short
        self.max_size = max_size
        self.train_patterns = train_patterns
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms

        assert self.num_class > 0, "Invalid number of class : {}".format(self.num_class)
        assert roi_mode.lower() in ['align', 'pool'], "Invalid roi_mode: {}".format(roi_mode)
        self._roi_mode = roi_mode.lower()
        assert len(roi_size) == 2, "Require (h, w) as roi_size, given {}".format(roi_size)
        self._roi_size = roi_size
        self._stride = stride

        with self.name_scope():
            self.features = features
            self.top_features = top_features
            self.global_avg_pool = nn.GlobalAvgPool2D()
            self.class_predictor = nn.Dense(
                self.num_class + 1, weight_initializer=mx.init.Normal(0.01))
            self.box_predictor = nn.Dense(
                self.num_class * 4, weight_initializer=mx.init.Normal(0.001))
            self.cls_decoder = MultiPerClassDecoder(num_class=self.num_class+1)
            self.box_to_center = BBoxCornerToCenter()
            self.box_decoder = NormalizedBoxCenterDecoder(clip=clip)

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

    def set_nms(self, nms_thresh=0.3, nms_topk=400, post_nms=100):
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

    def reset_class(self, classes, reuse_weights=None):
        """Reset class categories and class predictors.

        Parameters
        ----------
        classes : iterable of str
            The new categories. ['apple', 'orange'] for example.

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
            self.class_predictor.initialize()
            self.box_predictor.initialize()
            if reuse_weights:
                assert isinstance(reuse_weights, dict)
                # class predictors
                srcs = (old_class_pred, old_box_pred, 1)
                dsts = (self.class_predictor, self.box_predictor, 0)
                for src, dst, offset in zip(srcs, dsts):
                    for old_params, new_params in zip(src.params.values(),
                                                      dst.params.values()):
                        # slice and copy weights
                        old_data = old_params.data()
                        new_data = new_params.data()

                        print(new_data.shape, old_data.shape)
                        for k, v in reuse_weights.items():
                            if k >= len(self.classes) - 1 or v >= len(old_classes) - 1:
                                warnings.warn("reuse mapping {}/{} -> {}/{} out of range".format(
                                    k, self.classes, v, old_classes))
                                continue

                            # always increment k and v (background is always the 0th)
                            new_data[k+offset::len(self.classes)+offset] = old_data[v+offset::len(old_classes)+offset]
                        # reuse background weights as well
                        new_data[0::len(self.classes)+offset] = old_data[0::len(old_classes)+offset]
                        # set data to new conv layers
                        new_params.set_data(new_data)





    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, x, width, height):
        """Not implemented yet."""
        raise NotImplementedError
