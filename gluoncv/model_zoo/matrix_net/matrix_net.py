"""MatrixNet object detector: using matrix layers to extract features and heads of CenterNet to predict
     matrix layers: https://arxiv.org/abs/2001.03194 and its code on github"""
from __future__ import absolute_import

import os
import warnings
from collections import OrderedDict

import mxnet as mx
from mxnet.gluon import nn
from mxnet import autograd
from ...nn.coder import MatrixNetDecoder
from ...nn.feature import FPNFeatureExpander

__all__ = ['MatrixNet', 'get_matrix_net','matrix_net_resnet101_v1d_coco']

class MatrixNet(nn.HybridBlock):
    """https://arxiv.org/abs/2001.03194

    Parameters
    ----------
    base_network : mxnet.gluon.nn.SymbolBlock
        The base feature extraction network.
        Currently just using pre-defined resnet101_v1d_fpn
    heads : OrderedDict
        OrderedDict with specifications for each head.
        For example: OrderedDict([
            ('heatmap', {'num_output': len(classes), 'bias': -2.19}),
            ('wh', {'num_output': 2}),
            ('reg', {'num_output': 2})
            ])
    classes : list of str
        Category names.
    layers_range : list of list of number(list of number)
        Denotes the size of objects assigned to each matrix layer
        layers_range is a 5 * 5 matrix, where each element is -1 or a list of 4 numbers
         -1 denotes this layer is pruned, a list of 4 numbers is min height, max height, 
         min width, max width of the objects respectively.
    head_conv_channel : int, default is 0
        If > 0, will use an extra conv layer before each of the real heads.
    base_layer_scale : float, default is 4.0
        The downsampling ratio of the first (top-left) layer in the matrix.
    topk : int, default is 100
        Number of outputs .
    flip_test : bool
        Whether apply flip test in inference (training mode not affected).
    nms_thresh : float, default is 0.5.
        Non-maximum suppression threshold. You can specify < 0 or > 1 to disable NMS.
    nms_topk : int, default is 300
        Apply NMS to top k detection results, use -1 to disable so that every Detection
         result is used in NMS.
        Choose the default value according to the code of matrixnets. 
    post_nms : int, default is 100
        Only return top `post_nms` detection results, the rest is discarded. The number is
        based on COCO dataset which has maximum 100 objects per image. You can adjust this
        number if expecting more objects. You can use -1 to return all detections.

    """
    def __init__(self, base_network, heads, classes, layers_range,
                 head_conv_channel=0, base_layer_scale=4.0, topk=100, flip_test=False,
                 nms_thresh=0.5, nms_topk=300, post_nms=100, **kwargs):
        if 'norm_layer' in kwargs:
            kwargs.pop('norm_layer')
        if 'norm_kwargs' in kwargs:
            kwargs.pop('norm_kwargs')
        super(MatrixNet, self).__init__(**kwargs)
        assert isinstance(heads, OrderedDict), \
            "Expecting heads to be a OrderedDict per head, given {}" \
            .format(type(heads))
        self.classes = classes
        self.topk = topk
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        post_nms = min(post_nms, topk)
        self.post_nms = post_nms
        self.base_layer_scale = base_layer_scale
        self.layers_range = layers_range
        self.flip_test = flip_test
        with self.name_scope():
            self.base_network = base_network
            self.heatmap_nms = nn.MaxPool2D(pool_size=3, strides=1, padding=1)
            weight_initializer = mx.init.Normal(0.01)
            # the following two layers are used to generate the off-diagonal layers' features from diagonal layers' features
            self.downsample_transformation_12 = nn.Conv2D(
                        256, kernel_size=3, padding=1, strides=(1,2), use_bias=True,
                        weight_initializer=weight_initializer, bias_initializer='zeros')
            self.downsample_transformation_21 = nn.Conv2D(
                        256, kernel_size=3, padding=1, strides=(2,1), use_bias=True,
                        weight_initializer=weight_initializer, bias_initializer='zeros')
            self.decoder = MatrixNetDecoder(topk=topk, base_layer_scale=base_layer_scale)
            # using heads of CenterNet( Objects as Point )
            self.heads = nn.HybridSequential('heads')
            for name, values in heads.items():
                head = nn.HybridSequential(name)
                num_output = values['num_output']
                bias = values.get('bias', 0.0)
                weight_initializer = mx.init.Normal(0.001) if bias == 0 else mx.init.Xavier()
                if head_conv_channel > 0:
                    head.add(nn.Conv2D(
                        head_conv_channel, kernel_size=3, padding=1, use_bias=True,
                        weight_initializer=weight_initializer, bias_initializer='zeros'))
                    head.add(nn.Activation('relu'))
                head.add(nn.Conv2D(num_output, kernel_size=1, strides=1, padding=0, use_bias=True,
                                   weight_initializer=weight_initializer,
                                   bias_initializer=mx.init.Constant(bias)))

                self.heads.add(head)

    @property
    def num_classes(self):
        """Return number of foreground classes.

        Returns
        -------
        int
            Number of foreground classes

        """
        return len(self.classes)

    def set_nms(self, nms_thresh=0, nms_topk=400, post_nms=100):
        """Set non-maximum suppression parameters.

        Parameters
        ----------
        nms_thresh : float, default is 0.
            Non-maximum suppression threshold. You can specify < 0 or > 1 to disable NMS.
            By default NMS is disabled.
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
        post_nms = min(post_nms, self.nms_topk)
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
        raise NotImplementedError("Not yet implemented, please wait for future updates.")

    def hybrid_forward(self, F, x):
        # pylint: disable=arguments-differ
        """Hybrid forward of matrixnet"""
        # following lines computes the features of 19 matrix layers
        # 5 diagonal features are FPN outputs, others are computed from the diagonal features
        # this part can be impoved by modifying the code of FPNFeatureExpander
        feature_2, feature_3, feature_4, feature_5, feature_6 = self.base_network(x)
        _dict = {}
        _dict[11] = feature_2
        _dict[22] = feature_3
        _dict[33] = feature_4
        _dict[44] = feature_5
        _dict[55] = feature_6
        _dict[12] = self.downsample_transformation_21(_dict[11])
        _dict[13] = self.downsample_transformation_21(_dict[12])
        _dict[23] = self.downsample_transformation_21(_dict[22])
        _dict[24] = self.downsample_transformation_21(_dict[23])
        _dict[34] = self.downsample_transformation_21(_dict[33])
        _dict[35] = self.downsample_transformation_21(_dict[34])
        _dict[45] = self.downsample_transformation_21(_dict[44])
        _dict[21] = self.downsample_transformation_12(_dict[11])
        _dict[31] = self.downsample_transformation_12(_dict[21])
        _dict[32] = self.downsample_transformation_12(_dict[22])
        _dict[42] = self.downsample_transformation_12(_dict[32])
        _dict[43] = self.downsample_transformation_12(_dict[33])
        _dict[53] = self.downsample_transformation_12(_dict[43])
        _dict[54] = self.downsample_transformation_12(_dict[44])
        
        # run the shared heads on the 19 features
        ys = [ _dict[i] for i in sorted(_dict)]
        heatmaps = [self.heads[0](y) for y in ys]
        wh_preds = [self.heads[1](y) for y in ys]
        center_regrs = [self.heads[2](y) for y in ys]
        heatmaps = [F.sigmoid(heatmap) for heatmap in heatmaps]
        if autograd.is_training():
            heatmaps = [F.clip(heatmap, 1e-4, 1 - 1e-4) for heatmap in heatmaps]
            return heatmaps, wh_preds, center_regrs
        print('whether flip_test: {}'.format(self.flip_test))
        if self.flip_test:
            # some duplicate code, can be optimized by modifying the code of FPNFeatureExpander.
            feature_2_flip, feature_3_flip, feature_4_flip, feature_5_flip, feature_6_flip = self.base_network(x.flip(axis=3))
            _dict_flip = {}
            _dict_flip[11] = feature_2_flip
            _dict_flip[22] = feature_3_flip
            _dict_flip[33] = feature_4_flip
            _dict_flip[44] = feature_5_flip
            _dict_flip[55] = feature_6_flip
            _dict_flip[12] = self.downsample_transformation_21(_dict_flip[11])
            _dict_flip[13] = self.downsample_transformation_21(_dict_flip[12])
            _dict_flip[23] = self.downsample_transformation_21(_dict_flip[22])
            _dict_flip[24] = self.downsample_transformation_21(_dict_flip[23])
            _dict_flip[34] = self.downsample_transformation_21(_dict_flip[33])
            _dict_flip[35] = self.downsample_transformation_21(_dict_flip[34])
            _dict_flip[45] = self.downsample_transformation_21(_dict_flip[44])
            _dict_flip[21] = self.downsample_transformation_12(_dict_flip[11])
            _dict_flip[31] = self.downsample_transformation_12(_dict_flip[21])
            _dict_flip[32] = self.downsample_transformation_12(_dict_flip[22])
            _dict_flip[42] = self.downsample_transformation_12(_dict_flip[32])
            _dict_flip[43] = self.downsample_transformation_12(_dict_flip[33])
            _dict_flip[53] = self.downsample_transformation_12(_dict_flip[43])
            _dict_flip[54] = self.downsample_transformation_12(_dict_flip[44])
            ys_flip = [ _dict_flip[i] for i in sorted(_dict_flip)]
            heatmaps_flip = [self.heads[0](y) for y in ys_flip]
            wh_preds_flip = [self.heads[1](y) for y in ys_flip]
            center_regrs_flip = [self.heads[2](y) for y in ys_flip]
            heatmaps_flip = [F.sigmoid(heatmap) for heatmap in heatmaps_flip]
            for i in range(len(heatmaps)):
                heatmaps[i] = (heatmaps[i] + heatmaps_flip[i].flip(axis=3)) * 0.5
                wh_preds[i] = (wh_preds[i] + wh_preds_flip[i].flip(axis=3)) * 0.5

        
        keeps = [F.broadcast_equal(self.heatmap_nms(heatmap), heatmap) for heatmap in heatmaps]
        results = self.decoder(keeps, heatmaps, wh_preds, center_regrs)
        #since the 19 matrix layers may generate duplicate results, add soft-nms for post-processing
        if self.nms_thresh > 0 and self.nms_thresh < 1:
            results = F.contrib.box_nms(
                results, overlap_thresh=self.nms_thresh, topk=self.nms_topk, valid_thresh=0.01,
                id_index=0, score_index=1, coord_start=2, force_suppress=False)
            if self.post_nms > 0:
                results = results.slice_axis(axis=1, begin=0, end=self.post_nms)
        ids = F.slice_axis(results, axis=2, begin=0, end=1)
        scores = F.slice_axis(results, axis=2, begin=1, end=2)
        bboxes = F.slice_axis(results, axis=2, begin=2, end=6)
        return ids, scores, bboxes
        

def get_matrix_net(name, dataset, pretrained=False, ctx=mx.cpu(),
                   root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    """Get a matrix net instance.

    Parameters
    ----------
    name : str or None
        Model name, if `None` is used, you must specify `features` to be a `HybridBlock`.
    dataset : str
        Name of dataset. This is used to identify model name because models trained on
        different datasets are going to be very different.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : mxnet.Context
        Context such as mx.cpu(), mx.gpu(0).
    root : str
        Model weights storing path.

    Returns
    -------
    HybridBlock
        A MatrixNet detection network.

    """
    # pylint: disable=unused-variable
    net = MatrixNet(**kwargs)
    if pretrained:
        from ..model_store import get_model_file
        full_name = '_'.join(('matrix_net', name, dataset))
        net.load_parameters(get_model_file(full_name, tag=pretrained, root=root), ctx=ctx)
    else:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            net.initialize()
        for v in net.collect_params().values():
            try:
                v.reset_ctx(ctx)
            except ValueError:
                pass
    return net

def matrix_net_resnet101_v1d_coco(pretrained=False, pretrained_base=True, **kwargs):
    """MatrixNet with resnet101_v1d base network on coco dataset.

    Parameters
    ----------
    classes : iterable of str
        Names of custom foreground classes. `len(classes)` is the number of foreground classes.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized.

    Returns
    -------
    HybridBlock
        A MatrixNet detection network.

    """
    from ...model_zoo.resnetv1b import resnet101_v1d
    from ...data import COCODetection
    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet101_v1d(pretrained=pretrained_base, dilated=False,
                                 use_global_stats=True, **kwargs)
    features = FPNFeatureExpander(
        network=base_network,
        outputs=['layers1_relu8_fwd', 'layers2_relu11_fwd', 'layers3_relu68_fwd',
                 'layers4_relu8_fwd'], num_filters=[256, 256, 256, 256], use_1x1=True,
        use_upsample=True, use_elewadd=True, use_p6=True, no_bias=False, pretrained=pretrained_base)
    heads = OrderedDict([
        ('heatmap', {'num_output': len(classes), 'bias': -2.19}), # use bias = -log((1 - 0.1) / 0.1)
        ('wh', {'num_output': 2}),
        ('reg', {'num_output': 2})
        ])
    # according to the reference code of the paper(https://arxiv.org/abs/2001.03194), there can be up to 25 matrix layers
    # layers_range is the configuration containing 5*5 elements. 
    # Each element can be -1 (meaning this layer is cut, so this position is empty, no need to generate features
    # Or the element can be a list of 4 numbers, standing for min_height, max_height, min_width, max_width of the objects assigned
    #    as this layers' traing target
    layers_range = [[[0,48,0,48],[48,96,0,48],[96,192,0,48], -1, -1],
                 [[0,48,48,96],[48,96,48,96],[96,192,48,96],[192,384,0,96], -1],
                 [[0,48,96,192],[48,96,96,192],[96,192,96,192],[192,384,96,192],[384,2000,96,192]],
                 [-1, [0,96,192,384],[96,192,192,384],[192,384,192,384],[384,2000,192,384]],
                 [-1, -1, [0,192,384,2000],[192,384,384,2000],[384,2000,384,2000]]]
    return get_matrix_net('resnet101_v1d', 'coco', base_network=features, heads=heads, layers_range = layers_range, 
                          head_conv_channel=64, pretrained=pretrained, classes=classes,
                          base_layer_scale=4.0, topk=100, **kwargs)
    
