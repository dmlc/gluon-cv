"""CenterNet object detector: Objects as Points, https://arxiv.org/abs/1904.07850"""
from __future__ import absolute_import

import os
import warnings
from collections import OrderedDict

import mxnet as mx
from mxnet.gluon import nn
from mxnet import autograd
from ...nn.coder import CenterNetDecoder

__all__ = ['CenterNet', 'get_center_net', 'get_base_network',
           'center_net_resnet18_v1b_voc', 'center_net_resnet18_v1b_dcnv2_voc',
           'center_net_resnet18_v1b_coco', 'center_net_resnet18_v1b_dcnv2_coco',
           'center_net_resnet50_v1b_voc', 'center_net_resnet50_v1b_dcnv2_voc',
           'center_net_resnet50_v1b_coco', 'center_net_resnet50_v1b_dcnv2_coco',
           'center_net_resnet101_v1b_voc', 'center_net_resnet101_v1b_dcnv2_voc',
           'center_net_resnet101_v1b_coco', 'center_net_resnet101_v1b_dcnv2_coco',
           'center_net_dla34_voc', 'center_net_dla34_dcnv2_voc',
           'center_net_dla34_coco', 'center_net_dla34_dcnv2_coco',
           'center_net_mobilenetv3_large_duc_voc', 'center_net_mobilenetv3_large_duc_coco',
           'center_net_mobilenetv3_small_duc_voc', 'center_net_mobilenetv3_small_duc_coco'
           ]

class CenterNet(nn.HybridBlock):
    """Objects as Points. https://arxiv.org/abs/1904.07850v2

    Parameters
    ----------
    base_network : mxnet.gluon.nn.HybridBlock
        The base feature extraction network.
    heads : OrderedDict
        OrderedDict with specifications for each head.
        For example: OrderedDict([
            ('heatmap', {'num_output': len(classes), 'bias': -2.19}),
            ('wh', {'num_output': 2}),
            ('reg', {'num_output': 2})
            ])
    classes : list of str
        Category names.
    head_conv_channel : int, default is 0
        If > 0, will use an extra conv layer before each of the real heads.
    scale : float, default is 4.0
        The downsampling ratio of the entire network.
    topk : int, default is 100
        Number of outputs .
    flip_test : bool
        Whether apply flip test in inference (training mode not affected).
    nms_thresh : float, default is 0.
        Non-maximum suppression threshold. You can specify < 0 or > 1 to disable NMS.
        By default nms is disabled.
    nms_topk : int, default is 400
        Apply NMS to top k detection results, use -1 to disable so that every Detection
         result is used in NMS.
    post_nms : int, default is 100
        Only return top `post_nms` detection results, the rest is discarded. The number is
        based on COCO dataset which has maximum 100 objects per image. You can adjust this
        number if expecting more objects. You can use -1 to return all detections.

    """
    def __init__(self, base_network, heads, classes,
                 head_conv_channel=0, scale=4.0, topk=100, flip_test=False,
                 nms_thresh=0, nms_topk=400, post_nms=100, **kwargs):
        if 'norm_layer' in kwargs:
            kwargs.pop('norm_layer')
        if 'norm_kwargs' in kwargs:
            kwargs.pop('norm_kwargs')
        super(CenterNet, self).__init__(**kwargs)
        assert isinstance(heads, OrderedDict), \
            "Expecting heads to be a OrderedDict per head, given {}" \
            .format(type(heads))
        self.classes = classes
        self.topk = topk
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        post_nms = min(post_nms, topk)
        self.post_nms = post_nms
        self.scale = scale
        self.flip_test = flip_test
        self._head_setups = heads
        self._head_conv_channel = head_conv_channel
        with self.name_scope():
            self.base_network = base_network
            self.heatmap_nms = nn.MaxPool2D(pool_size=3, strides=1, padding=1)
            self.decoder = CenterNetDecoder(topk=topk, scale=scale)
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

        Example
        -------
        >>> net = gluoncv.model_zoo.get_model('center_net_resnet50_v1b_voc', pretrained=True)
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
            hm_head = nn.HybridSequential('heatmap')
            orig_head = self.heads
            orig_hm = self.heads[0]
            for i in range(len(orig_hm) - 1):
                hm_head.add(orig_hm[i])
            num_output = len(classes)
            bias = self._head_setups['heatmap'].get('bias', 0.0)
            weight_initializer = mx.init.Normal(0.001) if bias == 0 else mx.init.Xavier()
            # to avoid deferred init, number of in_channels must be defined
            in_channels = list(orig_hm[0].params.values())[0].shape[1]
            hm_head.add(nn.Conv2D(num_output, kernel_size=1, strides=1, padding=0, use_bias=True,
                                  weight_initializer=weight_initializer,
                                  bias_initializer=mx.init.Constant(bias),
                                  in_channels=in_channels))
            with warnings.catch_warnings(record=True) as _:
                warnings.simplefilter("always")
                ctx = list(orig_hm[0].params.values())[0].list_ctx()
                hm_head.initialize(ctx=ctx)
            if reuse_weights:
                assert isinstance(reuse_weights, dict)
                for old_params, new_params in zip(orig_hm[2].params.values(),
                                                  hm_head[2].params.values()):
                    old_data = old_params.data()
                    new_data = new_params.data()

                    for k, v in reuse_weights.items():
                        if k > len(self.classes) or v > len(old_classes):
                            warnings.warn("reuse mapping {}/{} -> {}/{} out of range".format(
                                k, self.classes, v, old_classes))
                            continue
                        new_data[k::len(self.classes)] = old_data[v::len(old_classes)]
                    # set data to new conv layers
                    new_params.set_data(new_data)
            old_heads = self.heads
            self.heads = nn.HybridSequential('heads')
            self.heads.add(hm_head)
            self.heads.add(orig_head[1])
            self.heads.add(orig_head[2])

    def hybrid_forward(self, F, x):
        # pylint: disable=arguments-differ
        """Hybrid forward of center net"""
        y = self.base_network(x)
        out = [head(y) for head in self.heads]
        out[0] = F.sigmoid(out[0])
        if autograd.is_training():
            out[0] = F.clip(out[0], 1e-4, 1 - 1e-4)
            return tuple(out)
        if self.flip_test:
            y_flip = self.base_network(x.flip(axis=3))
            out_flip = [head(y_flip) for head in self.heads]
            out_flip[0] = F.sigmoid(out_flip[0])
            out[0] = (out[0] + out_flip[0].flip(axis=3)) * 0.5
            out[1] = (out[1] + out_flip[1].flip(axis=3)) * 0.5
        heatmap = out[0]
        keep = F.broadcast_equal(self.heatmap_nms(heatmap), heatmap)
        results = self.decoder(keep * heatmap, out[1], out[2])
        return results

def get_base_network(name, **kwargs):
    """Get centernet base network"""
    if 'dla' in name:
        from .deconv_dla import get_deconv_dla
        kwargs['use_dcnv2'] = 'dcnv2' in name
        return get_deconv_dla(name.split('_')[0], **kwargs)
    elif 'resnet' in name:
        from .deconv_resnet import get_deconv_resnet
        kwargs['use_dcnv2'] = 'dcnv2' in name
        return get_deconv_resnet('_'.join(name.split('_')[:2]), **kwargs)
    else:
        raise ValueError(name)

def get_center_net(name, dataset, pretrained=False, ctx=mx.cpu(),
                   root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    """Get a center net instance.

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
        A CenterNet detection network.

    """
    # pylint: disable=unused-variable
    net = CenterNet(**kwargs)
    if pretrained:
        from ..model_store import get_model_file
        full_name = '_'.join(('center_net', name, dataset))
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

def center_net_resnet18_v1b_voc(pretrained=False, pretrained_base=True, **kwargs):
    """Center net with resnet18_v1b base network on voc dataset.

    Parameters
    ----------
    classes : iterable of str
        Names of custom foreground classes. `len(classes)` is the number of foreground classes.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized.

    Returns
    -------
    HybridBlock
        A CenterNet detection network.

    """
    from .deconv_resnet import resnet18_v1b_deconv
    from ...data import VOCDetection
    classes = VOCDetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet18_v1b_deconv(pretrained=pretrained_base, **kwargs)
    heads = OrderedDict([
        ('heatmap', {'num_output': len(classes), 'bias': -2.19}), # use bias = -log((1 - 0.1) / 0.1)
        ('wh', {'num_output': 2}),
        ('reg', {'num_output': 2})
        ])
    return get_center_net('resnet18_v1b', 'voc', base_network=base_network, heads=heads,
                          head_conv_channel=64, pretrained=pretrained, classes=classes,
                          scale=4.0, topk=40, **kwargs)

def center_net_resnet18_v1b_dcnv2_voc(pretrained=False, pretrained_base=True, **kwargs):
    """Center net with resnet18_v1b base network with deformable v2 conv layers on voc dataset.

    Parameters
    ----------
    classes : iterable of str
        Names of custom foreground classes. `len(classes)` is the number of foreground classes.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized.

    Returns
    -------
    HybridBlock
        A CenterNet detection network.

    """
    from .deconv_resnet import resnet18_v1b_deconv_dcnv2
    from ...data import VOCDetection
    classes = VOCDetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet18_v1b_deconv_dcnv2(pretrained=pretrained_base, **kwargs)
    heads = OrderedDict([
        ('heatmap', {'num_output': len(classes), 'bias': -2.19}), # use bias = -log((1 - 0.1) / 0.1)
        ('wh', {'num_output': 2}),
        ('reg', {'num_output': 2})
        ])
    return get_center_net('resnet18_v1b_dcnv2', 'voc', base_network=base_network, heads=heads,
                          head_conv_channel=64, pretrained=pretrained, classes=classes,
                          scale=4.0, topk=40, **kwargs)

def center_net_resnet18_v1b_coco(pretrained=False, pretrained_base=True, **kwargs):
    """Center net with resnet18_v1b base network on coco dataset.

    Parameters
    ----------
    classes : iterable of str
        Names of custom foreground classes. `len(classes)` is the number of foreground classes.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized.

    Returns
    -------
    HybridBlock
        A CenterNet detection network.

    """
    from .deconv_resnet import resnet18_v1b_deconv
    from ...data import COCODetection
    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet18_v1b_deconv(pretrained=pretrained_base, **kwargs)
    heads = OrderedDict([
        ('heatmap', {'num_output': len(classes), 'bias': -2.19}), # use bias = -log((1 - 0.1) / 0.1)
        ('wh', {'num_output': 2}),
        ('reg', {'num_output': 2})
        ])
    return get_center_net('resnet18_v1b', 'coco', base_network=base_network, heads=heads,
                          head_conv_channel=64, pretrained=pretrained, classes=classes,
                          scale=4.0, topk=100, **kwargs)

def center_net_resnet18_v1b_dcnv2_coco(pretrained=False, pretrained_base=True, **kwargs):
    """Center net with resnet18_v1b base network with deformable v2 conv layer on coco dataset.

    Parameters
    ----------
    classes : iterable of str
        Names of custom foreground classes. `len(classes)` is the number of foreground classes.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized.

    Returns
    -------
    HybridBlock
        A CenterNet detection network.

    """
    from .deconv_resnet import resnet18_v1b_deconv_dcnv2
    from ...data import COCODetection
    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet18_v1b_deconv_dcnv2(pretrained=pretrained_base, **kwargs)
    heads = OrderedDict([
        ('heatmap', {'num_output': len(classes), 'bias': -2.19}), # use bias = -log((1 - 0.1) / 0.1)
        ('wh', {'num_output': 2}),
        ('reg', {'num_output': 2})
        ])
    return get_center_net('resnet18_v1b_dcnv2', 'coco', base_network=base_network, heads=heads,
                          head_conv_channel=64, pretrained=pretrained, classes=classes,
                          scale=4.0, topk=100, **kwargs)

def center_net_resnet50_v1b_voc(pretrained=False, pretrained_base=True, **kwargs):
    """Center net with resnet50_v1b base network on voc dataset.

    Parameters
    ----------
    classes : iterable of str
        Names of custom foreground classes. `len(classes)` is the number of foreground classes.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized.

    Returns
    -------
    HybridBlock
        A CenterNet detection network.

    """
    from .deconv_resnet import resnet50_v1b_deconv
    from ...data import VOCDetection
    classes = VOCDetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet50_v1b_deconv(pretrained=pretrained_base, **kwargs)
    heads = OrderedDict([
        ('heatmap', {'num_output': len(classes), 'bias': -2.19}), # use bias = -log((1 - 0.1) / 0.1)
        ('wh', {'num_output': 2}),
        ('reg', {'num_output': 2})
        ])
    return get_center_net('resnet50_v1b', 'voc', base_network=base_network, heads=heads,
                          head_conv_channel=64, pretrained=pretrained, classes=classes,
                          scale=4.0, topk=40, **kwargs)

def center_net_resnet50_v1b_dcnv2_voc(pretrained=False, pretrained_base=True, **kwargs):
    """Center net with resnet50_v1b base network with deformable conv layers on voc dataset.

    Parameters
    ----------
    classes : iterable of str
        Names of custom foreground classes. `len(classes)` is the number of foreground classes.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized.

    Returns
    -------
    HybridBlock
        A CenterNet detection network.

    """
    from .deconv_resnet import resnet50_v1b_deconv_dcnv2
    from ...data import VOCDetection
    classes = VOCDetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet50_v1b_deconv_dcnv2(pretrained=pretrained_base, **kwargs)
    heads = OrderedDict([
        ('heatmap', {'num_output': len(classes), 'bias': -2.19}), # use bias = -log((1 - 0.1) / 0.1)
        ('wh', {'num_output': 2}),
        ('reg', {'num_output': 2})
        ])
    return get_center_net('resnet50_v1b_dcnv2', 'voc', base_network=base_network, heads=heads,
                          head_conv_channel=64, pretrained=pretrained, classes=classes,
                          scale=4.0, topk=40, **kwargs)

def center_net_resnet50_v1b_coco(pretrained=False, pretrained_base=True, **kwargs):
    """Center net with resnet50_v1b base network on coco dataset.

    Parameters
    ----------
    classes : iterable of str
        Names of custom foreground classes. `len(classes)` is the number of foreground classes.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized.

    Returns
    -------
    HybridBlock
        A CenterNet detection network.

    """
    from .deconv_resnet import resnet50_v1b_deconv
    from ...data import COCODetection
    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet50_v1b_deconv(pretrained=pretrained_base, **kwargs)
    heads = OrderedDict([
        ('heatmap', {'num_output': len(classes), 'bias': -2.19}), # use bias = -log((1 - 0.1) / 0.1)
        ('wh', {'num_output': 2}),
        ('reg', {'num_output': 2})
        ])
    return get_center_net('resnet50_v1b', 'coco', base_network=base_network, heads=heads,
                          head_conv_channel=64, pretrained=pretrained, classes=classes,
                          scale=4.0, topk=100, **kwargs)

def center_net_resnet50_v1b_dcnv2_coco(pretrained=False, pretrained_base=True, **kwargs):
    """Center net with resnet50_v1b base network with deformable v2 conv layers on coco dataset.

    Parameters
    ----------
    classes : iterable of str
        Names of custom foreground classes. `len(classes)` is the number of foreground classes.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized.

    Returns
    -------
    HybridBlock
        A CenterNet detection network.

    """
    from .deconv_resnet import resnet50_v1b_deconv_dcnv2
    from ...data import COCODetection
    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet50_v1b_deconv_dcnv2(pretrained=pretrained_base, **kwargs)
    heads = OrderedDict([
        ('heatmap', {'num_output': len(classes), 'bias': -2.19}), # use bias = -log((1 - 0.1) / 0.1)
        ('wh', {'num_output': 2}),
        ('reg', {'num_output': 2})
        ])
    return get_center_net('resnet50_v1b_dcnv2', 'coco', base_network=base_network, heads=heads,
                          head_conv_channel=64, pretrained=pretrained, classes=classes,
                          scale=4.0, topk=100, **kwargs)

def center_net_resnet101_v1b_voc(pretrained=False, pretrained_base=True, **kwargs):
    """Center net with resnet101_v1b base network on voc dataset.

    Parameters
    ----------
    classes : iterable of str
        Names of custom foreground classes. `len(classes)` is the number of foreground classes.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized.

    Returns
    -------
    HybridBlock
        A CenterNet detection network.

    """
    from .deconv_resnet import resnet101_v1b_deconv
    from ...data import VOCDetection
    classes = VOCDetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet101_v1b_deconv(pretrained=pretrained_base, **kwargs)
    heads = OrderedDict([
        ('heatmap', {'num_output': len(classes), 'bias': -2.19}), # use bias = -log((1 - 0.1) / 0.1)
        ('wh', {'num_output': 2}),
        ('reg', {'num_output': 2})
        ])
    return get_center_net('resnet101_v1b', 'voc', base_network=base_network, heads=heads,
                          head_conv_channel=64, pretrained=pretrained, classes=classes,
                          scale=4.0, topk=40, **kwargs)

def center_net_resnet101_v1b_dcnv2_voc(pretrained=False, pretrained_base=True, **kwargs):
    """Center net with resnet101_v1b base network with deformable conv layers on voc dataset.

    Parameters
    ----------
    classes : iterable of str
        Names of custom foreground classes. `len(classes)` is the number of foreground classes.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized.

    Returns
    -------
    HybridBlock
        A CenterNet detection network.

    """
    from .deconv_resnet import resnet101_v1b_deconv_dcnv2
    from ...data import VOCDetection
    classes = VOCDetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet101_v1b_deconv_dcnv2(pretrained=pretrained_base, **kwargs)
    heads = OrderedDict([
        ('heatmap', {'num_output': len(classes), 'bias': -2.19}), # use bias = -log((1 - 0.1) / 0.1)
        ('wh', {'num_output': 2}),
        ('reg', {'num_output': 2})
        ])
    return get_center_net('resnet101_v1b_dcnv2', 'voc', base_network=base_network, heads=heads,
                          head_conv_channel=64, pretrained=pretrained, classes=classes,
                          scale=4.0, topk=40, **kwargs)

def center_net_resnet101_v1b_coco(pretrained=False, pretrained_base=True, **kwargs):
    """Center net with resnet101_v1b base network on coco dataset.

    Parameters
    ----------
    classes : iterable of str
        Names of custom foreground classes. `len(classes)` is the number of foreground classes.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized.

    Returns
    -------
    HybridBlock
        A CenterNet detection network.

    """
    from .deconv_resnet import resnet101_v1b_deconv
    from ...data import COCODetection
    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet101_v1b_deconv(pretrained=pretrained_base, **kwargs)
    heads = OrderedDict([
        ('heatmap', {'num_output': len(classes), 'bias': -2.19}), # use bias = -log((1 - 0.1) / 0.1)
        ('wh', {'num_output': 2}),
        ('reg', {'num_output': 2})
        ])
    return get_center_net('resnet101_v1b', 'coco', base_network=base_network, heads=heads,
                          head_conv_channel=64, pretrained=pretrained, classes=classes,
                          scale=4.0, topk=100, **kwargs)

def center_net_resnet101_v1b_dcnv2_coco(pretrained=False, pretrained_base=True, **kwargs):
    """Center net with resnet101_v1b base network with deformable v2 conv layers on coco dataset.

    Parameters
    ----------
    classes : iterable of str
        Names of custom foreground classes. `len(classes)` is the number of foreground classes.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized.

    Returns
    -------
    HybridBlock
        A CenterNet detection network.

    """
    from .deconv_resnet import resnet101_v1b_deconv_dcnv2
    from ...data import COCODetection
    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = resnet101_v1b_deconv_dcnv2(pretrained=pretrained_base, **kwargs)
    heads = OrderedDict([
        ('heatmap', {'num_output': len(classes), 'bias': -2.19}), # use bias = -log((1 - 0.1) / 0.1)
        ('wh', {'num_output': 2}),
        ('reg', {'num_output': 2})
        ])
    return get_center_net('resnet101_v1b_dcnv2', 'coco', base_network=base_network, heads=heads,
                          head_conv_channel=64, pretrained=pretrained, classes=classes,
                          scale=4.0, topk=100, **kwargs)

def center_net_dla34_voc(pretrained=False, pretrained_base=True, **kwargs):
    """Center net with dla34 base network on voc dataset.

    Parameters
    ----------
    classes : iterable of str
        Names of custom foreground classes. `len(classes)` is the number of foreground classes.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized.

    Returns
    -------
    HybridBlock
        A CenterNet detection network.

    """
    from .deconv_dla import dla34_deconv
    from ...data import VOCDetection
    classes = VOCDetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = dla34_deconv(pretrained=pretrained_base, **kwargs)
    heads = OrderedDict([
        ('heatmap', {'num_output': len(classes), 'bias': -2.19}), # use bias = -log((1 - 0.1) / 0.1)
        ('wh', {'num_output': 2}),
        ('reg', {'num_output': 2})
        ])
    return get_center_net('dla34', 'voc', base_network=base_network, heads=heads,
                          head_conv_channel=64, pretrained=pretrained, classes=classes,
                          scale=4.0, topk=40, **kwargs)

def center_net_dla34_dcnv2_voc(pretrained=False, pretrained_base=True, **kwargs):
    """Center net with dla34 base network with deformable conv layers on voc dataset.

    Parameters
    ----------
    classes : iterable of str
        Names of custom foreground classes. `len(classes)` is the number of foreground classes.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized.

    Returns
    -------
    HybridBlock
        A CenterNet detection network.

    """
    from .deconv_dla import dla34_deconv_dcnv2
    from ...data import VOCDetection
    classes = VOCDetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = dla34_deconv_dcnv2(pretrained=pretrained_base, **kwargs)
    heads = OrderedDict([
        ('heatmap', {'num_output': len(classes), 'bias': -2.19}), # use bias = -log((1 - 0.1) / 0.1)
        ('wh', {'num_output': 2}),
        ('reg', {'num_output': 2})
        ])
    return get_center_net('dla34_dcnv2', 'voc', base_network=base_network, heads=heads,
                          head_conv_channel=64, pretrained=pretrained, classes=classes,
                          scale=4.0, topk=40, **kwargs)

def center_net_dla34_coco(pretrained=False, pretrained_base=True, **kwargs):
    """Center net with dla34 base network on coco dataset.

    Parameters
    ----------
    classes : iterable of str
        Names of custom foreground classes. `len(classes)` is the number of foreground classes.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized.

    Returns
    -------
    HybridBlock
        A CenterNet detection network.

    """
    from .deconv_dla import dla34_deconv
    from ...data import COCODetection
    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = dla34_deconv(pretrained=pretrained_base, **kwargs)
    heads = OrderedDict([
        ('heatmap', {'num_output': len(classes), 'bias': -2.19}), # use bias = -log((1 - 0.1) / 0.1)
        ('wh', {'num_output': 2}),
        ('reg', {'num_output': 2})
        ])
    return get_center_net('dla34', 'coco', base_network=base_network, heads=heads,
                          head_conv_channel=64, pretrained=pretrained, classes=classes,
                          scale=4.0, topk=100, **kwargs)

def center_net_dla34_dcnv2_coco(pretrained=False, pretrained_base=True, **kwargs):
    """Center net with dla34 base network with deformable v2 conv layers on coco dataset.

    Parameters
    ----------
    classes : iterable of str
        Names of custom foreground classes. `len(classes)` is the number of foreground classes.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized.

    Returns
    -------
    HybridBlock
        A CenterNet detection network.

    """
    from .deconv_dla import dla34_deconv_dcnv2
    from ...data import COCODetection
    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = dla34_deconv_dcnv2(pretrained=pretrained_base, **kwargs)
    heads = OrderedDict([
        ('heatmap', {'num_output': len(classes), 'bias': -2.19}), # use bias = -log((1 - 0.1) / 0.1)
        ('wh', {'num_output': 2}),
        ('reg', {'num_output': 2})
        ])
    return get_center_net('dla34_dcnv2', 'coco', base_network=base_network, heads=heads,
                          head_conv_channel=64, pretrained=pretrained, classes=classes,
                          scale=4.0, topk=100, **kwargs)

def center_net_mobilenetv3_large_duc_voc(pretrained=False, pretrained_base=True, **kwargs):
    """Center net with mobilenetv3_large base network on voc dataset.

    Parameters
    ----------
    classes : iterable of str
        Names of custom foreground classes. `len(classes)` is the number of foreground classes.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized.

    Returns
    -------
    HybridBlock
        A CenterNet detection network.

    """
    from .duc_mobilenet import mobilenetv3_large_duc
    from ...data import VOCDetection
    classes = VOCDetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = mobilenetv3_large_duc(pretrained=pretrained_base, **kwargs)
    heads = OrderedDict([
        ('heatmap', {'num_output': len(classes), 'bias': -2.19}), # use bias = -log((1 - 0.1) / 0.1)
        ('wh', {'num_output': 2}),
        ('reg', {'num_output': 2})
        ])
    return get_center_net('mobilenetv3_large_duc', 'voc', base_network=base_network, heads=heads,
                          head_conv_channel=64, pretrained=pretrained, classes=classes,
                          scale=4.0, topk=40, **kwargs)

def center_net_mobilenetv3_small_duc_voc(pretrained=False, pretrained_base=True, **kwargs):
    """Center net with mobilenetv3_small base network with DUC layers on voc dataset.

    Parameters
    ----------
    classes : iterable of str
        Names of custom foreground classes. `len(classes)` is the number of foreground classes.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized.

    Returns
    -------
    HybridBlock
        A CenterNet detection network.

    """
    from .duc_mobilenet import mobilenetv3_small_duc
    from ...data import VOCDetection
    classes = VOCDetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = mobilenetv3_small_duc(pretrained=pretrained_base, **kwargs)
    heads = OrderedDict([
        ('heatmap', {'num_output': len(classes), 'bias': -2.19}), # use bias = -log((1 - 0.1) / 0.1)
        ('wh', {'num_output': 2}),
        ('reg', {'num_output': 2})
        ])
    return get_center_net('mobilenetv3_small_duc', 'voc', base_network=base_network, heads=heads,
                          head_conv_channel=64, pretrained=pretrained, classes=classes,
                          scale=4.0, topk=40, **kwargs)

def center_net_mobilenetv3_large_duc_coco(pretrained=False, pretrained_base=True, **kwargs):
    """Center net with mobilenetv3_large base network on coco dataset.

    Parameters
    ----------
    classes : iterable of str
        Names of custom foreground classes. `len(classes)` is the number of foreground classes.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized.

    Returns
    -------
    HybridBlock
        A CenterNet detection network.

    """
    from .duc_mobilenet import mobilenetv3_large_duc
    from ...data import COCODetection
    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = mobilenetv3_large_duc(pretrained=pretrained_base, **kwargs)
    heads = OrderedDict([
        ('heatmap', {'num_output': len(classes), 'bias': -2.19}), # use bias = -log((1 - 0.1) / 0.1)
        ('wh', {'num_output': 2}),
        ('reg', {'num_output': 2})
        ])
    return get_center_net('mobilenetv3_large_duc', 'coco', base_network=base_network, heads=heads,
                          head_conv_channel=64, pretrained=pretrained, classes=classes,
                          scale=4.0, topk=100, **kwargs)

def center_net_mobilenetv3_small_duc_coco(pretrained=False, pretrained_base=True, **kwargs):
    """Center net with mobilenetv3_small base network with DUC layers on coco dataset.

    Parameters
    ----------
    classes : iterable of str
        Names of custom foreground classes. `len(classes)` is the number of foreground classes.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized.

    Returns
    -------
    HybridBlock
        A CenterNet detection network.

    """
    from .duc_mobilenet import mobilenetv3_small_duc
    from ...data import COCODetection
    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = mobilenetv3_small_duc(pretrained=pretrained_base, **kwargs)
    heads = OrderedDict([
        ('heatmap', {'num_output': len(classes), 'bias': -2.19}), # use bias = -log((1 - 0.1) / 0.1)
        ('wh', {'num_output': 2}),
        ('reg', {'num_output': 2})
        ])
    return get_center_net('mobilenetv3_small_duc', 'coco', base_network=base_network, heads=heads,
                          head_conv_channel=64, pretrained=pretrained, classes=classes,
                          scale=4.0, topk=100, **kwargs)
