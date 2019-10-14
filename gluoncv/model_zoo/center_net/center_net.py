"""CenterNet object detector: Objects as Points, https://arxiv.org/abs/1904.07850"""
from __future__ import absolute_import

import os
import warnings
from collections import OrderedDict

import mxnet as mx
from mxnet.context import cpu
from mxnet.gluon import nn
from mxnet import autograd
from ...nn.coder import CenterNetDecoder

__all__ = ['CenterNet', 'get_center_net',
           'center_net_resnet18_v1b_voc', 'center_net_resnet18_v1b_coco']

class CenterNet(nn.HybridBlock):
    def __init__(self, base_network, heads, classes,
                 head_conv_channel=0, scale=4.0, topk=40, flip_test=False, **kwargs):
        if 'norm_layer' in kwargs:
            kwargs.pop('norm_layer')
        if 'norm_kwargs' in kwargs:
            kwargs.pop('norm_kwargs')
        super(CenterNet, self).__init__(**kwargs)
        assert isinstance(heads, OrderedDict), \
            "Expecting heads to be a OrderedDict of {head_name: # outputs} per head, given {}" \
            .format(type(heads))
        self.classes = classes
        self.scale = scale
        self.flip_test = flip_test
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
                    head.add(nn.Conv2D(head_conv_channel, kernel_size=3, padding=1, use_bias=True,
                             weight_initializer=weight_initializer, bias_initializer='zeros'))
                    head.add(nn.Activation('relu'))
                head.add(nn.Conv2D(num_output, kernel_size=1, strides=1, padding=0, use_bias=True,
                                   weight_initializer=weight_initializer,
                                   bias_initializer=mx.init.Constant(bias)))

                self.heads.add(head)


    def hybrid_forward(self, F, x):
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
        keep = self.heatmap_nms(heatmap) == heatmap
        results = self.decoder(keep * heatmap, out[1], out[2])
        return results

def get_center_net(name, dataset, pretrained=False, ctx=mx.cpu(),
                   root=os.path.join('~', '.mxnet', 'models'), **kwargs):
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
    from .deconv_resnet import deconv_resnet18_v1b
    from ...data import VOCDetection
    classes = VOCDetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = deconv_resnet18_v1b(pretrained=pretrained_base, **kwargs)
    heads = OrderedDict([
        ('heatmap', {'num_output': len(classes), 'bias': -2.19}),  # use bias = -log((1 - 0.1) / 0.1)
        ('wh', {'num_output': 2}),
        ('reg', {'num_output': 2})
        ])
    return get_center_net('resnet18_v1b', 'voc', base_network=base_network, heads=heads,
                          head_conv_channel=64, pretrained=pretrained, classes=classes,
                          scale=4.0, topk=40, **kwargs)

def center_net_resnet18_v1b_coco(pretrained=False, pretrained_base=True, **kwargs):
    from .deconv_resnet import deconv_resnet18_v1b
    from ...data import COCODetection
    classes = COCODetection.CLASSES
    pretrained_base = False if pretrained else pretrained_base
    base_network = deconv_resnet18_v1b(pretrained=pretrained_base, **kwargs)
    heads = OrderedDict([
        ('heatmap', {'num_output': len(classes), 'bias': -2.19}),  # use bias = -log((1 - 0.1) / 0.1)
        ('wh', {'num_output': 2}),
        ('reg', {'num_output': 2})
        ])
    return get_center_net('resnet18_v1b', 'coco', base_network=base_network, heads=heads,
                          head_conv_channel=64, pretrained=pretrained, classes=classes,
                          scale=4.0, topk=40, **kwargs)
