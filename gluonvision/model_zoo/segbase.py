"""Base Model for Semantic Segmentation"""
from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock

from . import dilatedresnet

class SegBaseModel(HybridBlock):
    r"""Base Model for Semantic Segmentation

    Parameters
    ----------
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    """
    # pylint : disable=arguments-differ
    def __init__(self, backbone='resnet50', norm_layer=nn.BatchNorm, **kwargs):
        super(SegBaseModel, self).__init__(**kwargs)
        with self.name_scope():
            if backbone == 'resnet50':
                pretrained = dilatedresnet.dilated_resnet50(pretrained=True, norm_layer=norm_layer)
            elif backbone == 'resnet101':
                pretrained = dilatedresnet.dilated_resnet101(pretrained=True, norm_layer=norm_layer)
            elif backbone == 'resnet152':
                pretrained = dilatedresnet.dilated_resnet152(pretrained=True, norm_layer=norm_layer)
            else:
                raise RuntimeError('unknown backbone: {}'.format(backbone))

            self.pretrained = nn.HybridSequential()
            with self.pretrained.name_scope():
                for layer in pretrained.features:
                    self.pretrained.add(layer)

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, x):
        pass
