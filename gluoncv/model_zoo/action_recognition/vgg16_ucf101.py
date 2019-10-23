# pylint: disable=line-too-long,too-many-lines,missing-docstring,arguments-differ,unused-argument
import mxnet as mx
from mxnet import init
from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock
from ...nn.block import Consensus
from ..vgg import vgg16

__all__ = ['vgg16_ucf101']

class ActionRecVGG16(HybridBlock):
    r"""VGG16 model for video action recognition
    Limin Wang, etal, Towards Good Practices for Very Deep Two-Stream ConvNets, arXiv 2015
    https://arxiv.org/abs/1507.02159

    Parameters
    ----------
    nclass : int, number of classes
    pretrained_base : bool, load pre-trained weights or not

    Input: a single video frame
    Output: a single predicted action label
    """
    def __init__(self, nclass, pretrained_base=True,
                 dropout_ratio=0.9, init_std=0.001, feat_dim=4096,
                 num_segments=1, num_crop=1, **kwargs):
        super(ActionRecVGG16, self).__init__()
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.num_segments = num_segments
        self.num_crop = num_crop
        self.feat_dim = feat_dim

        pretrained_model = vgg16(pretrained=pretrained_base, **kwargs)
        self.features = pretrained_model.features
        def update_dropout_ratio(block):
            if isinstance(block, nn.basic_layers.Dropout):
                block._rate = self.dropout_ratio
        self.apply(update_dropout_ratio)
        self.output = nn.Dense(units=nclass, in_units=self.feat_dim,
                               weight_initializer=init.Normal(sigma=self.init_std))
        self.output.initialize()

    def hybrid_forward(self, F, x):
        x = self.features(x)

        # segmental consensus
        x = F.reshape(x, shape=(-1, self.num_segments * self.num_crop, self.feat_dim))
        x = F.mean(x, axis=1)

        x = self.output(x)
        return x

class ActionRecVGG16TSN(HybridBlock):
    r"""VGG16 model with temporal segments for video action recognition
    Limin Wang, etal, Temporal Segment Networks: Towards Good Practices for Deep Action Recognition, ECCV 2016
    https://arxiv.org/abs/1608.00859

    Parameters
    ----------
    nclass : int, number of classes
    pretrained_base : bool, load pre-trained weights or not

    Input: N images from N segments in a single video
    Output: a single predicted action label
    """
    def __init__(self, nclass, pretrained_base=True, num_segments=3, **kwargs):
        super(ActionRecVGG16TSN, self).__init__()

        self.basenet = ActionRecVGG16(nclass=nclass, pretrained_base=pretrained_base)
        self.tsn_consensus = Consensus(nclass=nclass, num_segments=num_segments)

    def hybrid_forward(self, F, x):
        pred = self.basenet(x)
        consensus_out = self.tsn_consensus(pred)
        return consensus_out

def vgg16_ucf101(nclass=101, pretrained=False, pretrained_base=True,
                 tsn=False, num_segments=1, num_crop=1,
                 ctx=mx.cpu(), root='~/.mxnet/models', **kwargs):
    if tsn:
        model = ActionRecVGG16TSN(nclass=nclass,
                                  num_segments=num_segments)
    else:
        model = ActionRecVGG16(nclass=nclass,
                               pretrained_base=pretrained_base,
                               num_segments=num_segments,
                               num_crop=num_crop)

    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('vgg16_ucf101',
                                             tag=pretrained, root=root))
        from ...data import UCF101Attr
        attrib = UCF101Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)
    return model
