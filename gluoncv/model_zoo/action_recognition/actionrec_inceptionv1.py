# pylint: disable=line-too-long,too-many-lines,missing-docstring,arguments-differ,unused-argument
import mxnet as mx
from mxnet import init
from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock
from gluoncv.model_zoo.googlenet import googlenet

__all__ = ['inceptionv1_ucf101', 'inceptionv1_hmdb51', 'inceptionv1_kinetics400',
           'inceptionv1_sthsthv2']

class ActionRecInceptionV1(HybridBlock):
    r"""Inception v1 model for video action recognition
    Christian Szegedy, etal, Going Deeper with Convolutions, CVPR 2015
    https://arxiv.org/abs/1409.4842
    Limin Wang, etal, Towards Good Practices for Very Deep Two-Stream ConvNets, arXiv 2015
    https://arxiv.org/abs/1507.02159
    Limin Wang, etal, Temporal Segment Networks: Towards Good Practices for Deep Action Recognition, ECCV 2016
    https://arxiv.org/abs/1608.00859

    Parameters
    ----------
    nclass : int, number of classes
    pretrained_base : bool, load pre-trained weights or not
    dropout_ratio : float, add a dropout layer to prevent overfitting on small datasets, such as UCF101
    init_std : float, standard deviation value when initialize the last classification layer
    feat_dim : int, feature dimension. Default is 4096 for VGG16 network
    num_segments : int, number of segments used
    num_crop : int, number of crops used during evaluation. Default choice is 1, 3 or 10

    Input: a single video frame or N images from N segments when num_segments > 1
    Output: a single predicted action label
    """
    def __init__(self, nclass, pretrained_base=True,
                 partial_bn=True, dropout_ratio=0.8, init_std=0.001,
                 feat_dim=1024, num_segments=1, num_crop=1, **kwargs):
        super(ActionRecInceptionV1, self).__init__()
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.num_segments = num_segments
        self.num_crop = num_crop
        self.feat_dim = feat_dim

        pretrained_model = googlenet(pretrained=pretrained_base, partial_bn=partial_bn, **kwargs)

        self.conv1 = pretrained_model.conv1
        self.maxpool1 = pretrained_model.maxpool1

        self.conv2 = pretrained_model.conv2
        self.conv3 = pretrained_model.conv3
        self.maxpool2 = pretrained_model.maxpool2

        self.inception3a = pretrained_model.inception3a
        self.inception3b = pretrained_model.inception3b
        self.maxpool3 = pretrained_model.maxpool3

        self.inception4a = pretrained_model.inception4a
        self.inception4b = pretrained_model.inception4b
        self.inception4c = pretrained_model.inception4c
        self.inception4d = pretrained_model.inception4d
        self.inception4e = pretrained_model.inception4e
        self.maxpool4 = pretrained_model.maxpool4

        self.inception5a = pretrained_model.inception5a
        self.inception5b = pretrained_model.inception5b

        self.avgpool = nn.AvgPool2D(pool_size=7)
        self.dropout = nn.Dropout(self.dropout_ratio)
        self.output = nn.Dense(units=nclass, in_units=self.feat_dim,
                               weight_initializer=init.Normal(sigma=self.init_std))
        self.output.initialize()

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = self.dropout(x)

        # segmental consensus
        x = F.reshape(x, shape=(-1, self.num_segments * self.num_crop, self.feat_dim))
        x = F.mean(x, axis=1)

        x = self.output(x)
        return x

def inceptionv1_ucf101(nclass=101, pretrained=False, pretrained_base=True,
                       use_tsn=False, num_segments=1, num_crop=1, partial_bn=True,
                       ctx=mx.cpu(), root='~/.mxnet/models', **kwargs):
    model = ActionRecInceptionV1(nclass=nclass,
                                 partial_bn=partial_bn,
                                 pretrained_base=pretrained_base,
                                 num_segments=num_segments,
                                 num_crop=num_crop,
                                 dropout_ratio=0.8,
                                 init_std=0.001)

    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('inceptionv1_ucf101',
                                             tag=pretrained, root=root))
        from ...data import UCF101Attr
        attrib = UCF101Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)
    return model

def inceptionv1_hmdb51(nclass=51, pretrained=False, pretrained_base=True,
                       use_tsn=False, num_segments=1, num_crop=1, partial_bn=True,
                       ctx=mx.cpu(), root='~/.mxnet/models', **kwargs):
    model = ActionRecInceptionV1(nclass=nclass,
                                 partial_bn=partial_bn,
                                 pretrained_base=pretrained_base,
                                 num_segments=num_segments,
                                 num_crop=num_crop,
                                 dropout_ratio=0.8,
                                 init_std=0.001)

    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('inceptionv1_hmdb51',
                                             tag=pretrained, root=root))
        from ...data import HMDB51Attr
        attrib = HMDB51Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)
    return model

def inceptionv1_kinetics400(nclass=400, pretrained=False, pretrained_base=True,
                            tsn=False, num_segments=1, num_crop=1, partial_bn=True,
                            ctx=mx.cpu(), root='~/.mxnet/models', **kwargs):
    model = ActionRecInceptionV1(nclass=nclass,
                                 partial_bn=partial_bn,
                                 pretrained_base=pretrained_base,
                                 num_segments=num_segments,
                                 num_crop=num_crop,
                                 dropout_ratio=0.5,
                                 init_std=0.01)

    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('inceptionv1_kinetics400',
                                             tag=pretrained, root=root))
        from ...data import Kinetics400Attr
        attrib = Kinetics400Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)
    return model

def inceptionv1_sthsthv2(nclass=174, pretrained=False, pretrained_base=True,
                         tsn=False, num_segments=1, num_crop=1, partial_bn=True,
                         ctx=mx.cpu(), root='~/.mxnet/models', **kwargs):
    model = ActionRecInceptionV1(nclass=nclass,
                                 partial_bn=partial_bn,
                                 pretrained_base=pretrained_base,
                                 num_segments=num_segments,
                                 num_crop=num_crop,
                                 dropout_ratio=0.5,
                                 init_std=0.01)

    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('inceptionv1_sthsthv2',
                                             tag=pretrained, root=root))
        from ...data import SomethingSomethingV2Attr
        attrib = SomethingSomethingV2Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)
    return model
