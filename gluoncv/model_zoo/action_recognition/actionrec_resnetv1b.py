# pylint: disable=line-too-long,too-many-lines,missing-docstring,arguments-differ,unused-argument
import mxnet as mx
from mxnet import init
from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock
from ...nn.block import Consensus
from ..resnetv1b import resnet18_v1b, resnet34_v1b, resnet50_v1b, resnet101_v1b, resnet152_v1b

__all__ = ['resnet18_v1b_sthsthv2', 'resnet34_v1b_sthsthv2', 'resnet50_v1b_sthsthv2',
           'resnet101_v1b_sthsthv2', 'resnet152_v1b_sthsthv2', 'resnet18_v1b_kinetics400',
           'resnet34_v1b_kinetics400', 'resnet50_v1b_kinetics400', 'resnet101_v1b_kinetics400',
           'resnet152_v1b_kinetics400']

class ActionRecResNetV1b(HybridBlock):
    r"""ResNet models for video action recognition

    Parameters
    ----------
    depth : int, number of layers in a ResNet model
    nclass : int, number of classes
    pretrained_base : bool, load pre-trained weights or not
    dropout_ratio : float, add a dropout layer with a ratio p to prevent overfitting
    init_std : float, initialization of the standard deviation value of
               the output (classification) layer

    Input: a single image
    Output: a single predicted action label
    """
    def __init__(self, depth, nclass, pretrained_base=True,
                 dropout_ratio=0.5, init_std=0.01,
                 partial_bn=False, **kwargs):
        super(ActionRecResNetV1b, self).__init__()

        if depth == 18:
            pretrained_model = resnet18_v1b(pretrained=pretrained_base, **kwargs)
            self.expansion = 1
        elif depth == 34:
            pretrained_model = resnet34_v1b(pretrained=pretrained_base, **kwargs)
            self.expansion = 1
        elif depth == 50:
            pretrained_model = resnet50_v1b(pretrained=pretrained_base, **kwargs)
            self.expansion = 4
        elif depth == 101:
            pretrained_model = resnet101_v1b(pretrained=pretrained_base, **kwargs)
            self.expansion = 4
        elif depth == 152:
            pretrained_model = resnet152_v1b(pretrained=pretrained_base, **kwargs)
            self.expansion = 4
        else:
            print('No such ResNet configuration for depth=%d' % (depth))

        self.dropout_ratio = dropout_ratio
        self.init_std = init_std

        self.conv1 = pretrained_model.conv1
        self.bn1 = pretrained_model.bn1
        self.relu = pretrained_model.relu
        self.maxpool = pretrained_model.maxpool
        self.layer1 = pretrained_model.layer1
        self.layer2 = pretrained_model.layer2
        self.layer3 = pretrained_model.layer3
        self.layer4 = pretrained_model.layer4
        self.avgpool = pretrained_model.avgpool
        self.flat = pretrained_model.flat
        self.drop = nn.Dropout(rate=self.dropout_ratio)
        self.output = nn.Dense(units=nclass, in_units=512 * self.expansion,
                               weight_initializer=init.Normal(sigma=self.init_std))
        self.output.initialize()

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flat(x)
        x = self.drop(x)
        x = self.output(x)
        return x

class ActionRecResNetV1bTSN(HybridBlock):
    r"""ResNet models with temporal segments for video action recognition

    Parameters
    ----------
    depth : int, number of layers in a ResNet model
    nclass : int, number of classes
    pretrained_base : bool, load pre-trained weights or not
    dropout_ratio : float, add a dropout layer with a ratio p to prevent overfitting
    init_std : float, initialization of the standard deviation value of
               the output (classification) layer

    Input: N images from N segments in a single video
    Output: a single predicted action label
    """
    def __init__(self, depth, nclass, pretrained_base=True,
                 partial_bn=False, num_segments=3,
                 dropout_ratio=0.5, init_std=0.01,
                 **kwargs):
        super(ActionRecResNetV1bTSN, self).__init__()

        self.basenet = ActionRecResNetV1b(depth=depth,
                                          nclass=nclass,
                                          pretrained_base=pretrained_base,
                                          partial_bn=partial_bn)
        self.tsn_consensus = Consensus(nclass=nclass, num_segments=num_segments)

    def hybrid_forward(self, F, x):
        pred = self.basenet(x)
        consensus_out = self.tsn_consensus(pred)
        return consensus_out

def resnet18_v1b_sthsthv2(nclass=174, pretrained=False, tsn=False, partial_bn=False,
                          num_segments=1, root='~/.mxnet/models', ctx=mx.cpu(), **kwargs):
    if tsn:
        model = ActionRecResNetV1bTSN(depth=18,
                                      nclass=nclass,
                                      partial_bn=partial_bn,
                                      num_segments=num_segments)
    else:
        model = ActionRecResNetV1b(depth=18,
                                   nclass=nclass,
                                   partial_bn=partial_bn)

    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('resnet18_v1b_sthsthv2',
                                             tag=pretrained, root=root))
        from ...data import SomethingSomethingV2Attr
        attrib = SomethingSomethingV2Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)
    return model

def resnet34_v1b_sthsthv2(nclass=174, pretrained=False, tsn=False, partial_bn=False,
                          num_segments=1, root='~/.mxnet/models', ctx=mx.cpu(), **kwargs):
    if tsn:
        model = ActionRecResNetV1bTSN(depth=34,
                                      nclass=nclass,
                                      partial_bn=partial_bn,
                                      num_segments=num_segments)
    else:
        model = ActionRecResNetV1b(depth=34,
                                   nclass=nclass,
                                   partial_bn=partial_bn)

    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('resnet34_v1b_sthsthv2',
                                             tag=pretrained, root=root))
        from ...data import SomethingSomethingV2Attr
        attrib = SomethingSomethingV2Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)
    return model

def resnet50_v1b_sthsthv2(nclass=174, pretrained=False, tsn=False, partial_bn=False,
                          num_segments=1, root='~/.mxnet/models', ctx=mx.cpu(), **kwargs):
    if tsn:
        model = ActionRecResNetV1bTSN(depth=50,
                                      nclass=nclass,
                                      partial_bn=partial_bn,
                                      num_segments=num_segments)
    else:
        model = ActionRecResNetV1b(depth=50,
                                   nclass=nclass,
                                   partial_bn=partial_bn)

    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('resnet50_v1b_sthsthv2',
                                             tag=pretrained, root=root))
        from ...data import SomethingSomethingV2Attr
        attrib = SomethingSomethingV2Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)
    return model

def resnet101_v1b_sthsthv2(nclass=174, pretrained=False, tsn=False, partial_bn=False,
                           num_segments=1, root='~/.mxnet/models', ctx=mx.cpu(), **kwargs):
    if tsn:
        model = ActionRecResNetV1bTSN(depth=101,
                                      nclass=nclass,
                                      partial_bn=partial_bn,
                                      num_segments=num_segments)
    else:
        model = ActionRecResNetV1b(depth=101,
                                   nclass=nclass,
                                   partial_bn=partial_bn)

    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('resnet101_v1b_sthsthv2',
                                             tag=pretrained, root=root))
        from ...data import SomethingSomethingV2Attr
        attrib = SomethingSomethingV2Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)
    return model

def resnet152_v1b_sthsthv2(nclass=174, pretrained=False, tsn=False, partial_bn=False,
                           num_segments=1, root='~/.mxnet/models', ctx=mx.cpu(), **kwargs):
    if tsn:
        model = ActionRecResNetV1bTSN(depth=152,
                                      nclass=nclass,
                                      partial_bn=partial_bn,
                                      num_segments=num_segments)
    else:
        model = ActionRecResNetV1b(depth=152,
                                   nclass=nclass,
                                   partial_bn=partial_bn)

    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('resnet152_v1b_sthsthv2',
                                             tag=pretrained, root=root))
        from ...data import SomethingSomethingV2Attr
        attrib = SomethingSomethingV2Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)
    return model

def resnet18_v1b_kinetics400(nclass=400, pretrained=False, tsn=False, partial_bn=False,
                             num_segments=1, root='~/.mxnet/models', ctx=mx.cpu(), **kwargs):
    if tsn:
        model = ActionRecResNetV1bTSN(depth=18,
                                      nclass=nclass,
                                      partial_bn=partial_bn,
                                      num_segments=num_segments)
    else:
        model = ActionRecResNetV1b(depth=18,
                                   nclass=nclass,
                                   partial_bn=partial_bn)

    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('resnet18_v1b_kinetics400',
                                             tag=pretrained, root=root))
        from ...data import Kinetics400Attr
        attrib = Kinetics400Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)
    return model

def resnet34_v1b_kinetics400(nclass=400, pretrained=False, tsn=False, partial_bn=False,
                             num_segments=1, root='~/.mxnet/models', ctx=mx.cpu(), **kwargs):
    if tsn:
        model = ActionRecResNetV1bTSN(depth=34,
                                      nclass=nclass,
                                      partial_bn=partial_bn,
                                      num_segments=num_segments)
    else:
        model = ActionRecResNetV1b(depth=34,
                                   nclass=nclass,
                                   partial_bn=partial_bn)

    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('resnet34_v1b_kinetics400',
                                             tag=pretrained, root=root))
        from ...data import Kinetics400Attr
        attrib = Kinetics400Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)
    return model

def resnet50_v1b_kinetics400(nclass=400, pretrained=False, tsn=False, partial_bn=False,
                             num_segments=1, root='~/.mxnet/models', ctx=mx.cpu(), **kwargs):
    if tsn:
        model = ActionRecResNetV1bTSN(depth=50,
                                      nclass=nclass,
                                      partial_bn=partial_bn,
                                      num_segments=num_segments)
    else:
        model = ActionRecResNetV1b(depth=50,
                                   nclass=nclass,
                                   partial_bn=partial_bn)

    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('resnet50_v1b_kinetics400',
                                             tag=pretrained, root=root))
        from ...data import Kinetics400Attr
        attrib = Kinetics400Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)
    return model

def resnet101_v1b_kinetics400(nclass=400, pretrained=False, tsn=False, partial_bn=False,
                              num_segments=1, root='~/.mxnet/models', ctx=mx.cpu(), **kwargs):
    if tsn:
        model = ActionRecResNetV1bTSN(depth=101,
                                      nclass=nclass,
                                      partial_bn=partial_bn,
                                      num_segments=num_segments)
    else:
        model = ActionRecResNetV1b(depth=101,
                                   nclass=nclass,
                                   partial_bn=partial_bn)

    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('resnet101_v1b_kinetics400',
                                             tag=pretrained, root=root))
        from ...data import Kinetics400Attr
        attrib = Kinetics400Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)
    return model

def resnet152_v1b_kinetics400(nclass=400, pretrained=False, tsn=False, partial_bn=False,
                              num_segments=1, root='~/.mxnet/models', ctx=mx.cpu(), **kwargs):
    if tsn:
        model = ActionRecResNetV1bTSN(depth=152,
                                      nclass=nclass,
                                      partial_bn=partial_bn,
                                      num_segments=num_segments)
    else:
        model = ActionRecResNetV1b(depth=152,
                                   nclass=nclass,
                                   partial_bn=partial_bn)

    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('resnet152_v1b_kinetics400',
                                             tag=pretrained, root=root))
        from ...data import Kinetics400Attr
        attrib = Kinetics400Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)
    return model
