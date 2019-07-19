from mxnet import gluon, nd, init
from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock
from ..inception import inception_v3

__all__ = ['inceptionv3_ucf101', 'ActionRecInceptionV3', 'ActionRecInceptionV3TSN']

def inceptionv3_ucf101(nclass, pretrained, tsn, partial_bn, **kwargs):
    if tsn:
        model = ActionRecInceptionV3TSN(nclass=nclass, pretrained_base=pretrained, partial_bn=partial_bn)
    else:
        model = ActionRecInceptionV3(nclass=nclass, pretrained_base=pretrained, partial_bn=partial_bn)
    return model

class ActionRecInceptionV3(HybridBlock):
    r"""InceptionV3 model for video action recognition

    Parameters
    ----------
    nclass: int, number of classes
    pretrained_base: bool, load pre-trained weights or not

    Input: a single image 
    Output: a single predicted action label
    """
    # pylint : disable=arguments-differ
    def __init__(self, nclass, pretrained_base=True, partial_bn=False, **kwargs):
        super(ActionRecInceptionV3, self).__init__()

        pretrained = inception_v3(pretrained=pretrained_base, partial_bn=partial_bn, **kwargs)
        self.features = pretrained.features
        self.features[19]._rate = 0.8
        self.output = nn.Dense(units=nclass, in_units=2048, weight_initializer=init.Normal(sigma=0.001))
        self.output.initialize()

    def hybrid_forward(self, F, x):
    	x = self.features(x)
    	x = self.output(x)
    	return x

class ActionRecInceptionV3TSN(HybridBlock):
    r"""InceptionV3 model with temporal segments for video action recognition

    Parameters
    ----------
    nclass: int, number of classes
    pretrained_base: bool, load pre-trained weights or not

    Input: N images from N segments in a single video
    Output: a single predicted action label
    """
    # pylint : disable=arguments-differ
    def __init__(self, nclass, pretrained_base=True, partial_bn=False, **kwargs):
        super(ActionRecInceptionV3TSN, self).__init__()

        pretrained = inception_v3(pretrained=pretrained_base, partial_bn=partial_bn, **kwargs)
        self.features = pretrained.features
        self.features[19]._rate = 0.8
        self.output = nn.Dense(units=nclass, in_units=2048, weight_initializer=init.Normal(sigma=0.001))
        self.output.initialize()

    # def forward(self, x):
    #     b, c, h, w = x.shape
    #     reshape_data = x.reshape((-1, 3, h, w))
    #     feat = self.features(reshape_data)
    #     out = self.output(feat)
    #     _, class_dim = out.shape
    #     reshape_out = out.reshape((b, 3, class_dim))
    #     consensus_out = reshape_out.mean(axis=1)
    #     return consensus_out

    def hybrid_forward(self, F, x):
        reshape_data = x.reshape((-1, 3, 299, 299))
        feat = self.features(reshape_data)
        out = self.output(feat)
        reshape_out = out.reshape((-1, 3, 101))
        consensus_out = reshape_out.mean(axis=1)
        return consensus_out




