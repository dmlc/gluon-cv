from mxnet import gluon, nd, init
from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock
from ..vgg import vgg16

__all__ = ['vgg16_ucf101', 'ActionRecVGG16', 'ActionRecVGG16TSN']

def vgg16_ucf101(nclass, pretrained, tsn, **kwargs):
    if tsn:
        model = ActionRecVGG16TSN(nclass=nclass, pretrained_base=pretrained)
    else:
        model = ActionRecVGG16(nclass=nclass, pretrained_base=pretrained)
    return model

class ActionRecVGG16(HybridBlock):
    r"""VGG16 model for video action recognition

    Parameters
    ----------
    nclass: int, number of classes
    pretrained_base: bool, load pre-trained weights or not

    Input: a single image 
    Output: a single predicted action label
    """
    # pylint : disable=arguments-differ
    def __init__(self, nclass, pretrained_base=True, **kwargs):
        super(ActionRecVGG16, self).__init__()

        pretrained = vgg16(pretrained=pretrained_base, **kwargs)
        self.features = pretrained.features
        # set high dropout ratio as in [1] to prevent overfitting
        # [1] Limin Wang, etal, Towards Good Practices for Very Deep Two-Stream ConvNets, arXiv 2015
        self.features[32]._rate = 0.9
        self.features[34]._rate = 0.9
        self.output = nn.Dense(units=nclass, in_units=4096, weight_initializer=init.Normal(sigma=0.001))
        self.output.initialize()

    def hybrid_forward(self, F, x):
    	x = self.features(x)
    	x = self.output(x)
    	return x

class ActionRecVGG16TSN(HybridBlock):
    r"""VGG16 model with temporal segments for video action recognition

    Parameters
    ----------
    nclass: int, number of classes
    pretrained_base: bool, load pre-trained weights or not

    Input: N images from N segments in a single video
    Output: a single predicted action label
    """
    # pylint : disable=arguments-differ
    def __init__(self, nclass, pretrained_base=True, **kwargs):
        super(ActionRecVGG16TSN, self).__init__()

        pretrained = vgg16(pretrained=pretrained_base, **kwargs)
        self.features = pretrained.features
        self.features[32]._rate = 0.9
        self.features[34]._rate = 0.9
        self.output = nn.Dense(units=nclass, in_units=4096, weight_initializer=init.Normal(sigma=0.001))
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
        reshape_data = x.reshape((-1, 3, 224, 224))
        feat = self.features(reshape_data)
        out = self.output(feat)
        reshape_out = out.reshape((-1, 3, 101))
        consensus_out = reshape_out.mean(axis=1)
        return consensus_out




