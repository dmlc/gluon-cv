"""Encoder module of Monodepth2
Code partially borrowed from
https://github.com/nianticlabs/monodepth2/blob/master/networks/resnet_encoder.py
"""
from __future__ import absolute_import, division, print_function

import numpy as np

from mxnet.gluon import nn
from mxnet.context import cpu
from ...model_zoo.resnetv1b import \
    resnet18_v1b, resnet34_v1b, resnet50_v1s, resnet101_v1s, resnet152_v1s


# def resnet_multiimage_input(backbone, pretrained=False, num_input_images=1):
#     """Constructs a ResNet model.
#     Args:
#         backbone (str): resnet backbone. Must be resnet18 or resnet18
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#         num_input_images (int): Number of frames stacked as input
#     """
#     assert backbone in ['resnet18', 'resnet18'], "Can only run with 18 or 50 layer resnet"
#
#     blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
#     block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
#     model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)
#
#     if pretrained:
#         loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
#         loaded['conv1.weight'] = torch.cat(
#             [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
#         model.load_state_dict(loaded)
#     return model


class ResnetEncoder(nn.HybridBlock):
    r"""Encoder of Monodepth2

    Parameters
    ----------
    backbone : string
        Pre-trained dilated backbone network type ('resnet18', 'resnet34', 'resnet50',
        'resnet101' or 'resnet152').
    pretrained : bool or str
        Refers to if the backbone is pretrained or not. If `True`,
        model weights of a model that was trained on ImageNet is loaded.
    num_input_images : int
        The number of input sequences. 1 for depth encoder, larger than 1 for pose encoder.
        (Default: 1)
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    def __init__(self, backbone, pretrained, num_input_images=1,
                 root='~/.mxnet/models', ctx=cpu(), **kwargs):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {'resnet18': resnet18_v1b,
                   'resnet34': resnet34_v1b,
                   'resnet50': resnet50_v1s,
                   'resnet101': resnet101_v1s,
                   'resnet152': resnet152_v1s}

        num_layers = {'resnet18': 18,
                      'resnet34': 34,
                      'resnet50': 50,
                      'resnet101': 101,
                      'resnet152': 152}

        if backbone not in resnets:
            raise ValueError("{} is not a valid resnet".format(backbone))

        if num_input_images > 1:
            self.encoder = resnets[backbone](pretrained=False, ctx=ctx, **kwargs)
            if pretrained:
                from ..model_store import get_model_file
                self.encoder.load_parameters(
                    get_model_file('resnet%d_v%db' % (num_layers[backbone], 1),
                                   tag=pretrained, root=root), ctx=ctx)
                from ...data import ImageNet1kAttr
                attrib = ImageNet1kAttr()
                self.encoder.synset = attrib.synset
                self.encoder.classes = attrib.classes
                self.encoder.classes_long = attrib.classes_long
        else:
            self.encoder = resnets[backbone](pretrained=pretrained, ctx=ctx, **kwargs)

        if backbone not in ('resnet18', 'resnet34'):
            self.num_ch_enc[1:] *= 4

    def hybrid_forward(self, F, input_image):
        # pylint: disable=unused-argument, missing-function-docstring
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features

    def predict(self, input_image):
        # pylint: disable=unused-argument, missing-function-docstring
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features
