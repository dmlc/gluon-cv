from __future__ import absolute_import, division, print_function

import numpy as np

from mxnet.gluon import nn
from mxnet.context import cpu
from gluoncv.model_zoo.resnetv1b import \
    resnet18_v1b, resnet34_v1b, resnet50_v1s, resnet101_v1s, resnet152_v1s


class ResnetEncoder(nn.HybridBlock):
    def __init__(self, num_layers, pretrained, num_input_images=1, ctx=cpu(), **kwargs):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: resnet18_v1b,
                   34: resnet34_v1b,
                   50: resnet50_v1s,
                   101: resnet101_v1s,
                   152: resnet152_v1s}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            pass
            # self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained=pretrained, ctx=ctx, **kwargs)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def hybrid_forward(self, F, input_image):
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
