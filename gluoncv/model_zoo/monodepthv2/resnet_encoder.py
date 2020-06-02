from __future__ import absolute_import, division, print_function

import numpy as np

from mxnet.gluon import nn
from gluoncv.model_zoo.resnetv1b import \
    resnet18_v1b, resnet34_v1b, resnet50_v1s, resnet101_v1s, resnet152_v1s


class ResnetEncoder(nn.HybridBlock):
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: resnet18_v1b,
                   34: resnet34_v1b,
                   50: resnet50_v1s,
                   101: resnet101_v1s,
                   152: resnet152_v1s}