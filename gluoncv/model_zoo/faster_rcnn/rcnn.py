"""RCNN Base Model"""
from mxnet import gluon
import mxnet.gluon.nn as nn

from ..resnetv1b import resnet50_v1b, resnet101_v1b, resnet152_v1b
# pylint: disable=unused-argument

class RCNN_ResNet(gluon.Block):
    """RCNN Base model"""
    def __init__(self, classes, backbone, dilated=False, **kwargs):
        super(RCNN_ResNet, self).__init__(**kwargs)
        self.classes = classes
        self.stride = 8 if dilated else 16
        with self.name_scope():
            # base network
            if backbone == 'resnet50':
                pretrained = resnet50_v1b(pretrained=True, dilated=dilated, **kwargs)
            elif backbone == 'resnet101':
                pretrained = resnet101_v1b(pretrained=True, dilated=dilated, **kwargs)
            elif backbone == 'resnet152':
                pretrained = resnet152_v1b(pretrained=True, dilated=dilated, **kwargs)
            else:
                raise RuntimeError('unknown backbone: {}'.format(backbone))
            self.conv1 = pretrained.conv1
            self.bn1 = pretrained.bn1
            self.relu = pretrained.relu
            self.maxpool = pretrained.maxpool
            self.layer1 = pretrained.layer1
            self.layer2 = pretrained.layer2
            self.layer3 = pretrained.layer3
            self.layer4 = pretrained.layer4

            # TODO FIXME, disable after testing
            # hacky for load caffe pretrained weight
            self.layer2[0].conv1._kwargs['stride'] = (2, 2)
            self.layer2[0].conv2._kwargs['stride'] = (1, 1)
            self.layer3[0].conv1._kwargs['stride'] = (2, 2)
            self.layer3[0].conv2._kwargs['stride'] = (1, 1)
            self.layer4[0].conv1._kwargs['stride'] = (2, 2)
            self.layer4[0].conv2._kwargs['stride'] = (1, 1)

            # RCNN cls and bbox reg
            self.conv_cls = nn.Dense(in_units=2048, units=classes)
            self.conv_reg = nn.Dense(in_units=2048, units=4*classes)
            self.globalavgpool = nn.GlobalAvgPool2D()
        self.conv_cls.initialize()
        self.conv_reg.initialize()
        # TODO lock BN

    def forward(self, *inputs):
        raise NotImplementedError

    def base_forward(self, x):
        """forwarding base network"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        c3 = self.layer3(x)
        return c3

    def top_forward(self, x):
        """forwarding roi feature"""
        c4 = self.layer4(x)
        c4 = self.globalavgpool(c4)
        f_cls = self.conv_cls(c4)
        f_reg = self.conv_reg(c4)
        return f_cls, f_reg
