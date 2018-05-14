"""RCNN Base Model"""
from mxnet import gluon
import mxnet.ndarray as F
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

    @staticmethod
    def rcnn_nms(rcnn_cls, bbox_pred, thresh=0.5, pre_nms_topN=-1, topK=100):
        """RCNN NMS"""
        nclass = rcnn_cls.shape[1]
        nboxs = bbox_pred.shape[0]
        ids = F.concatenate([classid * F.ones((nboxs, 1), rcnn_cls.context)
                             for classid in range(nclass)], axis=1)
        rcnn_cls = rcnn_cls.reshape(nboxs * nclass, 1)
        bbox_pred = bbox_pred.reshape(nboxs * nclass, 4)
        data = F.concat(ids.reshape(nboxs * nclass, 1), rcnn_cls, bbox_pred, dim=1)
        nms_pred = F.contrib.box_nms(data, thresh, pre_nms_topN, coord_start=2,
                                     score_index=1, id_index=0, force_suppress=False)
        # topK with effective rois
        effect = int(F.sum(nms_pred[:, 0] >= 0).asscalar())
        topK = topK if effect > topK else effect
        # N,1
        rcnn_ids = nms_pred[:topK, 0]
        rcnn_scores = nms_pred[:topK, 1]
        # N, 4
        rcnn_bbox = nms_pred[:topK, 2:]
        return rcnn_ids, rcnn_scores, rcnn_bbox
