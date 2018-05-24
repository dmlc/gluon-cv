"""RCNN Model."""
from __future__ import absolute_import

from mxnet import gluon
from mxnet.gluon import nn
from ...nn.bbox import BBoxCornerToCenter
from ...nn.coder import NormalizedBoxCenterDecoder, MultiPerClassDecoder


class RCNN(gluon.HybridBlock):
    def __init__(self, features, top_features, classes, roi_mode, roi_size,
                 nms_thresh=0.3, nms_topk=400, **kwargs):
        super(RCNN, self).__init__(**kwargs)
        self.classes = classes
        self.num_class = len(classes)
        assert self.num_class > 0, "Invalid number of class : {}".format(self.num_class)
        assert roi_mode.lower() in ['align', 'pool'], "Invalid roi_mode: {}".format(roi_mode)
        self._roi_mode = roi_mode.lower()
        assert len(roi_size) == 2, "Require (h, w) as roi_size, given {}".format(roi_size)
        self._roi_size = roi_size
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk

        with self.name_scope():
            self.features = features
            self.top_features = top_features
            # if network is None:
            #     # use fine-grained manually designed block as features
            #     self.features = features(pretrained=pretrained)
            #     self.top_features = top_features(pretrained=pretrained)
            # else:
            #     self.features = FeatureExtractor(
            #         network=network, outputs=features, pretrained=pretrained)
            #     self.top_features = FeatureExtractor(network=network, outputs=top_features,
            #         inputs=features, pretrained=pretrained)
            self.global_avg_pool = nn.GlobalAvgPool2D()
            self.class_predictor = nn.Dense(self.num_class + 1)
            self.box_predictor = nn.Dense(self.num_class * 4)
            self.cls_decoder = MultiPerClassDecoder(num_class=self.num_class+1)
            self.box_to_center = BBoxCornerToCenter()
            self.box_decoder = NormalizedBoxCenterDecoder()

    def set_nms(self, nms_thresh=0.3, nms_topk=400):
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk

    def hybrid_forward(self, F, x, width, height):
        raise NotImplementedError
