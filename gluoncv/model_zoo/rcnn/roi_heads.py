"""ROI heads."""

from __future__ import absolute_import

from mxnet.gluon import nn


class MLPFeatureExtractor(nn.HybridBlock):
    """
    Head for FPN for classification and box regression.
    """

    def __init__(self, channels=1024):
        super(MLPFeatureExtractor, self).__init__()
        self.roi_head = nn.HybridSequential()
        with self.name_scope():
            self.roi_head.add(nn.Dense(channels, prefix='rcnn_dense0_'),
                              nn.Activation('relu'),
                              nn.Dense(channels, prefix='rcnn_dense1_'),
                              nn.Activation('relu'))

    def hybrid_forward(self, F, x, *args, **kwargs):
        return self.roi_head(x)
