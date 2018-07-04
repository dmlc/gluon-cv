"""Target generators for YOLOs."""
from __future__ import absolute_import
from __future__ import division

from mxnet import gluon


class YOLOTargetGeneratorV3(gluon.Block):
    def __init__(self, **kwargs):
        super(YOLOTargetGeneratorV3, self).__init__(**kwargs)
        pass

    def forward(self, x):
        pass
