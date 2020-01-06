from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
import mxnet as mx
from gluoncv.model_zoo.siamrpn.siam_alexnet import alexnetlegacy
from gluoncv.model_zoo.siamrpn.siam_rpn import DepthwiseRPN

class SiamrpnNet(nn.HybridBlock):
    def __init__(self):
        super(SiamrpnNet, self).__init__()
        self.backbone = alexnetlegacy()
        self.rpn_head = DepthwiseRPN()

    def template(self, z):
        zf = self.backbone(z)
        self.zf = zf
    
    def track(self, x):
        xf = self.backbone(x)
        cls, loc = self.rpn_head(self.zf, xf)
        return {
                'cls': cls,
                'loc': loc,
               }