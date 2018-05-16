from __future__ import print_function

import numpy as np
import gluoncv as gcv
from mxnet.gluon.nn import BatchNorm

def check_bn_frozen_callback(net, value):
    if isinstance(net, BatchNorm):
        assert value == net._kwargs['use_global_stats']

def test_block_freeze_bn():
    net = gcv.model_zoo.get_model('resnet18_v1')
    gcv.utils.recursive_visit(net, check_bn_frozen_callback, value=False)
    gcv.utils.freeze_bn(net, True)
    gcv.utils.recursive_visit(net, check_bn_frozen_callback, value=True)
    gcv.utils.freeze_bn(net, False)
    gcv.utils.recursive_visit(net, check_bn_frozen_callback, value=False)

if __name__ == '__main__':
    import nose
    nose.runmodule()
