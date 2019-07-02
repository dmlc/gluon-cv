# pylint: disable=abstract-method,unused-argument
"""Customized Layers.
"""
from __future__ import absolute_import
from mxnet.gluon.nn import BatchNorm, HybridBlock

__all__ = ['BatchNormCudnnOff', 'ReLU6', 'HardSigmoid', 'HardSwish']

class BatchNormCudnnOff(BatchNorm):
    """Batch normalization layer without CUDNN. It is a temporary solution.

    Parameters
    ----------
    kwargs : arguments goes to mxnet.gluon.nn.BatchNorm
    """
    def __init__(self, **kwargs):
        super(BatchNormCudnnOff, self).__init__(**kwargs)

    def hybrid_forward(self, F, x, gamma, beta, running_mean, running_var):
        return F.BatchNorm(x, gamma, beta, running_mean, running_var,
                           name='fwd', cudnn_off=True, **self._kwargs)

class ReLU6(HybridBlock):
    """RelU6 used in MobileNetV2 and MobileNetV3.

    Parameters
    ----------
    kwargs : arguments goes to mxnet.gluon.nn.ReLU6
    """

    def __init__(self, **kwargs):
        super(ReLU6, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return F.clip(x, 0, 6, name="relu6")

class HardSigmoid(HybridBlock):
    """HardSigmoid used in MobileNetV3.

    Parameters
    ----------
    kwargs : arguments goes to mxnet.gluon.nn.HardSigmoid
    """
    def __init__(self, **kwargs):
        super(HardSigmoid, self).__init__(**kwargs)
        self.act = ReLU6()

    def hybrid_forward(self, F, x):
        return self.act(x + 3.) / 6.

class HardSwish(HybridBlock):
    """HardSwish used in MobileNetV3.

    Parameters
    ----------
    kwargs : arguments goes to mxnet.gluon.nn.HardSwish
    """
    def __init__(self, **kwargs):
        super(HardSwish, self).__init__(**kwargs)
        self.act = HardSigmoid()

    def hybrid_forward(self, F, x):
        return x * self.act(x)
