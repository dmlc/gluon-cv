# pylint: disable=abstract-method
"""Customized Layers.
"""
from __future__ import absolute_import
from mxnet.gluon.nn import BatchNorm

__all__ = ['BatchNormCudnnOff']

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
