# pylint: disable= arguments-differ,missing-docstring
"""Basic neural network layers."""
__all__ = ['GroupNorm']
import numpy as np

import mxnet as mx
from mxnet.gluon.parameter import Parameter
from mxnet.gluon.block import HybridBlock
from mxnet import autograd
from mxnet import use_np # pylint: disable=unused-import
mx.npx.set_np()

@use_np
class GroupNorm(HybridBlock):
    """GroupNorm normalization layer (Wu and He, 2014).

    Parameters
    ----------
    ngroups : int
        Numnber of channel groups in GN.
    in_channels : int, default 0
        Number of channels (feature maps) in input data. If not specified,
        initialization will be deferred to the first time `forward` is called
        and `in_channels` will be inferred from the shape of input data.
    axis : int, default 1
        The axis that should be normalized. This is typically the channels
        (C) axis. For instance, after a `Conv2D` layer with `layout='NCHW'`,
        set `axis=1` in `GroupNorm`. If `layout='NHWC'`, then set `axis=3`.
    epsilon: float, default 1e-5
        Small float added to variance to avoid dividing by zero.
    beta_initializer: str or `Initializer`, default 'zeros'
        Initializer for the beta weight.
    gamma_initializer: str or `Initializer`, default 'ones'
        Initializer for the gamma weight.
    scale: bool, default True
        If True, multiply by `gamma`. If False, `gamma` is not used.

    Inputs:
        - **data**: input tensor with arbitrary shape.

    Outputs:
        - **out**: output tensor with the same shape as `data`.
    """
    def __init__(self, ngroups=32, in_channels=0, axis=1, epsilon=1e-5,
                 beta_initializer='zeros', gamma_initializer='ones', scale=True, **kwargs):
        super(GroupNorm, self).__init__(**kwargs)
        self._kwargs = {'axis': axis, 'eps': epsilon, 'momentum': 0,
                        'fix_gamma': True, 'use_global_stats': False}
        self.ngroups = ngroups
        self.scale = scale
        assert in_channels % ngroups == 0, "Channel number should be divisible by groups."
        if in_channels != 0:
            self.in_channels = in_channels

        self.gamma = Parameter('gamma', grad_req='write',
                               shape=(in_channels,), init=gamma_initializer,
                               allow_deferred_init=True, differentiable=True)
        self.beta = Parameter('beta', grad_req='write',
                              shape=(in_channels,), init=beta_initializer,
                              allow_deferred_init=True, differentiable=True)
        # hacky
        self.inited = False # orphan?

    def cast(self, dtype):
        if np.dtype(dtype).name == 'float16':
            dtype = 'float32'
        super(GroupNorm, self).cast(dtype)

    def forward(self, x, gamma=mx.np.ones(16), beta=mx.np.zeros(16)):
        # normalization
        with autograd.train_mode():
            y = mx.np.expand_dims(x, axis=0)
            y = mx.np.reshape(y, (y.shape[0], y.shape[1], self.ngroups, -1))
            y = mx.np.reshape(y, (1, y.shape[1] * y.shape[2], -1))
            batch = x.shape[0]
            y = mx.npx.batch_norm(y,
                               mx.np.ones(batch*self.ngroups, ctx=x.context),
                               mx.np.zeros(batch*self.ngroups, ctx=x.context),
                               mx.np.zeros(batch*self.ngroups, ctx=x.context),
                               mx.np.ones(batch*self.ngroups, ctx=x.context),
                               name='fwd', **self._kwargs)
        # scale and shift
        y = mx.npx.reshape_like(y, x).reshape(y.shape[0], y.shape[1], -1)
        if self.scale:
            y = y * gamma.reshape(1, -1, 1) + beta.reshape(1, -1, 1)
        else:
            y = y + beta.reshape(1, -1, 1)
        return mx.npx.reshape_like(y, x)

    def __repr__(self):
        s = '{name}({content}'
        in_channels = self.gamma.shape[0]
        s += ', ngroups={0}'.format(self.ngroups)
        s += ', in_channels={0}'.format(in_channels if in_channels else None)
        s += ')'
        return s.format(name=self.__class__.__name__,
                        content=', '.join(['='.join([k, v.__repr__()])
                                           for k, v in self._kwargs.items()]))
