# pylint: disable=missing-function-docstring, unused-argument
"""
Non-local module from Non-local Neural Networks
CVPR 2018, https://arxiv.org/abs/1711.07971
Code adapted from https://github.com/open-mmlab/mmaction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm3d


def build_nonlocal_block(cfg):
    """ Build nonlocal block from
    `"Non-local Neural Networks"
    <https://arxiv.org/abs/1711.07971>`_ paper.
    Code adapted from mmaction.
    """
    assert isinstance(cfg, dict)
    cfg_ = cfg.copy()
    return NonLocal(**cfg_)


class NonLocal(nn.Module):
    """
    Non-local module from Non-local Neural Networks
    CVPR 2018, https://arxiv.org/abs/1711.07971
    """
    def __init__(self, in_channels=1024, nonlocal_type="gaussian", dim=3,
                 embed=True, embed_dim=None, sub_sample=False, use_bn=True,
                 norm_layer=BatchNorm3d, norm_kwargs=None, **kwargs):
        super(NonLocal, self).__init__()

        assert nonlocal_type in ['gaussian', 'dot', 'concat']
        self.nonlocal_type = nonlocal_type
        self.embed = embed
        self.embed_dim = embed_dim if embed_dim is not None else in_channels // 2
        self.sub_sample = sub_sample
        self.use_bn = use_bn

        if self.embed:
            if dim == 2:
                self.theta = nn.Conv2d(in_channels=in_channels,
                                       out_channels=self.embed_dim,
                                       kernel_size=(1, 1),
                                       stride=(1, 1),
                                       padding=(0, 0))
                self.phi = nn.Conv2d(in_channels=in_channels,
                                     out_channels=self.embed_dim,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=(0, 0))
                self.g = nn.Conv2d(in_channels=in_channels,
                                   out_channels=self.embed_dim,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=(0, 0))
            elif dim == 3:
                self.theta = nn.Conv3d(in_channels=in_channels,
                                       out_channels=self.embed_dim,
                                       kernel_size=(1, 1, 1),
                                       stride=(1, 1, 1),
                                       padding=(0, 0, 0))
                self.phi = nn.Conv3d(in_channels=in_channels,
                                     out_channels=self.embed_dim,
                                     kernel_size=(1, 1, 1),
                                     stride=(1, 1, 1),
                                     padding=(0, 0, 0))
                self.g = nn.Conv3d(in_channels=in_channels,
                                   out_channels=self.embed_dim,
                                   kernel_size=(1, 1, 1),
                                   stride=(1, 1, 1),
                                   padding=(0, 0, 0))

        if self.nonlocal_type == 'concat':
            if dim == 2:
                self.concat_proj = nn.Sequential(
                    nn.Conv2d(in_channels=self.embed_dim * 2,
                              out_channels=1,
                              kernel_size=(1, 1),
                              stride=(1, 1),
                              padding=(0, 0)),
                    nn.ReLU(inplace=True))
            elif dim == 3:
                self.concat_proj = nn.Sequential(
                    nn.Conv3d(in_channels=self.embed_dim * 2,
                              out_channels=1,
                              kernel_size=(1, 1, 1),
                              stride=(1, 1, 1),
                              padding=(0, 0, 0)),
                    nn.ReLU(inplace=True))

        if sub_sample:
            if dim == 2:
                self.max_pool = nn.MaxPool2d(pool_size=(2, 2))
            elif dim == 3:
                self.max_pool = nn.MaxPool3d(pool_size=(1, 2, 2))
            self.sub_phi = nn.Sequential(self.phi, self.max_pool)
            self.sub_g = nn.Sequential(self.g, self.max_pool)

        if dim == 2:
            self.W = nn.Conv2d(in_channels=self.embed_dim,
                               out_channels=in_channels,
                               kernel_size=(1, 1),
                               stride=(1, 1),
                               padding=(0, 0))
        elif dim == 3:
            self.W = nn.Conv3d(in_channels=self.embed_dim,
                               out_channels=in_channels,
                               kernel_size=(1, 1, 1),
                               stride=(1, 1, 1),
                               padding=(0, 0, 0))

        if use_bn:
            # TODO: need to add zero initialized BN, also the conv output
            self.bn = norm_layer(num_features=in_channels,
                                 momentum=0.9,
                                 **({} if norm_kwargs is None else norm_kwargs))
            self.W_bn = nn.Sequential(self.W, self.bn)

    def forward(self, x):
        if self.embed:
            theta = self.theta(x)
            if self.sub_sample:
                phi = self.sub_phi(x)
                g = self.sub_g(x)
            else:
                phi = self.phi(x)
                g = self.g(x)
        else:
            theta = x
            phi = x
            g = x

        theta_shape_5d = theta.shape
        if self.nonlocal_type == 'gaussian':
            # reshape [BxCxTxHxW] to [BxCxTHW]
            theta = theta.view(theta_shape_5d[0], theta_shape_5d[1], -1)
            phi = phi.view(theta_shape_5d[0], theta_shape_5d[1], -1)
            g = g.view(theta_shape_5d[0], theta_shape_5d[1], -1)

            theta_phi = torch.bmm(theta.transpose(1, 2), phi)
            theta_phi = theta_phi * (self.embed_dim ** -.5)
            attn = F.softmax(theta_phi, dim=-1)
        elif self.non_local_type == 'concat':
            raise NotImplementedError
        elif self.non_local_type == 'dot':
            raise NotImplementedError
        else:
            raise NotImplementedError

        y = torch.bmm(g, attn.transpose(1, 2))
        y = y.view(theta_shape_5d)

        if self.use_bn:
            z = self.W_bn(y) + x
        else:
            z = self.W(y) + x
        return z
