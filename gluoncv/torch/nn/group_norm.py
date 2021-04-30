"""Group Norm implementation"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter
from torch.nn import init

__all__ = ['GCN', 'NaiveGroupNorm']


class Conv2D(nn.Module):
    """Inline Conv2D for GroupNorm module"""
    def __init__(self, in_channels, out_channels, kernel_size, padding='same',
                 stride=1, dilation=1, groups=1):
        super(Conv2D, self).__init__()

        assert isinstance(kernel_size, (tuple, int)), "Allowed kernel type [int or tuple], not {}".format(type(kernel_size))
        assert padding == 'same', "Allowed padding type {}, not {}".format('same', padding)

        self.kernel_size = kernel_size
        if isinstance(kernel_size, tuple):
            self.h_kernel = kernel_size[0]
            self.w_kernel = kernel_size[1]
        else:
            self.h_kernel = kernel_size
            self.w_kernel = kernel_size

        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=self.stride, dilation=self.dilation, groups=self.groups)

    def forward(self, x):

        if self.padding == 'same':

            height, width = x.shape[2:]

            h_pad_need = max(0, (height - 1) * self.stride + self.h_kernel - height)
            w_pad_need = max(0, (width - 1) * self.stride + self.w_kernel - width)

            pad_left = w_pad_need // 2
            pad_right = w_pad_need - pad_left
            pad_top = h_pad_need // 2
            pad_bottom = h_pad_need - pad_top

            padding = (pad_left, pad_right, pad_top, pad_bottom)

            x = F.pad(x, padding, 'constant', 0)

        x = self.conv(x)

        return x


class GCN(nn.Module):
    """
    Large Kernel Matters -- https://arxiv.org/abs/1703.02719
    """
    def __init__(self, in_channels, out_channels, k=3):
        super(GCN, self).__init__()

        self.conv_l1 = Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=(k, 1), padding='same')
        self.conv_l2 = Conv2D(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, k), padding='same')

        self.conv_r1 = Conv2D(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, k), padding='same')
        self.conv_r2 = Conv2D(in_channels=out_channels, out_channels=out_channels, kernel_size=(k, 1), padding='same')

    def forward(self, x):
        x1 = self.conv_l1(x)
        x1 = self.conv_l2(x1)

        x2 = self.conv_r1(x)
        x2 = self.conv_r2(x2)

        out = x1 + x2

        return out


class NaiveGroupNorm(Module):
    r"""NaiveGroupNorm implements Group Normalization with the high-level matrix operations in PyTorch.
    It is a temporary solution to export GN by ONNX before the official GN can be exported by ONNX.
    The usage of NaiveGroupNorm is exactly the same as the official :class:`torch.nn.GroupNorm`.
    Args:
        num_groups (int): number of groups to separate the channels into
        num_channels (int): number of channels expected in input
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        affine: a boolean value that when set to ``True``, this module
            has learnable per-channel affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.

    Shape:
        - Input: :math:`(N, C, *)` where :math:`C=\text{num\_channels}`
        - Output: :math:`(N, C, *)` (same shape as input)

    Examples::

        >>> input = torch.randn(20, 6, 10, 10)
        >>> # Separate 6 channels into 3 groups
        >>> m = NaiveGroupNorm(3, 6)
        >>> # Separate 6 channels into 6 groups (equivalent with InstanceNorm)
        >>> m = NaiveGroupNorm(6, 6)
        >>> # Put all 6 channels into a single group (equivalent with LayerNorm)
        >>> m = NaiveGroupNorm(1, 6)
        >>> # Activating the module
        >>> output = m(input)

    .. _`Group Normalization`: https://arxiv.org/abs/1803.08494
    """
    __constants__ = ['num_groups', 'num_channels', 'eps', 'affine', 'weight',
                     'bias']

    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True):
        super(NaiveGroupNorm, self).__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_channels))
            self.bias = Parameter(torch.Tensor(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input):
        N, C, H, W = input.size()
        assert C % self.num_groups == 0
        input = input.reshape(N, self.num_groups, -1)
        mean = input.mean(dim=-1, keepdim=True)
        var = (input ** 2).mean(dim=-1, keepdim=True) - mean ** 2
        std = torch.sqrt(var + self.eps)

        input = (input - mean) / std
        input = input.reshape(N, C, H, W)
        if self.affine:
            input = input * self.weight.reshape(1, C, 1, 1) + self.bias.reshape(1, C, 1, 1)
        return input

    def extra_repr(self):
        return '{num_groups}, {num_channels}, eps={eps}, ' \
            'affine={affine}'.format(**self.__dict__)
