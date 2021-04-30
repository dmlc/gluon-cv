"""Custom and derived deformable convolution"""
# pylint: disable=missing-function-docstring,bad-continuation
import torch
from torch import nn
from torch.nn.modules.utils import _pair
from torchvision.ops import deform_conv2d

__all__ = ['DeformConvWithChangeableStride']


class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None


class DeformConvWithChangeableStride(nn.Module):
    """A deformable conv layer with dynamic stride.

    See "https://github.com/pytorch/vision/blob/master/torchvision/ops/deform_conv.py" for usage.
    There are currently some limitations, e.g. deformable_groups must be 1.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        dilation=1,
        groups=1,
        deformable_groups=1,
        bias=False,
        norm=None,
        activation=None,
    ):
        super(DeformConvWithChangeableStride, self).__init__()

        assert in_channels % groups == 0, "in_channels {} cannot be divisible by groups {}".format(
            in_channels, groups
        )
        assert (
            out_channels % groups == 0
        ), "out_channels {} cannot be divisible by groups {}".format(out_channels, groups)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.deformable_groups = deformable_groups
        # TODO(zhreshold): torchvision version of deform_conv limitation
        assert self.deformable_groups == 1, "deformable_groups can only be 1 at this time"
        self.norm = norm
        self.activation = activation

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None

        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        nn.init.constant_(self.bias, 0.)

    def forward(self, x, offset, stride):
        if x.numel() == 0:
            # When input is empty, we want to return a empty tensor with "correct" shape,
            # So that the following operations will not panic
            # if they check for the shape of the tensor.
            # This computes the height and width of the output tensor
            output_shape = [
                (i + 2 * p - (di * (k - 1) + 1)) // s + 1
                for i, p, di, k, s in zip(
                    x.shape[-2:], self.padding, self.dilation, self.kernel_size, self.stride
                )
            ]
            output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
            return _NewEmptyTensorOp.apply(x, output_shape)

        stride = _pair(stride)
        x = deform_conv2d(
            x,
            offset,
            self.weight,
            self.bias,
            stride,
            self.padding,
            self.dilation,
        )
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
