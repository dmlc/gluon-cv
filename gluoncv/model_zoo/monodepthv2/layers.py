# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

import mxnet as mx
from mxnet.gluon import nn


def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth


def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()

    if invert:
        R = R.transpose(1, 2)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = mx.nd.dot(R, T)
    else:
        M = mx.nd.dot(T, R)

    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = mx.nd.zeros(translation_vector.shape[0], 4, 4).as_in_context(context=translation_vector.context)

    t = translation_vector.contiguous().view(-1, 3, 1)

    T[:, 0, 0] = 1
    T[:, 1, 1] = 1
    T[:, 2, 2] = 1
    T[:, 3, 3] = 1
    T[:, :3, 3, None] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = mx.nd.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)

    ca = mx.nd.cos(angle)
    sa = mx.nd.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(1)
    y = axis[..., 1].unsqueeze(1)
    z = axis[..., 2].unsqueeze(1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = mx.nd.zeros((vec.shape[0], 4, 4)).as_in_context(context=vec.context)

    rot[:, 0, 0] = mx.nd.squeeze(x * xC + ca)
    rot[:, 0, 1] = mx.nd.squeeze(xyC - zs)
    rot[:, 0, 2] = mx.nd.squeeze(zxC + ys)
    rot[:, 1, 0] = mx.nd.squeeze(xyC + zs)
    rot[:, 1, 1] = mx.nd.squeeze(y * yC + ca)
    rot[:, 1, 2] = mx.nd.squeeze(yzC - xs)
    rot[:, 2, 0] = mx.nd.squeeze(zxC - ys)
    rot[:, 2, 1] = mx.nd.squeeze(yzC + xs)
    rot[:, 2, 2] = mx.nd.squeeze(z * zC + ca)
    rot[:, 3, 3] = 1

    return rot


class ConvBlock(nn.HybridBlock):
    """Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        with self.name_scope():
            self.conv = Conv3x3(in_channels, out_channels)
            self.nonlin = nn.ELU()

    def hybrid_forward(self, F, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.HybridBlock):
    """Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()
        self.use_refl = use_refl

        # if use_refl:
        #     self.pad = nn.ReflectionPad2D(1)
        # else:
        #     self.pad = nn.ZeroPad2d(1)
        with self.name_scope():
            self.pad = nn.ReflectionPad2D(1)
            self.conv = nn.Conv2D(in_channels=int(in_channels), channels=int(out_channels), kernel_size=3)

    def hybrid_forward(self, F, x):
        if self.use_refl:
            out = self.pad(x)
        else:
            out = mx.nd.pad(x, mode='constant', constant_value=0, pad_width=(0, 0, 0, 0, 1, 1, 1, 1))

        out = self.conv(out)
        return out


# class BackprojectDepth(nn.Module):
#     """Layer to transform a depth image into a point cloud
#     """
#     def __init__(self, batch_size, height, width):
#         super(BackprojectDepth, self).__init__()
#
#         self.batch_size = batch_size
#         self.height = height
#         self.width = width
#
#         meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
#         self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
#         self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
#                                       requires_grad=False)
#
#         self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
#                                  requires_grad=False)
#
#         self.pix_coords = torch.unsqueeze(torch.stack(
#             [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
#         self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
#         self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
#                                        requires_grad=False)
#
#     def forward(self, depth, inv_K):
#         cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
#         cam_points = depth.view(self.batch_size, 1, -1) * cam_points
#         cam_points = torch.cat([cam_points, self.ones], 1)
#
#         return cam_points


# class Project3D(nn.Module):
#     """Layer which projects 3D points into a camera with intrinsics K and at position T
#     """
#     def __init__(self, batch_size, height, width, eps=1e-7):
#         super(Project3D, self).__init__()
#
#         self.batch_size = batch_size
#         self.height = height
#         self.width = width
#         self.eps = eps
#
#     def forward(self, points, K, T):
#         P = torch.matmul(K, T)[:, :3, :]
#
#         cam_points = torch.matmul(P, points)
#
#         pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
#         pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
#         pix_coords = pix_coords.permute(0, 2, 3, 1)
#         pix_coords[..., 0] /= self.width - 1
#         pix_coords[..., 1] /= self.height - 1
#         pix_coords = (pix_coords - 0.5) * 2
#         return pix_coords


def upsample(x):
    """Upsample input tensor by a factor of 2
    """
    return mx.nd.contrib.BilinearResize2D(x, height=x.shape[2]*2, width=x.shape[3]*2)


# def get_smooth_loss(disp, img):
#     """Computes the smoothness loss for a disparity image
#     The color image is used for edge-aware smoothness
#     """
#     grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
#     grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])
#
#     grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
#     grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)
#
#     grad_disp_x *= torch.exp(-grad_img_x)
#     grad_disp_y *= torch.exp(-grad_img_y)
#
#     return grad_disp_x.mean() + grad_disp_y.mean()
#
#
# class SSIM(nn.Module):
#     """Layer to compute the SSIM loss between a pair of images
#     """
#     def __init__(self):
#         super(SSIM, self).__init__()
#         self.mu_x_pool   = nn.AvgPool2d(3, 1)
#         self.mu_y_pool   = nn.AvgPool2d(3, 1)
#         self.sig_x_pool  = nn.AvgPool2d(3, 1)
#         self.sig_y_pool  = nn.AvgPool2d(3, 1)
#         self.sig_xy_pool = nn.AvgPool2d(3, 1)
#
#         self.refl = nn.ReflectionPad2d(1)
#
#         self.C1 = 0.01 ** 2
#         self.C2 = 0.03 ** 2
#
#     def forward(self, x, y):
#         x = self.refl(x)
#         y = self.refl(y)
#
#         mu_x = self.mu_x_pool(x)
#         mu_y = self.mu_y_pool(y)
#
#         sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
#         sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
#         sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
#
#         SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
#         SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
#
#         return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
#
#
# def compute_depth_errors(gt, pred):
#     """Computation of error metrics between predicted and ground truth depths
#     """
#     thresh = torch.max((gt / pred), (pred / gt))
#     a1 = (thresh < 1.25     ).float().mean()
#     a2 = (thresh < 1.25 ** 2).float().mean()
#     a3 = (thresh < 1.25 ** 3).float().mean()
#
#     rmse = (gt - pred) ** 2
#     rmse = torch.sqrt(rmse.mean())
#
#     rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
#     rmse_log = torch.sqrt(rmse_log.mean())
#
#     abs_rel = torch.mean(torch.abs(gt - pred) / gt)
#
#     sq_rel = torch.mean((gt - pred) ** 2 / gt)
#
#     return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3
