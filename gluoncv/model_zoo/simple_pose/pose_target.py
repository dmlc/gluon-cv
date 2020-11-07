"""
Target generator for Simple Baselines for Human Pose Estimation and Tracking
(https://arxiv.org/abs/1804.06208)

---------------------------------------------
Copyright (c) Microsoft
Licensed under the MIT License.
Written by Bin Xiao (Bin.Xiao@microsoft.com)
---------------------------------------------
"""
import numpy as np


class SimplePoseGaussianTargetGenerator(object):
    """Gaussian heatmap target generator for simple pose.
    Adapted from https://github.com/Microsoft/human-pose-estimation.pytorch

    Parameters
    ----------
    num_joints : int
        Number of joints defined by dataset
    image_size : tuple of int
        Image size, as (width, height).
    heatmap_size : tuple of int
        Heatmap size, as (width, height).
    sigma : float
        Gaussian sigma for the heatmap generation.

    """
    def __init__(self, num_joints, image_size, heatmap_size, sigma=2):
        self._num_joints = num_joints
        self._sigma = sigma
        self._image_size = np.array(image_size)
        self._heatmap_size = np.array(heatmap_size)
        assert self._image_size.shape == (2,), "Invalid shape of image_size, expected (2,)"
        assert self._heatmap_size.shape == (2,), "Invalid shape of heatmap_size, expected (2,)"
        self._feat_stride = self._image_size / self._heatmap_size

    def __call__(self, joints_3d):
        """Generate heatmap target and target_weight

        Parameters
        ----------
        joints_3d : numpy.ndarray
            3D joints, with shape (num_joints, 3, 2)

        Returns
        -------
        (numpy.ndarray, numpy.ndarray)
            target : regression target, with shape (num_joints, heatmap_h, heatmap_w)
            target_weight : target weight to mask out non-interest target, shape (num_joints, 1, 1)

        """
        target_weight = np.ones((self._num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_3d[:, 0, 1]
        target = np.zeros((self._num_joints, self._heatmap_size[1], self._heatmap_size[0]),
                          dtype=np.float32)
        tmp_size = self._sigma * 3

        for i in range(self._num_joints):
            mu_x = int(joints_3d[i, 0, 0] / self._feat_stride[0] + 0.5)
            mu_y = int(joints_3d[i, 1, 0] / self._feat_stride[1] + 0.5)
            # check if any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if (ul[0] >= self._heatmap_size[0] or ul[1] >= self._heatmap_size[1] or
                    br[0] < 0 or br[1] < 0):
                # return image as is
                target_weight[i] = 0
                continue

            # generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # the gaussian is not normalized, we want the center value to be equal to 1
            g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * (self._sigma ** 2)))

            # usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], self._heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self._heatmap_size[1]) - ul[1]
            # image range
            img_x = max(0, ul[0]), min(br[0], self._heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], self._heatmap_size[1])

            v = target_weight[i]
            if v > 0.5:
                target[i, img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, np.expand_dims(target_weight, -1)
