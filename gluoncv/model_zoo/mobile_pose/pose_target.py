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

class MobilePoseGaussianTargetGenerator(object):
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
    def __init__(self, num_joints, image_size, heatmap_size, sigma=2, normalize=True):
        self._num_joints = num_joints
        self._sigma = sigma
        self._normalize = normalize
        self._image_size = np.array(image_size)
        self._heatmap_size = np.array(heatmap_size)
        assert self._image_size.shape == (2,), "Invalid shape of image_size, expected (2,)"
        assert self._heatmap_size.shape == (2,), "Invalid shape of heatmap_size, expected (2,)"

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

        # size = (self._heatmap_size[1], self._heatmap_size[0])
        size = self._heatmap_size
        x_linspace = np.stack([np.linspace(1 / (2*size[0]), 1-1/(2*size[0]), size[0]) \
                               for i in range(size[1])])
        y_linspace = np.stack([np.linspace(1 / (2*size[1]), 1-1/(2*size[1]), size[1]) \
                               for i in range(size[0])]).transpose()
        x_denom = - 1 / (2 * (self._sigma / size[0]) ** 2)
        y_denom = - 1 / (2 * (self._sigma / size[1]) ** 2)
        for i in range(self._num_joints):
            x_coords = joints_3d[i, 0, 0]/self._image_size[0]
            y_coords = joints_3d[i, 1, 0]/self._image_size[1]
            kernel_dists = (x_linspace - x_coords) ** 2 * x_denom + \
                (y_linspace - y_coords) ** 2 * y_denom
            gauss = np.exp(kernel_dists)
            if not self._normalize:
                target[i, :, :] = gauss
            else:
                norm = gauss.sum() + 1e-12
                target[i, :, :] = (gauss / norm)

        return target, np.expand_dims(target_weight, -1)
