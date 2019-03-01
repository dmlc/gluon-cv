# pylint: disable=all
"""Transforms for simple pose estimation."""
from __future__ import absolute_import

import random
import numpy as np
import mxnet as mx
from ..image import random_flip as random_flip_image
from ..pose import flip_joints_3d, get_affine_transform, affine_transform
from ....utils.filesystem import try_import_cv2


class SimplePoseDefaultTrainTransform(object):
    """Default training transform for simple pose.

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
    def __init__(self, num_joints, joint_pairs, image_size=(256, 256), heatmap_size=(64, 64),
                 sigma=2, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                 random_flip=True, scale_factor=0.25, rotation_factor=30,
                 **kwargs):
        from ....model_zoo.simple_pose.pose_target import SimplePoseGaussianTargetGenerator
        self._target_generator = SimplePoseGaussianTargetGenerator(
            num_joints, (image_size[1], image_size[0]), (heatmap_size[1], heatmap_size[0]), sigma)
        self._num_joints = num_joints
        self._image_size = image_size
        self._joint_pairs = joint_pairs
        self._width = image_size[0]
        self._height = image_size[1]
        self._mean = mean
        self._std = std
        self._random_flip = random_flip
        self._scale_factor = scale_factor
        self._rotation_factor = rotation_factor

    def __call__(self, src, label, img_path):
        cv2 = try_import_cv2()
        joints_3d = label['joints_3d']
        center = label['center']
        scale = label['scale']
        # score = label.get('score', 1)

        # rescale
        sf = self._scale_factor
        scale = scale * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

        # rotation
        rf = self._rotation_factor
        r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) if random.random() <= 0.6 else 0

        joints = joints_3d
        if self._random_flip and random.random() > 0.5:
            # src, fliped = random_flip_image(src, px=0.5, py=0)
            # if fliped[0]:
            src = src[:, ::-1, :]
            joints = flip_joints_3d(joints_3d, src.shape[1], self._joint_pairs)
            center[0] = src.shape[1] - center[0] - 1

        h, w = self._image_size
        trans = get_affine_transform(center, scale, r, [w, h])
        img = cv2.warpAffine(src.asnumpy(), trans, (int(w), int(h)), flags=cv2.INTER_LINEAR)

        # deal with joints visibility
        for i in range(self._num_joints):
            if joints[i, 0, 1] > 0.0:
                joints[i, 0:2, 0] = affine_transform(joints[i, 0:2, 0], trans)

        # generate training targets
        target, target_weight = self._target_generator(joints)

        # to tensor
        img = mx.nd.image.to_tensor(mx.nd.array(img))
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)
        return img, target, target_weight, img_path


class SimplePoseDefaultValTransform(object):
    """Default training transform for simple pose.

    Parameters
    ----------
    num_joints : int
        Number of joints defined by dataset
    image_size : tuple of int
        Image size, as (width, height).
    heatmap_size : tuple of int
        Heatmap size, as (width, height).

    """
    def __init__(self, num_joints, joint_pairs, image_size=(256, 256),
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), **kwargs):
        self._num_joints = num_joints
        self._image_size = image_size
        self._joint_pairs = joint_pairs
        self._width = image_size[0]
        self._height = image_size[1]
        self._mean = mean
        self._std = std

    def __call__(self, src, label, img_path):
        cv2 = try_import_cv2()
        joints_3d = label['joints_3d']
        center = label['center']
        scale = label['scale']
        score = label.get('score', 1)

        h, w = self._image_size
        trans = get_affine_transform(center, scale, 0, [w, h])
        img = cv2.warpAffine(src.asnumpy(), trans, (int(w), int(h)), flags=cv2.INTER_LINEAR)

        # to tensor
        img = mx.nd.image.to_tensor(mx.nd.array(img))
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)

        return img, scale, center, score, img_path
