"""Transforms for alpha pose estimation."""
from __future__ import absolute_import

import random
import numpy as np
import mxnet as mx
from .simple_pose import _box_to_center_scale
from ..image import random_flip as random_flip_image
from ..pose import flip_joints_3d, get_affine_transform, affine_transform
from .. import experimental
from ....utils.filesystem import try_import_cv2


class AlphaPoseDefaultTrainTransform(object):
    def __init__(self, num_joints, joint_pairs, image_size=(256, 256), heatmap_size=(64, 64),
                 sigma=1, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                 random_flip=True, random_sample=False, random_crop=False,
                 scale_factor=(0.2, 0.3), rotation_factor=30, **kwargs):
        from ....model_zoo.simple_pose.pose_target import SimplePoseGaussianTargetGenerator
        self._target_generator = SimplePoseGaussianTargetGenerator(
            num_joints, (image_size[1], image_size[0]), (heatmap_size[1], heatmap_size[0]), sigma)
        self._num_joints = num_joints
        self._joint_pairs = joint_pairs
        self._image_size = image_size
        self._heatmap_size = heatmap_size
        self._width = image_size[0]
        self._height = image_size[1]
        self._mean = mean
        self._std = std
        self._random_flip = random_flip
        self._random_sample = random_sample
        self._random_crop = random_crop
        self._scale_factor = scale_factor
        self._rotation_factor = rotation_factor
        self._aspect_ratio = float(self._width) / self._height

    def __call__(self, src, label, img_path):
        cv2 = try_import_cv2()
        bbox = label['bbox']
        assert len(bbox) == 4
        joints_3d = label['joints_3d']

        # color jitter
        img = experimental.image.random_color_distort(src)

        # scaling
        ul = np.array((int(bbox[0]), int(bbox[1])))
        br = np.array((int(bbox[2]), int(bbox[3])))
        h = br[1] - ul[1]
        w = br[0] - ul[0]
        sf = random.uniform(*self._scale_factor)
        ul[0] = max(0, ul[0] - w * sf / 2)
        ul[1] = max(0, ul[1] - h * sf / 2)
        br[0] = min(img.shape[0] - 1, br[0] + w * sf / 2)
        br[1] = min(img.shape[1] - 1, br[1] + h * sf / 2)
        if self._random_sample:
            ul, br = self._random_sample_bbox(ul, br, w, h, img.shape[1], img.shape[0])


        # boundary refine
        ul[0] = min(ul[0], br[0] - 5)
        ul[1] = min(ul[1], br[1] - 5)
        br[0] = max(br[0], ul[0] + 5)
        br[1] = max(br[1], ul[1] + 5)

        # counting number of joints
        num_visible_joint = np.sum(np.logical_and.reduce((
            joints_3d[:, 0, 0] > 0,
            joints_3d[:, 0, 0] > ul[0],
            joints_3d[:, 0, 0] < br[0],
            joints_3d[:, 1, 0] > 0,
            joints_3d[:, 1, 0] > ul[1],
            joints_3d[:, 1, 0] < br[1],
            joints_3d[:, 0, 1] > 0,
            joints_3d[:, 1, 1] > 0
            )))

        if self._random_crop and num_visible_joint > 10 and False:
            ul, br = self._random_crop_bbox(ul, br)

        if num_visible_joint < 1:
            # no valid keypoints
            img = mx.nd.zeros((3, self._image_size[1], self._image_size[0]))
            target = mx.nd.zeros((self._num_joints, self._heatmap_size[1], self._heatmap_size[0]))
            target_weight = mx.nd.zeros((self._num_joints, 1, 1))
            return img, target, target_weight, img_path

        center, scale = _box_to_center_scale(
            ul[0], ul[1], br[0] - ul[0], br[1] - ul[1], self._aspect_ratio)

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

    def _random_sample_bbox(self, ul, br, w, h, im_width, im_height):
        """Take random sample"""
        patch_scale = random.uniform(0, 1)
        if patch_scale > 0.85:
            ratio = float(h) / w
            if w < h:
                patch_w = patch_scale * w
                patch_h = patch_width * ratio
            else:
                patch_h = patch_scale * h
                patch_width = patch_h / ratio
            xmin = ul[0] + random.uniform(0, 1) * (w - patch_w)
            ymin = ul[1] + random.uniform(0, 1) * (h - patch_h)
            xmax = xmin + patch_w + 1
            ymax = ymin + patch_h + 1
        else:
            xmin = max(1, min(ul[0] + np.random.normal(-0.0142, 0.1158) * w, im_width - 3))
            ymin = max(1, min(ul[1] + np.random.normal(0.0043, 0.068) * h, im_height - 3))
            xmax = min(max(xmin + 2, br[0] + np.random.normal(0.0154, 0.1337) * w), im_width - 3)
            ymax = min(max(ymin + 2, br[1] + np.random.normal(-0.0013, 0.0711) * h), im_height - 3)

        ul[0] = xmin
        ul[1] = ymin
        br[0] = xmax
        br[1] = ymax
        return ul, br

    def _random_crop_bbox(self, ul, br):
        switch = random.uniform(0, 1)
        if switch > 0.96:
            br[0] = (ul[0] + br[0]) / 2
            br[1] = (ul[1] + br[1]) / 2
        elif switch > 0.92:
            ul[0] = (ul[0] + br[0]) / 2
            br[1] = (ul[1] + br[1]) / 2
        elif switch > 0.88:
            ul[1] = (ul[1] + br[1]) / 2
            br[0] = (ul[0] + br[0]) / 2
        elif switch > 0.84:
            ul[0] = (ul[0] + br[0]) / 2
            ul[1] = (ul[1] + br[1]) / 2
        elif switch > 0.80:
            br[0] = (ul[0] + br[0]) / 2
        elif switch > 0.76:
            ul[0] = (ul[0] + br[0]) / 2
        elif switch > 0.72:
            br[1] = (ul[1] + br[1]) / 2
        elif switch > 0.68:
            ul[1] = (ul[1] + br[1]) / 2

        return ul, br

class AlphaPoseDefaultValTransform(object):
    def __init__(self, num_joints, joint_pairs, image_size=(256, 256), heatmap_size=(64, 64),
                 sigma=1, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), **kwargs):
        self._num_joints = num_joints
        self._joint_pairs = joint_pairs
        self._image_size = image_size
        self._heatmap_size = heatmap_size
        self._width = image_size[0]
        self._height = image_size[1]
        self._mean = mean
        self._std = std
        self._aspect_ratio = float(self._width) / self._height

    def __call__(self, src, label, img_path):
        cv2 = try_import_cv2()
        bbox = label['bbox']
        assert len(bbox) == 4
        joints_3d = label['joints_3d']
        xmin, ymin, xmax, ymax = bbox
        center, scale = _box_to_center_scale(
            xmin, ymin, xmax - xmin, ymax - ymin, self._aspect_ratio)
        score = label.get('score', 1)

        h, w = self._image_size
        trans = get_affine_transform(center, scale, 0, [w, h])
        img = cv2.warpAffine(src.asnumpy(), trans, (int(w), int(h)), flags=cv2.INTER_LINEAR)

        # to tensor
        img = mx.nd.image.to_tensor(mx.nd.array(img))
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)

        return img, scale, center, score, img_path
