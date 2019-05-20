"""Transforms for alpha pose estimation."""
from __future__ import absolute_import

import random
import numpy as np
import mxnet as mx
from .simple_pose import _box_to_center_scale
from ..image import random_flip as random_flip_image
from ..pose import flip_joints_3d, get_affine_transform, affine_transform
from ..pose import random_sample_bbox, refine_bound, count_visible, random_crop_bbox
from ..pose import drawGaussian, transformBox, cv_cropBox, cv_rotate, detector_to_alpha_pose
from .. import experimental
from ....utils.filesystem import try_import_cv2


class AlphaPoseDefaultTrainTransform(object):
    def __init__(self, num_joints, joint_pairs, image_size=(256, 256), heatmap_size=(64, 64),
                 sigma=1, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                 random_flip=True, random_sample=False, random_crop=False,
                 scale_factor=(0.2, 0.3), rotation_factor=30, **kwargs):
        from ....model_zoo.simple_pose.pose_target import SimplePoseGaussianTargetGenerator
        self._sigma = sigma
        self._target_generator = SimplePoseGaussianTargetGenerator(
            num_joints, (image_size[1], image_size[0]), (heatmap_size[1], heatmap_size[0]), sigma)
        self._num_joints = num_joints
        self._joint_pairs = joint_pairs
        self._image_size = image_size
        self._heatmap_size = heatmap_size
        self._height = image_size[0]
        self._width = image_size[1]
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
        src = experimental.image.random_color_distort(src, hue_delta=0)

        # scaling
        ul = np.array((int(bbox[0]), int(bbox[1])))
        br = np.array((int(bbox[2]), int(bbox[3])))
        h = br[1] - ul[1]
        w = br[0] - ul[0]
        sf = random.uniform(*self._scale_factor)
        ul[0] = max(0, ul[0] - w * sf / 2)
        ul[1] = max(0, ul[1] - h * sf / 2)
        br[0] = min(src.shape[0] - 1, br[0] + w * sf / 2)
        br[1] = min(src.shape[1] - 1, br[1] + h * sf / 2)
        if self._random_sample:
            ul, br = random_sample_bbox(ul, br, w, h, src.shape[1], src.shape[0])

        # boundary refine
        ul, br = refine_bound(ul, br)

        # counting number of joints
        num_visible_joint, vis_joints = count_visible(ul, br, joints_3d)

        if self._random_crop and num_visible_joint > 10:
            ul, br = random_crop_bbox(ul, br)

        if num_visible_joint < 1:
            # no valid keypoints
            img = mx.nd.image.to_tensor(mx.image.imresize(src, w=self._width, h=self._height))
            img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)
            target = np.zeros((self._num_joints, self._heatmap_size[0], self._heatmap_size[1]), dtype='float32')
            target_weight = np.zeros((self._num_joints, 1, 1), dtype='float32')
            return img, target, target_weight, img_path

        # crop box with padding
        img = cv_cropBox(src.asnumpy(), ul, br, self._height, self._width)

        # generate labels
        target = np.zeros((self._num_joints, self._heatmap_size[0], self._heatmap_size[1]), dtype='float32')
        target_weight = np.zeros((self._num_joints, 1, 1), dtype='float32')

        for i, vis in enumerate(vis_joints):
            if vis:
                hm_part = transformBox(
                    (joints_3d[i, 0, 0], joints_3d[i, 1, 0]),
                    ul, br, self._height, self._width, self._heatmap_size[0], self._heatmap_size[1])
                target[i] = drawGaussian(target[i], hm_part, self._sigma)
                target_weight[i] = 1

        # random flip
        joints = joints_3d
        if self._random_flip and random.random() > 0.5:
            img = img[:, ::-1, :]
            joints = flip_joints_3d(joints_3d, img.shape[1], self._joint_pairs)

        # rotation
        rf = self._rotation_factor
        r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) if random.random() <= 0.6 else 0
        img = cv_rotate(img, r, self._width, self._height)
        target = cv_rotate(target.transpose((1, 2, 0)), r, self._heatmap_size[1], self._heatmap_size[0])
        target = target.transpose((2, 0, 1))
        # to tensor
        img = mx.nd.image.to_tensor(mx.nd.array(img))
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)
        return img, target, target_weight, img_path


class AlphaPoseDefaultValTransform(object):
    def __init__(self, num_joints, joint_pairs, image_size=(256, 256),
                 sigma=1, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), **kwargs):
        self._num_joints = num_joints
        self._joint_pairs = joint_pairs
        self._image_size = image_size
        self._height = image_size[0]
        self._width = image_size[1]
        self._mean = mean
        self._std = std
        self._aspect_ratio = float(self._width) / self._height

    def __call__(self, src, label, img_path):
        cv2 = try_import_cv2()
        bbox = label['bbox']
        assert len(bbox) == 4
        img, scale_box = detector_to_alpha_pose(
            src,
            class_ids=mx.nd.array([0.]),
            scores=mx.nd.array([1.]),
            bounding_boxs=mx.nd.array(bbox),
            output_shape=self._image_size,
            mean=self._mean,
            std=self._std)
        return img, scale_box, img_path

        # joints_3d = label['joints_3d']
        # xmin, ymin, xmax, ymax = bbox
        # center, scale = _box_to_center_scale(
        #     xmin, ymin, xmax - xmin, ymax - ymin, self._aspect_ratio, scale_mult=1.0)
        # score = label.get('score', 1)
        #
        # h, w = self._image_size
        # trans = get_affine_transform(center, scale, 0, [w, h])
        # img = cv2.warpAffine(src.asnumpy(), trans, (int(w), int(h)), flags=cv2.INTER_LINEAR)
        #
        # # to tensor
        # img = mx.nd.image.to_tensor(mx.nd.array(img))
        # img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)

        # return img, scale, center, score, img_path
