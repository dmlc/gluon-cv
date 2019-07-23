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
from ....utils.filesystem import try_import_cv2


class AlphaPoseDefaultTrainTransform(object):
    def __init__(self, num_joints, joint_pairs, image_size=(256, 256), heatmap_size=(64, 64),
                 sigma=1, random_flip=True,
                 random_sample=False, random_crop=False,
                 scale_factor=(0.2, 0.3), rotation_factor=30, **kwargs):
        self._sigma = sigma
        self._num_joints = num_joints
        self._joint_pairs = joint_pairs
        self._image_size = image_size
        self._heatmap_size = heatmap_size
        self._height = image_size[0]
        self._width = image_size[1]
        self._random_flip = random_flip
        self._random_sample = random_sample
        self._random_crop = random_crop
        self._scale_factor = scale_factor
        self._rotation_factor = rotation_factor
        self._aspect_ratio = float(self._width) / self._height

    def __call__(self, src, label, img_path):
        src = src.asnumpy()
        bbox = label['bbox']
        assert len(bbox) == 4
        joints_3d = label['joints_3d']

        # color jitter src-(0, 255)
        src[:, :, 0] = (src[:, :, 0] * random.uniform(0.7, 1.3)).clip(0, 255)
        src[:, :, 1] = (src[:, :, 1] * random.uniform(0.7, 1.3)).clip(0, 255)
        src[:, :, 2] = (src[:, :, 2] * random.uniform(0.7, 1.3)).clip(0, 255)
        imght, imgwd = src.shape[0], src.shape[1]
        assert src.shape[2] == 3

        # scaling
        ul = np.array((int(bbox[0]), int(bbox[1])))
        br = np.array((int(bbox[2]), int(bbox[3])))
        h = br[1] - ul[1]
        w = br[0] - ul[0]
        sf = random.uniform(*self._scale_factor)
        ul[0] = max(0, ul[0] - w * sf / 2)
        ul[1] = max(0, ul[1] - h * sf / 2)
        br[0] = min(imgwd - 1, br[0] + w * sf / 2)
        br[1] = min(imght - 1, br[1] + h * sf / 2)
        if self._random_sample:
            ul, br = random_sample_bbox(ul, br, w, h, imgwd, imght)

        # boundary refine
        # ul, br = refine_bound(ul, br)

        # counting number of joints
        num_visible_joint, vis_joints = count_visible(ul, br, joints_3d)

        if self._random_crop and num_visible_joint > 10:
            ul, br = random_crop_bbox(ul, br)

        if num_visible_joint < 1:
            # no valid keypoints
            img = mx.nd.zeros((3, self._height, self._width))
            target = np.zeros((self._num_joints, self._heatmap_size[0], self._heatmap_size[1]), dtype='float32')
            target_weight = np.ones((self._num_joints, 1, 1), dtype='float32')
            return img, target, target_weight, img_path

        # crop box with padding
        img = src
        img = cv_cropBox(img, ul, br, self._height, self._width)

        # generate labels
        target = np.zeros((self._num_joints, self._heatmap_size[0], self._heatmap_size[1]), dtype='float32')
        target_weight = np.ones((self._num_joints, 1, 1), dtype='float32')

        for i, vis in enumerate(vis_joints):
            if vis:
                hm_part = transformBox(
                    (joints_3d[i, 0, 0], joints_3d[i, 1, 0]),
                    ul, br, self._height, self._width, self._heatmap_size[0], self._heatmap_size[1])
                target[i] = drawGaussian(target[i], hm_part, self._sigma)
            target_weight[i] = 1

        # random flip
        if self._random_flip and random.random() > 0.5:
            assert img.shape[2] == 3
            img = img[:, ::-1, :]
            # target: (num_joints, h, w)
            target = flip_heatmap(target, self._joint_pairs)

        # rotation
        rf = self._rotation_factor
        r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) if random.random() <= 0.6 else 0
        img = cv_rotate(img, r, self._width, self._height)
        target = cv_rotate(target.transpose((1, 2, 0)), r, self._heatmap_size[1], self._heatmap_size[0])
        target = target.transpose((2, 0, 1))
        # to tensor
        src = mx.nd.image.to_tensor(mx.nd.array(img))
        src[0] = src[0] - 0.406
        src[1] = src[1] - 0.457
        src[2] = src[2] - 0.480
        assert src.shape[0] == 3

        return src, target, target_weight, img_path


class AlphaPoseDefaultValTransform(object):
    def __init__(self, num_joints, joint_pairs, image_size=(256, 256),
                 sigma=1, **kwargs):
        self._num_joints = num_joints
        self._joint_pairs = joint_pairs
        self._image_size = image_size
        self._height = image_size[0]
        self._width = image_size[1]
        self._aspect_ratio = float(self._width) / self._height

    def __call__(self, src, label, img_path):
        src = src.asnumpy()
        bbox = label['bbox']
        assert len(bbox) == 4
        score = label.get('score', 1)
        img, scale_box = detector_to_alpha_pose(
            src,
            class_ids=mx.nd.array([[0.]]),
            scores=mx.nd.array([[1.]]),
            bounding_boxs=mx.nd.array(np.array([bbox])),
            output_shape=self._image_size)
        # print(scale_box)
        # import cv2
        # img = img[0].asnumpy()
        # img = img.transpose((1, 2, 0))
        # img *= np.array([0.229,0.224,0.225])
        # img += np.array([0.406,0.457,0.480])
        # img *= 255
        # img = np.maximum(0, img)
        # img = np.minimum(255, img)
        # img = img.astype('uint8')
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # src = cv2.cvtColor(src.asnumpy(), cv2.COLOR_BGR2RGB)
        # sb = [int(x) for x in scale_box[0]]
        # cv2.rectangle(src, (sb[0], sb[1]), (sb[2], sb[3]), (255, 0, 0))
        # cv2.imshow('src', src)
        # cv2.imshow('debug', img)
        # cv2.waitKey()
        return img[0], scale_box, score, img_path

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
        #
        # return img, scale, center, score, img_path


def flip_heatmap(heatmap, joint_pairs, shift=False):
    """Flip pose heatmap according to joint pairs.

    Parameters
    ----------
    heatmap : numpy.ndarray
        Heatmap of joints.
    joint_pairs : list
        List of joint pairs
    shift : bool
        Whether to shift the output

    Returns
    -------
    numpy.ndarray
        Flipped heatmap

    """
    assert heatmap.ndim == 3, "heatmap should have shape (num_joints, height, width)"
    out = heatmap[:, :, ::-1]

    for pair in joint_pairs:
        tmp = out[pair[0], :, :].copy()
        out[pair[0], :, :] = out[pair[1], :, :]
        out[pair[1], :, :] = tmp

    if shift:
        out[:, :, 1:] = out[:, :, 0:-1]
    return out
