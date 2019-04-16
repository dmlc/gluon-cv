"""Transforms for alpha pose estimation."""
from __future__ import absolute_import

import numpy as np
import mxnet as mx
from ..image import random_flip as random_flip_image
from ..pose import
from ....utils.filesystem import try_import_cv2


class AlphaPoseDefaultTrainTransform(object):
    def __init__(self, num_joints, joint_pairs, image_size=(256, 256), heatmap_size=(64, 64),
                 sigma=1, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                 random_flip=True, scale_factor=(0.2, 0.3), rotation_factor=30, **kwargs):
        self._num_joints = num_joints
        self._joint_pairs = joint_pairs
        self._image_size = image_size
        self._width = image_size[0]
        self._height = image_size[1]
        self._mean = mean
        self._std = std
        self._random_flip = random_flip
        self._scale_factor = scale_factor
        self._rotation_factor = rotation_factor
