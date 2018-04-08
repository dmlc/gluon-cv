"""Transforms described in https://arxiv.org/abs/1512.02325."""
from __future__ import absolute_import
import numpy as np
import mxnet as mx
from .. import bbox
from .. import image
from .. import experimental

__all__ = ['SSDDefaultTrainTransform', 'SSDDefaultValTransform']


class SSDDefaultTrainTransform(object):
    """Default SSD training transform which includes tons of image augmentations.

    Parameters
    ----------
    width : int
        Image width.
    height : int
        Image height.
    mean : array-like of size 3
        Mean pixel values to be subtracted from image tensor. Default is [0.485, 0.456, 0.406].
    std : array-like of size 3
        Standard deviation to be divided from image. Default is [0.229, 0.224, 0.225].

    """
    def __init__(self, width, height, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self._width = width
        self._height = height
        self._mean = mean
        self._std = std

    def __call__(self, src, label):
        # random expansion with prob 0.5
        if np.random.uniform(0, 1) > 0.5:
            img, expand = image.random_expand(src, fill=[m * 255 for m in self._mean])
            bbox = bbox.translate(label, x_offset=expand[0], y_offset=expand[1])
        else:
            img, bbox = src, label

        # random cropping
        h, w, _ = img.shape
        bbox, crop = experimental.bbox.random_crop_with_constraints(bbox, (w, h))
        x0, y0, w, h = crop
        img = mx.image.fixed_crop(img, x0, y0, w, h)

        # resize with random interpolation
        h, w, _ = img.shape
        interp = np.random.randint(0, 5)
        img = image.imresize(img, self._width, self._height, interp=interp)
        bbox = bbox.resize(bbox, (w, h), (self._width, self._height))


        # random horizontal flip
        h, w, _ = img.shape
        img, flips = image.random_flip(img, px=0.5)
        bbox = bbox.flip(bbox, (w, h), flip_x=flips[0])

        # random color jittering
        img = experimental.image.random_color_distort(img)

        # to tensor
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)
        return img, bbox.astype('float32')


class SSDDefaultValTransform(object):
    """Default SSD validation transform.

    Parameters
    ----------
    width : int
        Image width.
    height : int
        Image height.
    mean : array-like of size 3
        Mean pixel values to be subtracted from image tensor. Default is [0.485, 0.456, 0.406].
    std : array-like of size 3
        Standard deviation to be divided from image. Default is [0.229, 0.224, 0.225].

    """
    def __init__(self, width, height, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self._width = width
        self._height = height
        self._mean = mean
        self._std = std

    def __call__(self, src, label):
        # resize
        h, w, _ = src.shape
        img = image.imresize(src, self._width, self._height)
        bbox = bbox.resize(label, in_size=(w, h), out_size=(self._width, self._height))

        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)
        return img, bbox.astype('float32')
