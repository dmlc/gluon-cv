"""Transforms for RCNN series."""
from __future__ import absolute_import
import numpy as np
import mxnet as mx
from .. import bbox as tbbox
from .. import image as timage
from .. import experimental

__all__ = ['load_test', ]


class FasterRCNNDefaultTrainTransform(object):
    """Default Faster-RCNN training transform.

    Parameters
    ----------
    anchors : mxnet.nd.NDArray, optional
        Anchors generated from SSD networks, the shape must be ``(1, N, 4)``.
        Since anchors are shared in the entire batch so it is ``1`` for the first dimension.
        ``N`` is the number of anchors for each image.

        .. hint::

            If anchors is ``None``, the transformation will not generate training targets.
            Otherwise it will generate training targets to accelerate the training phase
            since we push some workload to CPU workers instead of GPUs.

    mean : array-like of size 3
        Mean pixel values to be subtracted from image tensor. Default is [0.485, 0.456, 0.406].
    std : array-like of size 3
        Standard deviation to be divided from image. Default is [0.229, 0.224, 0.225].
    box_norm : array-like of size 4, default is (0.1, 0.1, 0.2, 0.2)
        Std value to be divided from encoded values.

    """
    def __init__(self, anchors=None, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225), box_norm=(1., 1., 1., 1.),
                 num_sample=256, pos_iou_thresh=0.7, neg_iou_thresh=0.3,
                 pos_ratio=0.5, stride=16, **kwargs):
        self._anchors = anchors
        self._mean = mean
        self._std = std
        self._stride = int(stride)
        if anchors is None:
            return

        from ....model_zoo.rpn.rpn_target import RPNTargetGenerator
        self._target_generator = RPNTargetGenerator(
            num_sample=num_sample, pos_iou_thresh=pos_iou_thresh,
            neg_iou_thresh=neg_iou_thresh, pos_ratio=pos_ratio,
            stds=box_norm, **kwargs)

    def __call__(self, src, label):
        # random horizontal flip
        h, w, _ = src.shape
        img, flips = timage.random_flip(src, px=0.5)
        bbox = tbbox.flip(label, (w, h), flip_x=flips[0])

        # to tensor
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)

        if self._anchors is None:
            return img, bbox.astype('float32')

        # generate RPN target so cpu workers can help reduce the workload
        feat_h, feat_w = (img.shape[1] // self._stride, img.shape[2] // self._stride)
        anchor = self._anchors[:, :, :feat_h, :feat_w, :].reshape((-1, 4))
        gt_bboxes = mx.nd.array(bbox[np.newaxis, :, :4])
        cls_target, cls_mask, box_target, box_mask = self._target_generator(
            gt_bboxes, anchor, img.shape[2], img.shape[1])
        return img, cls_target[0], cls_mask[0], box_target[0], box_mask[0]


class FasterRCNNDefaultValTransform(object):
    """Default SSD validation transform.

    Parameters
    ----------
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
        img = timage.imresize(src, self._width, self._height)
        bbox = tbbox.resize(label, in_size=(w, h), out_size=(self._width, self._height))

        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)
        return img, bbox.astype('float32')
