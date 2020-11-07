"""Transforms for RCNN series."""
from __future__ import absolute_import

import copy
from random import randint

import mxnet as mx

from .. import bbox as tbbox
from .. import image as timage
from .. import mask as tmask

__all__ = ['transform_test', 'load_test',
           'FasterRCNNDefaultTrainTransform', 'FasterRCNNDefaultValTransform',
           'MaskRCNNDefaultTrainTransform', 'MaskRCNNDefaultValTransform']


def transform_test(imgs, short=600, max_size=1000, mean=(0.485, 0.456, 0.406),
                   std=(0.229, 0.224, 0.225)):
    """A util function to transform all images to tensors as network input by applying
    normalizations. This function support 1 NDArray or iterable of NDArrays.

    Parameters
    ----------
    imgs : NDArray or iterable of NDArray
        Image(s) to be transformed.
    short : int, optional, default is 600
        Resize image short side to this `short` and keep aspect ratio.
    max_size : int, optional, default is 1000
        Maximum longer side length to fit image.
        This is to limit the input image shape, avoid processing too large image.
    mean : iterable of float
        Mean pixel values.
    std : iterable of float
        Standard deviations of pixel values.

    Returns
    -------
    (mxnet.NDArray, numpy.ndarray) or list of such tuple
        A (1, 3, H, W) mxnet NDArray as input to network, and a numpy ndarray as
        original un-normalized color image for display.
        If multiple image names are supplied, return two lists. You can use
        `zip()`` to collapse it.

    """
    if isinstance(imgs, mx.nd.NDArray):
        imgs = [imgs]
    for im in imgs:
        assert isinstance(im, mx.nd.NDArray), "Expect NDArray, got {}".format(type(im))

    tensors = []
    origs = []
    for img in imgs:
        img = timage.resize_short_within(img, short, max_size)
        orig_img = img.asnumpy().astype('uint8')
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=mean, std=std)
        tensors.append(img.expand_dims(0))
        origs.append(orig_img)
    if len(tensors) == 1:
        return tensors[0], origs[0]
    return tensors, origs


def load_test(filenames, short=600, max_size=1000, mean=(0.485, 0.456, 0.406),
              std=(0.229, 0.224, 0.225)):
    """A util function to load all images, transform them to tensor by applying
    normalizations. This function support 1 filename or list of filenames.

    Parameters
    ----------
    filenames : str or list of str
        Image filename(s) to be loaded.
    short : int, optional, default is 600
        Resize image short side to this `short` and keep aspect ratio.
    max_size : int, optional, default is 1000
        Maximum longer side length to fit image.
        This is to limit the input image shape, avoid processing too large image.
    mean : iterable of float
        Mean pixel values.
    std : iterable of float
        Standard deviations of pixel values.

    Returns
    -------
    (mxnet.NDArray, numpy.ndarray) or list of such tuple
        A (1, 3, H, W) mxnet NDArray as input to network, and a numpy ndarray as
        original un-normalized color image for display.
        If multiple image names are supplied, return two lists. You can use
        `zip()`` to collapse it.

    """
    if isinstance(filenames, str):
        filenames = [filenames]
    imgs = [mx.image.imread(f) for f in filenames]
    return transform_test(imgs, short, max_size, mean, std)


class FasterRCNNDefaultTrainTransform(object):
    """Default Faster-RCNN training transform.

    Parameters
    ----------
    short : int/tuple, default is 600
        Resize image shorter side to ``short``.
        Resize the shorter side of the image randomly within the given range, if it is a tuple.
    max_size : int, default is 1000
        Make sure image longer side is smaller than ``max_size``.
    net : mxnet.gluon.HybridBlock, optional
        The faster-rcnn network.

        .. hint::

            If net is ``None``, the transformation will not generate training targets.
            Otherwise it will generate training targets to accelerate the training phase
            since we push some workload to CPU workers instead of GPUs.

    mean : array-like of size 3
        Mean pixel values to be subtracted from image tensor. Default is [0.485, 0.456, 0.406].
    std : array-like of size 3
        Standard deviation to be divided from image. Default is [0.229, 0.224, 0.225].
    box_norm : array-like of size 4, default is (1., 1., 1., 1.)
        Std value to be divided from encoded values.
    num_sample : int, default is 256
        Number of samples for RPN targets.
    pos_iou_thresh : float, default is 0.7
        Anchors larger than ``pos_iou_thresh`` is regarded as positive samples.
    neg_iou_thresh : float, default is 0.3
        Anchors smaller than ``neg_iou_thresh`` is regarded as negative samples.
        Anchors with IOU in between ``pos_iou_thresh`` and ``neg_iou_thresh`` are
        ignored.
    pos_ratio : float, default is 0.5
        ``pos_ratio`` defines how many positive samples (``pos_ratio * num_sample``) is
        to be sampled.
    flip_p : float, default is 0.5
        Probability to flip horizontally, by default is 0.5 for random horizontal flip.
        You may set it to 0 to disable random flip or 1 to force flip.
    ashape : int, default is 128
        Defines shape of pre generated anchors for target generation
    multi_stage : boolean, default is False
        Whether the network output multi stage features.
    """

    def __init__(self, short=600, max_size=1000, net=None, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225), box_norm=(1., 1., 1., 1.),
                 num_sample=256, pos_iou_thresh=0.7, neg_iou_thresh=0.3,
                 pos_ratio=0.5, flip_p=0.5, ashape=128, multi_stage=False, **kwargs):
        self._short = short
        self._max_size = max_size
        self._mean = mean
        self._std = std
        self._anchors = None
        self._multi_stage = multi_stage
        self._random_resize = isinstance(self._short, (tuple, list))
        self._flip_p = flip_p
        if net is None:
            return

        # use fake data to generate fixed anchors for target generation
        anchors = []  # [P2, P3, P4, P5]
        # in case network has reset_ctx to gpu
        anchor_generator = copy.deepcopy(net.rpn.anchor_generator)
        anchor_generator.collect_params().reset_ctx(None)
        if self._multi_stage:
            for ag in anchor_generator:
                anchor = ag(mx.nd.zeros((1, 3, ashape, ashape))).reshape((1, 1, ashape, ashape, -1))
                ashape = max(ashape // 2, 16)
                anchors.append(anchor)
        else:
            anchors = anchor_generator(
                mx.nd.zeros((1, 3, ashape, ashape))).reshape((1, 1, ashape, ashape, -1))
        self._anchors = anchors
        # record feature extractor for infer_shape
        if not hasattr(net, 'features'):
            raise ValueError("Cannot find features in network, it is a Faster-RCNN network?")
        self._feat_sym = net.features(mx.sym.var(name='data'))
        from ....model_zoo.rcnn.rpn.rpn_target import RPNTargetGenerator
        self._target_generator = RPNTargetGenerator(
            num_sample=num_sample, pos_iou_thresh=pos_iou_thresh,
            neg_iou_thresh=neg_iou_thresh, pos_ratio=pos_ratio,
            stds=box_norm, **kwargs)

    def __call__(self, src, label):
        """Apply transform to training image/label."""
        # resize shorter side but keep in max_size
        h, w, _ = src.shape
        if self._random_resize:
            short = randint(self._short[0], self._short[1])
        else:
            short = self._short
        img = timage.resize_short_within(src, short, self._max_size, interp=1)
        bbox = tbbox.resize(label, (w, h), (img.shape[1], img.shape[0]))

        # random horizontal flip
        h, w, _ = img.shape
        img, flips = timage.random_flip(img, px=self._flip_p)
        bbox = tbbox.flip(bbox, (w, h), flip_x=flips[0])

        # to tensor
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)

        if self._anchors is None:
            return img, bbox.astype(img.dtype)

        # generate RPN target so cpu workers can help reduce the workload
        # feat_h, feat_w = (img.shape[1] // self._stride, img.shape[2] // self._stride)
        gt_bboxes = mx.nd.array(bbox[:, :4])
        if self._multi_stage:
            anchor_targets = []
            oshapes = []
            cls_targets, box_targets, box_masks = [], [], []
            for anchor, feat_sym in zip(self._anchors, self._feat_sym):
                oshape = feat_sym.infer_shape(data=(1, 3, img.shape[1], img.shape[2]))[1][0]
                anchor = anchor[:, :, :oshape[2], :oshape[3], :]
                oshapes.append(anchor.shape)
                anchor_targets.append(anchor.reshape((-1, 4)))
            anchor_targets = mx.nd.concat(*anchor_targets, dim=0)
            cls_target, box_target, box_mask = self._target_generator(
                gt_bboxes, anchor_targets, img.shape[2], img.shape[1])
            start_ind = 0
            for oshape in oshapes:
                size = oshape[2] * oshape[3] * (oshape[4] // 4)
                lvl_cls_target = cls_target[start_ind:start_ind + size] \
                    .reshape(oshape[2], oshape[3], -1)
                lvl_box_target = box_target[start_ind:start_ind + size] \
                    .reshape(oshape[2], oshape[3], -1)
                lvl_box_mask = box_mask[start_ind:start_ind + size] \
                    .reshape(oshape[2], oshape[3], -1)
                start_ind += size
                cls_targets.append(lvl_cls_target)
                box_targets.append(lvl_box_target)
                box_masks.append(lvl_box_mask)
        else:
            oshape = self._feat_sym.infer_shape(data=(1, 3, img.shape[1], img.shape[2]))[1][0]
            anchor = self._anchors[:, :, :oshape[2], :oshape[3], :]
            oshape = anchor.shape
            cls_target, box_target, box_mask = self._target_generator(
                gt_bboxes, anchor.reshape((-1, 4)), img.shape[2], img.shape[1])
            size = oshape[2] * oshape[3] * (oshape[4] // 4)
            cls_targets = [cls_target[0:size].reshape(oshape[2], oshape[3], -1)]
            box_targets = [box_target[0:size].reshape(oshape[2], oshape[3], -1)]
            box_masks = [box_mask[0:size].reshape(oshape[2], oshape[3], -1)]
        return img, bbox.astype(img.dtype), cls_targets, box_targets, box_masks


class FasterRCNNDefaultValTransform(object):
    """Default Faster-RCNN validation transform.

    Parameters
    ----------
    short : int, default is 600
        Resize image shorter side to ``short``.
    max_size : int, default is 1000
        Make sure image longer side is smaller than ``max_size``.
    mean : array-like of size 3
        Mean pixel values to be subtracted from image tensor. Default is [0.485, 0.456, 0.406].
    std : array-like of size 3
        Standard deviation to be divided from image. Default is [0.229, 0.224, 0.225].

    """

    def __init__(self, short=600, max_size=1000,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self._mean = mean
        self._std = std
        self._short = short
        self._max_size = max_size

    def __call__(self, src, label):
        """Apply transform to validation image/label."""
        # resize shorter side but keep in max_size
        h, w, _ = src.shape
        img = timage.resize_short_within(src, self._short, self._max_size, interp=1)
        # no scaling ground-truth, return image scaling ratio instead
        bbox = tbbox.resize(label, (w, h), (img.shape[1], img.shape[0]))
        im_scale = h / float(img.shape[0])

        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)
        return img, bbox.astype('float32'), mx.nd.array([im_scale])


class MaskRCNNDefaultTrainTransform(object):
    """Default Mask RCNN training transform.

    Parameters
    ----------
    short : int/tuple, default is 600
        Resize image shorter side to ``short``.
        Resize the shorter side of the image randomly within the given range, if it is a tuple.
    max_size : int, default is 1000
        Make sure image longer side is smaller than ``max_size``.
    net : mxnet.gluon.HybridBlock, optional
        The Mask R-CNN network.

        .. hint::

            If net is ``None``, the transformation will not generate training targets.
            Otherwise it will generate training targets to accelerate the training phase
            since we push some workload to CPU workers instead of GPUs.

    mean : array-like of size 3
        Mean pixel values to be subtracted from image tensor. Default is [0.485, 0.456, 0.406].
    std : array-like of size 3
        Standard deviation to be divided from image. Default is [0.229, 0.224, 0.225].
    box_norm : array-like of size 4, default is (1., 1., 1., 1.)
        Std value to be divided from encoded values.
    num_sample : int, default is 256
        Number of samples for RPN targets.
    pos_iou_thresh : float, default is 0.7
        Anchors larger than ``pos_iou_thresh`` is regarded as positive samples.
    neg_iou_thresh : float, default is 0.3
        Anchors smaller than ``neg_iou_thresh`` is regarded as negative samples.
        Anchors with IOU in between ``pos_iou_thresh`` and ``neg_iou_thresh`` are
        ignored.
    pos_ratio : float, default is 0.5
        ``pos_ratio`` defines how many positive samples (``pos_ratio * num_sample``) is
        to be sampled.
    ashape : int, default is 128
        Defines shape of pre generated anchors for target generation
    multi_stage : boolean, default is False
        Whether the network output multi stage features.
    """

    def __init__(self, short=600, max_size=1000, net=None, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225), box_norm=(1., 1., 1., 1.),
                 num_sample=256, pos_iou_thresh=0.7, neg_iou_thresh=0.3,
                 pos_ratio=0.5, ashape=128, multi_stage=False, **kwargs):
        self._short = short
        self._max_size = max_size
        self._mean = mean
        self._std = std
        self._anchors = None
        self._multi_stage = multi_stage
        self._random_resize = isinstance(self._short, (tuple, list))
        if net is None:
            return

        # use fake data to generate fixed anchors for target generation
        anchors = []  # [P2, P3, P4, P5]
        # in case network has reset_ctx to gpu
        anchor_generator = copy.deepcopy(net.rpn.anchor_generator)
        anchor_generator.collect_params().reset_ctx(None)
        if self._multi_stage:
            for ag in anchor_generator:
                anchor = ag(mx.nd.zeros((1, 3, ashape, ashape))).reshape((1, 1, ashape, ashape, -1))
                ashape = max(ashape // 2, 16)
                anchors.append(anchor)
        else:
            anchors = anchor_generator(
                mx.nd.zeros((1, 3, ashape, ashape))).reshape((1, 1, ashape, ashape, -1))
        self._anchors = anchors
        # record feature extractor for infer_shape
        if not hasattr(net, 'features'):
            raise ValueError("Cannot find features in network, it is a Mask RCNN network?")
        self._feat_sym = net.features(mx.sym.var(name='data'))
        from ....model_zoo.rcnn.rpn.rpn_target import RPNTargetGenerator
        self._target_generator = RPNTargetGenerator(
            num_sample=num_sample, pos_iou_thresh=pos_iou_thresh,
            neg_iou_thresh=neg_iou_thresh, pos_ratio=pos_ratio,
            stds=box_norm, **kwargs)

    def __call__(self, src, label, segm):
        """Apply transform to training image/label."""
        # resize shorter side but keep in max_size
        h, w, _ = src.shape
        if self._random_resize:
            short = randint(self._short[0], self._short[1])
        else:
            short = self._short
        img = timage.resize_short_within(src, short, self._max_size, interp=1)
        bbox = tbbox.resize(label, (w, h), (img.shape[1], img.shape[0]))
        segm = [tmask.resize(polys, (w, h), (img.shape[1], img.shape[0])) for polys in segm]

        # random horizontal flip
        h, w, _ = img.shape
        img, flips = timage.random_flip(img, px=0.5)
        bbox = tbbox.flip(bbox, (w, h), flip_x=flips[0])
        segm = [tmask.flip(polys, (w, h), flip_x=flips[0]) for polys in segm]

        # gt_masks (n, im_height, im_width) of uint8 -> float32 (cannot take uint8)
        masks = [mx.nd.array(tmask.to_mask(polys, (w, h))) for polys in segm]
        # n * (im_height, im_width) -> (n, im_height, im_width)
        masks = mx.nd.stack(*masks, axis=0)

        # to tensor
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)

        if self._anchors is None:
            return img, bbox.astype(img.dtype), masks

        # generate RPN target so cpu workers can help reduce the workload
        # feat_h, feat_w = (img.shape[1] // self._stride, img.shape[2] // self._stride)
        gt_bboxes = mx.nd.array(bbox[:, :4])
        if self._multi_stage:
            anchor_targets = []
            oshapes = []
            cls_targets, box_targets, box_masks = [], [], []
            for anchor, feat_sym in zip(self._anchors, self._feat_sym):
                oshape = feat_sym.infer_shape(data=(1, 3, img.shape[1], img.shape[2]))[1][0]
                anchor = anchor[:, :, :oshape[2], :oshape[3], :]
                oshapes.append(anchor.shape)
                anchor_targets.append(anchor.reshape((-1, 4)))
            anchor_targets = mx.nd.concat(*anchor_targets, dim=0)
            cls_target, box_target, box_mask = self._target_generator(
                gt_bboxes, anchor_targets, img.shape[2], img.shape[1])
            start_ind = 0
            for oshape in oshapes:
                size = oshape[2] * oshape[3] * (oshape[4] // 4)
                lvl_cls_target = cls_target[start_ind:start_ind + size] \
                    .reshape(oshape[2], oshape[3], -1)
                lvl_box_target = box_target[start_ind:start_ind + size] \
                    .reshape(oshape[2], oshape[3], -1)
                lvl_box_mask = box_mask[start_ind:start_ind + size] \
                    .reshape(oshape[2], oshape[3], -1)
                start_ind += size
                cls_targets.append(lvl_cls_target)
                box_targets.append(lvl_box_target)
                box_masks.append(lvl_box_mask)
        else:
            oshape = self._feat_sym.infer_shape(data=(1, 3, img.shape[1], img.shape[2]))[1][0]
            anchor = self._anchors[:, :, :oshape[2], :oshape[3], :]
            oshape = anchor.shape
            cls_target, box_target, box_mask = self._target_generator(
                gt_bboxes, anchor.reshape((-1, 4)), img.shape[2], img.shape[1])
            size = oshape[2] * oshape[3] * (oshape[4] // 4)
            cls_targets = [cls_target[0:size].reshape(oshape[2], oshape[3], -1)]
            box_targets = [box_target[0:size].reshape(oshape[2], oshape[3], -1)]
            box_masks = [box_mask[0:size].reshape(oshape[2], oshape[3], -1)]
        return img, bbox.astype(img.dtype), cls_targets, box_targets, box_masks, masks


class MaskRCNNDefaultValTransform(object):
    """Default Mask RCNN validation transform.

    Parameters
    ----------
    short : int, default is 600
        Resize image shorter side to ``short``.
    max_size : int, default is 1000
        Make sure image longer side is smaller than ``max_size``.
    mean : array-like of size 3
        Mean pixel values to be subtracted from image tensor. Default is [0.485, 0.456, 0.406].
    std : array-like of size 3
        Standard deviation to be divided from image. Default is [0.229, 0.224, 0.225].

    """

    def __init__(self, short=600, max_size=1000,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self._mean = mean
        self._std = std
        self._short = short
        self._max_size = max_size

    def __call__(self, src, label, mask):
        """Apply transform to validation image/label."""
        # resize shorter side but keep in max_size
        h, _, _ = src.shape
        img = timage.resize_short_within(src, self._short, self._max_size, interp=1)
        # no scaling ground-truth, return image scaling ratio instead
        im_scale = float(img.shape[0]) / h

        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)
        return img, mx.nd.array([img.shape[-2], img.shape[-1], im_scale])
