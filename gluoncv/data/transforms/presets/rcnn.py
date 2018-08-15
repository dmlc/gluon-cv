"""Transforms for RCNN series."""
from __future__ import absolute_import
import copy
import mxnet as mx
from .. import bbox as tbbox
from .. import image as timage
from .. import mask as tmask

__all__ = ['load_test',
           'FasterRCNNDefaultTrainTransform', 'FasterRCNNDefaultValTransform',
           'MaskRCNNDefaultTrainTransform', 'MaskRCNNDefaultValTransform']

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
    tensors = []
    origs = []
    for f in filenames:
        img = mx.image.imread(f)
        img = timage.resize_short_within(img, short, max_size)
        orig_img = img.asnumpy().astype('uint8')
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=mean, std=std)
        tensors.append(img.expand_dims(0))
        origs.append(orig_img)
    if len(tensors) == 1:
        return tensors[0], origs[0]
    return tensors, origs


class FasterRCNNDefaultTrainTransform(object):
    """Default Faster-RCNN training transform.

    Parameters
    ----------
    short : int, default is 600
        Resize image shorter side to ``short``.
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

    """
    def __init__(self, short=600, max_size=1000, net=None, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225), box_norm=(1., 1., 1., 1.),
                 num_sample=256, pos_iou_thresh=0.7, neg_iou_thresh=0.3,
                 pos_ratio=0.5, **kwargs):
        self._short = short
        self._max_size = max_size
        self._mean = mean
        self._std = std
        self._anchors = None
        if net is None:
            return

        # use fake data to generate fixed anchors for target generation
        ashape = 128
        # in case network has reset_ctx to gpu
        anchor_generator = copy.deepcopy(net.rpn.anchor_generator)
        anchor_generator.collect_params().reset_ctx(None)
        anchors = anchor_generator(
            mx.nd.zeros((1, 3, ashape, ashape))).reshape((1, 1, ashape, ashape, -1))
        self._anchors = anchors
        # record feature extractor for infer_shape
        if not hasattr(net, 'features'):
            raise ValueError("Cannot find features in network, it is a Faster-RCNN network?")
        self._feat_sym = net.features(mx.sym.var(name='data'))
        from ....model_zoo.rpn.rpn_target import RPNTargetGenerator
        self._target_generator = RPNTargetGenerator(
            num_sample=num_sample, pos_iou_thresh=pos_iou_thresh,
            neg_iou_thresh=neg_iou_thresh, pos_ratio=pos_ratio,
            stds=box_norm, **kwargs)

    def __call__(self, src, label):
        """Apply transform to training image/label."""
        # resize shorter side but keep in max_size
        h, w, _ = src.shape
        img = timage.resize_short_within(src, self._short, self._max_size, interp=1)
        bbox = tbbox.resize(label, (w, h), (img.shape[1], img.shape[0]))

        # random horizontal flip
        h, w, _ = img.shape
        img, flips = timage.random_flip(img, px=0.5)
        bbox = tbbox.flip(bbox, (w, h), flip_x=flips[0])

        # to tensor
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)

        if self._anchors is None:
            return img, bbox.astype(img.dtype)

        # generate RPN target so cpu workers can help reduce the workload
        # feat_h, feat_w = (img.shape[1] // self._stride, img.shape[2] // self._stride)
        oshape = self._feat_sym.infer_shape(data=(1, 3, img.shape[1], img.shape[2]))[1][0]
        anchor = self._anchors[:, :, :oshape[2], :oshape[3], :].reshape((-1, 4))
        gt_bboxes = mx.nd.array(bbox[:, :4])
        cls_target, box_target, box_mask = self._target_generator(
            gt_bboxes, anchor, img.shape[2], img.shape[1])
        return img, bbox.astype(img.dtype), cls_target, box_target, box_mask


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
    short : int, default is 600
        Resize image shorter side to ``short``.
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

    """
    def __init__(self, short=600, max_size=1000, net=None, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225), box_norm=(1., 1., 1., 1.),
                 num_sample=256, pos_iou_thresh=0.7, neg_iou_thresh=0.3,
                 pos_ratio=0.5, **kwargs):
        self._short = short
        self._max_size = max_size
        self._mean = mean
        self._std = std
        self._anchors = None
        if net is None:
            return

        # use fake data to generate fixed anchors for target generation
        ashape = 128
        # in case network has reset_ctx to gpu
        anchor_generator = copy.deepcopy(net.rpn.anchor_generator)
        anchor_generator.collect_params().reset_ctx(None)
        anchors = anchor_generator(
            mx.nd.zeros((1, 3, ashape, ashape))).reshape((1, 1, ashape, ashape, -1))
        self._anchors = anchors
        # record feature extractor for infer_shape
        if not hasattr(net, 'features'):
            raise ValueError("Cannot find features in network, it is a Mask RCNN network?")
        self._feat_sym = net.features(mx.sym.var(name='data'))
        from ....model_zoo.rpn.rpn_target import RPNTargetGenerator
        self._target_generator = RPNTargetGenerator(
            num_sample=num_sample, pos_iou_thresh=pos_iou_thresh,
            neg_iou_thresh=neg_iou_thresh, pos_ratio=pos_ratio,
            stds=box_norm, **kwargs)

    def __call__(self, src, label, segm):
        """Apply transform to training image/label."""
        # resize shorter side but keep in max_size
        h, w, _ = src.shape
        img = timage.resize_short_within(src, self._short, self._max_size, interp=1)
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
        oshape = self._feat_sym.infer_shape(data=(1, 3, img.shape[1], img.shape[2]))[1][0]
        anchor = self._anchors[:, :, :oshape[2], :oshape[3], :].reshape((-1, 4))
        gt_bboxes = mx.nd.array(bbox[:, :4])
        cls_target, box_target, box_mask = self._target_generator(
            gt_bboxes, anchor, img.shape[2], img.shape[1])
        return img, bbox.astype(img.dtype), masks, cls_target, box_target, box_mask


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
