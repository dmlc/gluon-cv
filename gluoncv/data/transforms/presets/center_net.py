"""Transforms described in https://arxiv.org/abs/1904.07850."""
# pylint: disable=too-many-function-args
from __future__ import absolute_import
import numpy as np
import mxnet as mx
from .. import bbox as tbbox
from .. import image as timage
from .. import experimental
from ....utils.filesystem import try_import_cv2

__all__ = ['transform_test', 'load_test',
           'CenterNetDefaultTrainTransform', 'CenterNetDefaultValTransform',
           'get_post_transform']

def transform_test(imgs, short, max_size=1024, mean=(0.485, 0.456, 0.406),
                   std=(0.229, 0.224, 0.225)):
    """A util function to transform all images to tensors as network input by applying
    normalizations. This function support 1 NDArray or iterable of NDArrays.

    Parameters
    ----------
    imgs : NDArray or iterable of NDArray
        Image(s) to be transformed.
    short : int
        Resize image short side to this `short` and keep aspect ratio.
    max_size : int, optional
        Maximum longer side length to fit image.
        This is to limit the input image shape. Aspect ratio is intact because we
        support arbitrary input size in our SSD implementation.
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

def load_test(filenames, short, max_size=1024, mean=(0.485, 0.456, 0.406),
              std=(0.229, 0.224, 0.225)):
    """A util function to load all images, transform them to tensor by applying
    normalizations. This function support 1 filename or iterable of filenames.

    Parameters
    ----------
    filenames : str or list of str
        Image filename(s) to be loaded.
    short : int
        Resize image short side to this `short` and keep aspect ratio.
    max_size : int, optional
        Maximum longer side length to fit image.
        This is to limit the input image shape. Aspect ratio is intact because we
        support arbitrary input size in our SSD implementation.
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


class CenterNetDefaultTrainTransform(object):
    """Default SSD training transform which includes tons of image augmentations.

    Parameters
    ----------
    width : int
        Image width.
    height : int
        Image height.
    num_class : int
        Number of categories
    scale_factor : int, default is 4
        The downsampling scale factor between input image and output heatmap
    mean : array-like of size 3
        Mean pixel values to be subtracted from image tensor. Default is [0.485, 0.456, 0.406].
    std : array-like of size 3
        Standard deviation to be divided from image. Default is [0.229, 0.224, 0.225].
    """
    def __init__(self, width, height, num_class, scale_factor=4, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225), **kwargs):
        self._kwargs = kwargs
        self._width = width
        self._height = height
        self._num_class = num_class
        self._scale_factor = scale_factor
        self._mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self._std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)

        from ....model_zoo.center_net.target_generator import CenterNetTargetGenerator
        self._target_generator = CenterNetTargetGenerator(
            num_class, width // scale_factor, height // scale_factor)

    def __call__(self, src, label):
        """Apply transform to training image/label."""
        # random color jittering
        img = src
        bbox = label

        # random horizontal flip
        h, w, _ = img.shape
        img, flips = timage.random_flip(img, px=0.5)
        bbox = tbbox.flip(bbox, (w, h), flip_x=flips[0])

        cv2 = try_import_cv2()
        input_h, input_w = self._height, self._width
        s = max(h, w) * 1.0
        c = np.array([w / 2., h / 2.], dtype=np.float32)
        sf = 0.4
        w_border = _get_border(128, img.shape[1])
        h_border = _get_border(128, img.shape[0])
        c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
        c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
        s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
        trans_input = tbbox.get_affine_transform(c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img.asnumpy(), trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)
        output_w = input_w // self._scale_factor
        output_h = input_h // self._scale_factor
        trans_output = tbbox.get_affine_transform(c, s, 0, [output_w, output_h])
        for i in range(bbox.shape[0]):
            bbox[i, :2] = tbbox.affine_transform(bbox[i, :2], trans_output)
            bbox[i, 2:4] = tbbox.affine_transform(bbox[i, 2:4], trans_output)
        bbox[:, :2] = np.clip(bbox[:, :2], 0, output_w - 1)
        bbox[:, 2:4] = np.clip(bbox[:, 2:4], 0, output_h - 1)
        img = inp

        # to tensor
        img = img.astype(np.float32) / 255.
        experimental.image.np_random_color_distort(img, data_rng=self._data_rng)
        img = (img - self._mean) / self._std
        img = img.transpose(2, 0, 1).astype(np.float32)
        img = mx.nd.array(img)

        # generate training target so cpu workers can help reduce the workload on gpu
        gt_bboxes = bbox[:, :4]
        gt_ids = bbox[:, 4:5]
        heatmap, wh_target, wh_mask, center_reg, center_reg_mask = self._target_generator(
            gt_bboxes, gt_ids)
        return img, heatmap, wh_target, wh_mask, center_reg, center_reg_mask


class CenterNetDefaultValTransform(object):
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
        self._mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self._std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    def __call__(self, src, label):
        """Apply transform to validation image/label."""
        # resize
        img, bbox = src.asnumpy(), label
        cv2 = try_import_cv2()
        input_h, input_w = self._height, self._width
        h, w, _ = src.shape
        s = max(h, w) * 1.0
        c = np.array([w / 2., h / 2.], dtype=np.float32)
        trans_input = tbbox.get_affine_transform(c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)
        output_w = input_w
        output_h = input_h
        trans_output = tbbox.get_affine_transform(c, s, 0, [output_w, output_h])
        for i in range(bbox.shape[0]):
            bbox[i, :2] = tbbox.affine_transform(bbox[i, :2], trans_output)
            bbox[i, 2:4] = tbbox.affine_transform(bbox[i, 2:4], trans_output)
        bbox[:, :2] = np.clip(bbox[:, :2], 0, output_w - 1)
        bbox[:, 2:4] = np.clip(bbox[:, 2:4], 0, output_h - 1)
        img = inp

        # to tensor
        img = img.astype(np.float32) / 255.
        img = (img - self._mean) / self._std
        img = img.transpose(2, 0, 1).astype(np.float32)
        img = mx.nd.array(img)
        return img, bbox.astype(img.dtype)

def get_post_transform(orig_w, orig_h, out_w, out_h):
    """Get the post prediction affine transforms. This will be used to adjust the prediction results
    according to original coco image resolutions.

    Parameters
    ----------
    orig_w : int
        Original width of the image.
    orig_h : int
        Original height of the image.
    out_w : int
        Width of the output image after prediction.
    out_h : int
        Height of the output image after prediction.

    Returns
    -------
    numpy.ndarray
        Affine transform matrix 3x2.

    """
    s = max(orig_w, orig_h) * 1.0
    c = np.array([orig_w / 2., orig_h / 2.], dtype=np.float32)
    trans_output = tbbox.get_affine_transform(c, s, 0, [out_w, out_h], inv=True)
    return trans_output

def _get_border(border, size):
    """Get the border size of the image"""
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i
