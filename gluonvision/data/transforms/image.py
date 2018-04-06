"""Extended image transformations to `mxnet.image`."""
from __future__ import division
import numpy
import random
import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet.base import numeric_types

__all__ = ['imresize', 'random_pca_lighting', 'random_expand', 'random_flip',
           'resize_contain', 'ten_crop']

def imresize(src, w, h, interp=1):
    """Resize image with OpenCV.

    This is a duplicate of mxnet.image.imresize for name space consistancy.

    Parameters
    ----------
    src : mxnet.nd.NDArray
        source image
    w : int, required
        Width of resized image.
    h : int, required
        Height of resized image.
    interp : int, optional, default='1'
        Interpolation method (default=cv2.INTER_LINEAR).

    out : NDArray, optional
        The output NDArray to hold the result.

    Returns
    -------
    out : NDArray or list of NDArrays
        The output of this function.
    """
    return mx.image.imresize(src, w, h, interp)

def random_pca_lighting(src, alphastd, eigval=None, eigvec=None):
    """Apply random pca lighting noise to input image.

    Parameters
    ----------
    img : mxnet.nd.NDArray
        Input image with HWC format.
    alphastd : float
        Noise level [0, 1) for image with range [0, 255].
    eigval : list of floats.
        Eigen values, defaults to [55.46, 4.794, 1.148].
    eigvec : nested lists of floats
        Eigen vectors with shape (3, 3), defaults to
        [[-0.5675, 0.7192, 0.4009],
         [-0.5808, -0.0045, -0.8140],
         [-0.5836, -0.6948, 0.4203]].

    Returns
    -------
    mxnet.nd.NDArray
        Augmented image.

    """
    if alphastd <= 0:
        return img

    if eigval is None:
        eigval = np.array([55.46, 4.794, 1.148])
    if eigvec is None:
        eigvec = np.array([[-0.5675, 0.7192, 0.4009],
                           [-0.5808, -0.0045, -0.8140],
                           [-0.5836, -0.6948, 0.4203]])

    alpha = np.random.normal(0, alphastd, size=(3,))
    rgb = np.dot(eigvec * alpha, eigval)
    src += nd.array(rgb, ctx=src.context)
    return src

def random_expand(src, max_ratio=4, fill=0, keep_ratio=True):
    """Random expand original image with borders, this is identical to placing
    the original image on a larger canvas.

    Parameters
    ----------
    src : mxnet.nd.NDArray
        The original image with HWC format.
    max_ratio : int or float
        Maximum ratio of the output image on both direction(vertical and horizontal)
    fill : int or float or array-like
        The value(s) for padded borders. If `fill` is numerical type, RGB channels
        will be padded with single value. Otherwise `fill` must have same length
        as image channels, which resulted in padding with per-channel values.
    keep_ratio : bool
        If `True`, will keep output image the same aspect ratio as input.

    Returns
    -------
    mxnet.nd.NDArray
        Augmented image.
    tuple
        Tuple of (offset_x, offset_y, new_width, new_height)

    """
    if max_ratio <= 1:
        return src, (0, 0, src.shape[1], src.shape[0])

    h, w, c = src.shape
    ratio_x = random.uniform(1, max_ratio)
    if keep_ratio:
        ratio_y = ratio_x
    else:
        ratio_y = random.uniform(1, max_ratio)

    oh, ow = int(h * ratio_y), int(w * ratio_x)
    off_y = random.randint(0, oh - h)
    off_x = random.randint(0, ow - w)

    # make canvas
    if isinstance(fill, numeric_types):
        dst = nd.full(shape=(oh, ow, c), val=fill, dtype=src.dtype)
    else:
        fill = nd.array(fill, dtype=src.dtype, ctx=src.context)
        if not c == fill.size:
            raise ValueError("Channel and fill size mismatch, {} vs {}".format(c, fill.size))
        dst = nd.tile(fill.reshape((1, c)), reps=(oh * ow, 1)).reshape((oh, ow, c))

    dst[off_y:off_y+h, off_x:off_x+w, :] = src
    return dst, (off_x, off_y, ow, oh)

def random_flip(src, px=0, py=0, copy=False):
    """Randomly flip image along horizontal and vertical with probabilities.

    Parameters
    ----------
    src : mxnet.nd.NDArray
        Input image with HWC format.
    px : float
        Horizontal flip probability [0, 1].
    py : float
        Vertical flip probability [0, 1].
    copy : bool
        If `True`, return a copy of input

    Returns
    -------
    mxnet.nd.NDArray
        Augmented image.
    tuple
        Tuple of (flip_x, flip_y), records of whether flips are applied.

    """
    flip_y = np.random.choice([False, True], p=[1-py, py])
    flip_x = np.random.choice([False, True], p=[1-px, px])
    if flip_y:
        src = nd.flip(src, axis=0)
    if flip_x:
        src = nd.flip(src, axis=1)
    if copy:
        src = src.copy()
    return src, (flip_x, flip_y)

def resize_contain(src, size, fill=0):
    """Resize the image to fit in the given area while keeping aspect ratio.

    If both the height and the width in `size` are larger than
    the height and the width of input image, the image is placed on
    the center with an appropriate padding to match `size`.
    Otherwise, the input image is scaled to fit in a canvas whose size
    is `size` while preserving aspect ratio.

    Parameters
    ----------
    src : mxnet.nd.NDArray
        The original image with HWC format.
    size : tuple
        Tuple of length 2 as (width, height).
    fill : int or float or array-like
        The value(s) for padded borders. If `fill` is numerical type, RGB channels
        will be padded with single value. Otherwise `fill` must have same length
        as image channels, which resulted in padding with per-channel values.

    Returns
    -------
    mxnet.nd.NDArray
        Augmented image.
    tuple
        Tuple of (offset_x, offset_y, scaled_x, scaled_y)

    """
    h, w, c = src.shape
    ow, oh = size
    scale_h = oh / h
    scale_w = oh / w
    scale = min(min(scale_h, scale_w), 1)
    scaled_x = int(w * scale)
    scaled_y = int(h * scale)
    if scale < 1:
        src = mx.image.imresize(src, scaled_x, scaled_y)

    off_y = (oh - h) // 2 if h < oh else 0
    off_x = (ow - h) // 2 if w < ow else 0

    # make canvas
    if isinstance(fill, numeric_types):
        dst = nd.full(shape=(oh, ow, c), val=fill, dtype=src.dtype)
    else:
        fill = nd.array(fill, ctx=src.context)
        if not c == fill.size:
            raise ValueError("Channel and fill size mismatch, {} vs {}".format(c, fill.size))
        dst = nd.repeat(fill, repeats=oh * ow).reshape((oh, ow, c))

    dst[off_y:off_y+h, off_x:off_x+w, :] = src
    return dst, (off_x, off_y, scaled_x, scaled_y)

def ten_crop(src, size):
    """Crop 10 regions from an array.
    This is performed same as:
    http://chainercv.readthedocs.io/en/stable/reference/transforms.html#ten-crop

    This method crops 10 regions. All regions will be in shape
    :obj`size`. These regions consist of 1 center crop and 4 corner
    crops and horizontal flips of them.
    The crops are ordered in this order.
    * center crop
    * top-left crop
    * bottom-left crop
    * top-right crop
    * bottom-right crop
    * center crop (flipped horizontally)
    * top-left crop (flipped horizontally)
    * bottom-left crop (flipped horizontally)
    * top-right crop (flipped horizontally)
    * bottom-right crop (flipped horizontally)

    Parameters
    ----------
    src : mxnet.nd.NDArray
        Input image.
    size : tuple
        Tuple of length 2, as (width, height) of the cropped areas.

    Returns
    -------
    mxnet.nd.NDArray
        The cropped images with shape (10, size[1], size[0], C)

    """
    h, w, _ = src.shape
    ow, oh = size

    if h < oh or w < ow:
        raise ValueError(
            "Cannot crop area {} from image with size ({}, {})".format(str(size), h, w))

    center = src[:, (h - oh) // 2:(h + oh) // 2, (w - ow) // 2:(w + ow) // 2]
    tl = src[:, 0:oh, 0:ow]
    bl = src[:, h - oh:h, 0:ow]
    tr = src[:, 0:oh, w - ow:w]
    br = src[:, h - oh:h, w - ow:w]
    crops = nd.stack([center, tl, bl, tr, br], axis=0)
    crops = nd.concat(*[crops, nd.flip(crops, axis=2)], dim=0)
    return crops
