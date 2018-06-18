"""Extended image transformations to `mxnet.image`."""
from __future__ import division
import random
import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet.base import numeric_types

__all__ = ['imresize', 'resize_long', 'resize_short_within',
           'random_pca_lighting', 'random_expand', 'random_flip',
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

    Examples
    --------
    >>> import mxnet as mx
    >>> from gluoncv import data as gdata
    >>> img = mx.random.uniform(0, 255, (300, 300, 3)).astype('uint8')
    >>> print(img.shape)
    (300, 300, 3)
    >>> img = gdata.transforms.image.imresize(img, 200, 200)
    >>> print(img.shape)
    (200, 200, 3)
    """
    return mx.image.imresize(src, w, h, interp)

def resize_long(src, size, interp=2):
    """Resizes longer edge to size.
    Note: `resize_short` uses OpenCV (not the CV2 Python library).
    MXNet must have been built with OpenCV for `resize_short` to work.
    Resizes the original image by setting the longer edge to size
    and setting the shorter edge accordingly. This will ensure the new image will
    fit into the `size` specified.
    Resizing function is called from OpenCV.

    Parameters
    ----------
    src : NDArray
        The original image.
    size : int
        The length to be set for the shorter edge.
    interp : int, optional, default=2
        Interpolation method used for resizing the image.
        Possible values:
        0: Nearest Neighbors Interpolation.
        1: Bilinear interpolation.
        2: Area-based (resampling using pixel area relation). It may be a
        preferred method for image decimation, as it gives moire-free
        results. But when the image is zoomed, it is similar to the Nearest
        Neighbors method. (used by default).
        3: Bicubic interpolation over 4x4 pixel neighborhood.
        4: Lanczos interpolation over 8x8 pixel neighborhood.
        9: Cubic for enlarge, area for shrink, bilinear for others
        10: Random select from interpolation method metioned above.
        Note:
        When shrinking an image, it will generally look best with AREA-based
        interpolation, whereas, when enlarging an image, it will generally look best
        with Bicubic (slow) or Bilinear (faster but still looks OK).
        More details can be found in the documentation of OpenCV, please refer to
        http://docs.opencv.org/master/da/d54/group__imgproc__transform.html.
    Returns
    -------
    NDArray
        An 'NDArray' containing the resized image.
    Example
    -------
    >>> with open("flower.jpeg", 'rb') as fp:
    ...     str_image = fp.read()
    ...
    >>> image = mx.img.imdecode(str_image)
    >>> image
    <NDArray 2321x3482x3 @cpu(0)>
    >>> size = 640
    >>> new_image = mx.img.resize_long(image, size)
    >>> new_image
    <NDArray 386x640x3 @cpu(0)>
    """
    from mxnet.image.image import _get_interp_method as get_interp
    h, w, _ = src.shape
    if h > w:
        new_h, new_w = size, size * w // h
    else:
        new_h, new_w = size * h // w, size
    return imresize(src, new_w, new_h, interp=get_interp(interp, (h, w, new_h, new_w)))

def resize_short_within(src, short, max_size, interp=2):
    """Resizes shorter edge to size but make sure it's capped at maximum size.
    Note: `resize_short_within` uses OpenCV (not the CV2 Python library).
    MXNet must have been built with OpenCV for `resize_short_within` to work.
    Resizes the original image by setting the shorter edge to size
    and setting the longer edge accordingly. Also this function will ensure
    the new image will not exceed ``max_size`` even at the longer side.
    Resizing function is called from OpenCV.

    Parameters
    ----------
    src : NDArray
        The original image.
    short : int
        Resize shorter side to ``short``.
    max_size : int
        Make sure the longer side of new image is smaller than ``max_size``.
    interp : int, optional, default=2
        Interpolation method used for resizing the image.
        Possible values:
        0: Nearest Neighbors Interpolation.
        1: Bilinear interpolation.
        2: Area-based (resampling using pixel area relation). It may be a
        preferred method for image decimation, as it gives moire-free
        results. But when the image is zoomed, it is similar to the Nearest
        Neighbors method. (used by default).
        3: Bicubic interpolation over 4x4 pixel neighborhood.
        4: Lanczos interpolation over 8x8 pixel neighborhood.
        9: Cubic for enlarge, area for shrink, bilinear for others
        10: Random select from interpolation method metioned above.
        Note:
        When shrinking an image, it will generally look best with AREA-based
        interpolation, whereas, when enlarging an image, it will generally look best
        with Bicubic (slow) or Bilinear (faster but still looks OK).
        More details can be found in the documentation of OpenCV, please refer to
        http://docs.opencv.org/master/da/d54/group__imgproc__transform.html.
    Returns
    -------
    NDArray
        An 'NDArray' containing the resized image.
    Example
    -------
    >>> with open("flower.jpeg", 'rb') as fp:
    ...     str_image = fp.read()
    ...
    >>> image = mx.img.imdecode(str_image)
    >>> image
    <NDArray 2321x3482x3 @cpu(0)>
    >>> new_image = mx.img.resize_short_within(image, size=600, max_size=1000)
    >>> new_image
    <NDArray 604x1000x3 @cpu(0)>
    >>> new_image = mx.img.resize_short_within(image, size=600, max_size=1200)
    >>> new_image
    <NDArray 600x993x3 @cpu(0)>
    """
    from mxnet.image.image import _get_interp_method as get_interp
    h, w, _ = src.shape
    im_size_min, im_size_max = (h, w) if w > h else (w, h)
    scale = float(short) / float(im_size_min)
    if np.round(scale * im_size_max) > max_size:
        # fit in max_size
        scale = float(max_size) / float(im_size_max)
    new_w, new_h = int(np.round(w * scale)), int(np.round(h * scale))
    return imresize(src, new_w, new_h, interp=get_interp(interp, (h, w, new_h, new_w)))

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
        return src

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

    off_y = (oh - scaled_y) // 2 if scaled_y < oh else 0
    off_x = (ow - scaled_x) // 2 if scaled_x < ow else 0

    # make canvas
    if isinstance(fill, numeric_types):
        dst = nd.full(shape=(oh, ow, c), val=fill, dtype=src.dtype)
    else:
        fill = nd.array(fill, ctx=src.context)
        if not c == fill.size:
            raise ValueError("Channel and fill size mismatch, {} vs {}".format(c, fill.size))
        dst = nd.repeat(fill, repeats=oh * ow).reshape((oh, ow, c))

    dst[off_y:off_y+scaled_y, off_x:off_x+scaled_x, :] = src
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

    center = src[(h - oh) // 2:(h + oh) // 2, (w - ow) // 2:(w + ow) // 2, :]
    tl = src[0:oh, 0:ow, :]
    bl = src[h - oh:h, 0:ow, :]
    tr = src[0:oh, w - ow:w, :]
    br = src[h - oh:h, w - ow:w, :]
    crops = nd.stack(*[center, tl, bl, tr, br], axis=0)
    crops = nd.concat(*[crops, nd.flip(crops, axis=2)], dim=0)
    return crops
