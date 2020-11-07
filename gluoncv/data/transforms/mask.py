"""Mask transformation functions."""
import copy

import numpy as np
from PIL import Image

from ..mscoco.utils import try_import_pycocotools

__all__ = ['flip', 'resize', 'to_mask', 'fill']


def flip(polys, size, flip_x=False, flip_y=False):
    """Flip polygons according to image flipping directions.

    Parameters
    ----------
    polys : list of numpy.ndarray
        Numpy.ndarray with shape (N, 2) where N is the number of bounding boxes.
        The second axis represents points of the polygons.
        Specifically, these are :math:`(x, y)`.
    size : tuple
        Tuple of length 2: (width, height).
    flip_x : bool
        Whether flip horizontally.
    flip_y : type
        Whether flip vertically.

    Returns
    -------
    list of numpy.ndarray
        Flipped polygons with original shape.
    """
    if not len(size) == 2:
        raise ValueError("size requires length 2 tuple, given {}".format(len(size)))
    width, height = size
    polys = copy.deepcopy(polys)
    if flip_y:
        for poly in polys:
            poly[:, 1] = height - poly[:, 1]
    if flip_x:
        for poly in polys:
            poly[:, 0] = width - poly[:, 0]
    return polys


def resize(polys, in_size, out_size):
    """Resize polygons according to image resize operation.

    Parameters
    ----------
    polys : list of numpy.ndarray
        Numpy.ndarray with shape (N, 2) where N is the number of bounding boxes.
        The second axis represents points of the polygons.
        Specifically, these are :math:`(x, y)`.
    in_size : tuple
        Tuple of length 2: (width, height) for input.
    out_size : tuple
        Tuple of length 2: (width, height) for output.

    Returns
    -------
    list of numpy.ndarray
        Resized polygons with original shape.
    """
    if not len(in_size) == 2:
        raise ValueError("in_size requires length 2 tuple, given {}".format(len(in_size)))
    if not len(out_size) == 2:
        raise ValueError("out_size requires length 2 tuple, given {}".format(len(out_size)))

    polys = copy.deepcopy(polys)
    x_scale = 1.0 * out_size[0] / in_size[0]
    y_scale = 1.0 * out_size[1] / in_size[1]
    for poly in polys:
        poly[:, 0] = x_scale * poly[:, 0]
        poly[:, 1] = y_scale * poly[:, 1]
    return polys


def to_mask(polys, size):
    """Convert list of polygons to full size binary mask

    Parameters
    ----------
    polys : list of numpy.ndarray
        Numpy.ndarray with shape (N, 2) where N is the number of bounding boxes.
        The second axis represents points of the polygons.
        Specifically, these are :math:`(x, y)`.
    size : tuple
        Tuple of length 2: (width, height).

    Returns
    -------
    numpy.ndarray
        Full size binary mask of shape (height, width)
    """
    try_import_pycocotools()
    import pycocotools.mask as cocomask
    width, height = size
    polys = [p.flatten().tolist() for p in polys]
    rles = cocomask.frPyObjects(polys, height, width)
    rle = cocomask.merge(rles)
    return cocomask.decode(rle)


def fill(masks, bboxes, size, fast_fill=True):
    """Fill mask to full image size

    Parameters
    ----------
    mask : numpy.ndarray with dtype=uint8
        Binary mask prediction of a box
    bbox : numpy.ndarray of float
        They are :math:`(xmin, ymin, xmax, ymax)`.
    size : tuple
        Tuple of length 2: (width, height).
    fast_fill : boolean, default is True.
        Whether to use fast fill. Fast fill is less accurate.

    Returns
    -------
    numpy.ndarray
        Full size binary mask of shape (height, width)
    """
    from scipy import interpolate

    width, height = size
    x1, y1, x2, y2 = np.split(bboxes, 4, axis=1)
    m_h, m_w = masks.shape[1:]
    x1, y1, x2, y2 = x1.reshape((-1,)), y1.reshape((-1,)), x2.reshape((-1,)), y2.reshape((-1,))
    # pad mask
    masks = np.pad(masks, [(0, 0), (1, 1), (1, 1)], mode='constant').astype(np.float32)
    # expand boxes
    x, y, hw, hh = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) / 2, (y2 - y1) / 2
    hw = hw * (float(m_w + 2) / m_w)
    hh = hh * (float(m_h + 2) / m_h)
    x1, y1, x2, y2 = x - hw, y - hh, x + hw, y + hh
    ret = np.zeros((masks.shape[0], height, width), dtype='uint8')
    if fast_fill:
        # quantize
        x1, y1, x2, y2 = np.round(x1).astype(int), np.round(y1).astype(int), \
                         np.round(x2).astype(int), np.round(y2).astype(int)
        w, h = (x2 - x1 + 1), (y2 - y1 + 1)
        xx1, yy1 = np.maximum(0, x1), np.maximum(0, y1)
        xx2, yy2 = np.minimum(width, x2 + 1), np.minimum(height, y2 + 1)
        for i, mask in enumerate(masks):
            mask = Image.fromarray(mask)
            mask = np.array(mask.resize((w[i], h[i]), Image.BILINEAR))
            # binarize and fill
            mask = (mask > 0.5).astype('uint8')
            ret[i, yy1[i]:yy2[i], xx1[i]:xx2[i]] = \
                mask[yy1[i] - y1[i]:yy2[i] - y1[i], xx1[i] - x1[i]:xx2[i] - x1[i]]
        return ret
    for i, mask in enumerate(masks):
        # resize mask
        mask_pixels = np.arange(0.5, mask.shape[0] + 0.5)
        mask_continuous = interpolate.interp2d(mask_pixels, mask_pixels, mask, fill_value=0.0)
        ys = np.arange(0.5, height + 0.5)
        xs = np.arange(0.5, width + 0.5)
        ys = (ys - y1[i]) / (y2[i] - y1[i]) * mask.shape[0]
        xs = (xs - x1[i]) / (x2[i] - x1[i]) * mask.shape[1]
        res = mask_continuous(xs, ys)
        ret[i, :, :] = (res >= 0.5).astype('uint8')
    return ret
