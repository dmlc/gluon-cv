"""Mask transformation functions."""
import copy
import mxnet as mx
import numpy as np
from ..mscoco.utils import try_import_pycocotools
# pylint: disable=wrong-import-position,wrong-import-order
try_import_pycocotools()
import pycocotools.mask as cocomask


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
    width, height = size
    polys = [p.flatten().tolist() for p in polys]
    rles = cocomask.frPyObjects(polys, height, width)
    rle = cocomask.merge(rles)
    return cocomask.decode(rle)


def fill(mask, bbox, size):
    """Fill mask to full image size

    Parameters
    ----------
    mask : numpy.ndarray with dtype=uint8
        Binary mask prediction of a box
    bbox : iterable of float
        They are :math:`(xmin, ymin, xmax, ymax)`.
    size : tuple
        Tuple of length 2: (width, height).

    Returns
    -------
    numpy.ndarray
        Full size binary mask of shape (height, width)
    """
    width, height = size
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    x1, y1 = int(x1 + 0.5), int(y1 + 0.5)
    x2, y2 = int(x2 - 0.5), int(y2 - 0.5)
    x2, y2 = max(x1, x2), max(y1, y2)
    w, h = (x2 - x1 + 1), (y2 - y1 + 1)
    mask = mx.nd.array(mask)
    mask = mask.reshape((0, 0, 1))
    mask = mx.image.imresize(mask, w=w, h=h, interp=1)
    mask = mask.reshape((0, 0))
    mask = mask.asnumpy()
    mask = (mask > 0.5).astype('uint8')
    ret = np.zeros((height, width), dtype='uint8')
    ret[y1:y2 + 1, x1:x2 + 1] = mask
    return ret
