# pylint: disable=arguments-differ
"""Bounding boxes operators"""
from __future__ import absolute_import

from mxnet import gluon


class BBoxCornerToCenter(gluon.HybridBlock):
    """Convert corner boxes to center boxes.
    Corner boxes are encoded as (xmin, ymin, xmax, ymax)
    Center boxes are encoded as (center_x, center_y, width, height)

    Parameters
    ----------
    split : bool
        Whether split boxes to individual elements after processing.

    Returns
    -------
     A BxNx4 NDArray if split is False, or 4 BxNx1 NDArray if split is True
    """
    def __init__(self, split=False):
        super(BBoxCornerToCenter, self).__init__()
        self._split = split

    def hybrid_forward(self, F, x):
        xmin, ymin, xmax, ymax = F.split(x, axis=-1, num_outputs=4)
        width = xmax - xmin
        height = ymax - ymin
        x = xmin + width / 2
        y = ymin + height / 2
        if not self._split:
            return F.concat(x, y, width, height, dim=-1)
        else:
            return x, y, width, height


class BBoxCenterToCorner(gluon.HybridBlock):
    """Convert center boxes to corner boxes.
    Corner boxes are encoded as (xmin, ymin, xmax, ymax)
    Center boxes are encoded as (center_x, center_y, width, height)

    Parameters
    ----------
    split : bool
        Whether split boxes to individual elements after processing.

    Returns
    -------
     A BxNx4 NDArray if split is False, or 4 BxNx1 NDArray if split is True.
    """
    def __init__(self, split=False):
        super(BBoxCenterToCorner, self).__init__()
        self._split = split

    def hybrid_forward(self, F, x):
        """Hybrid forward"""
        x, y, w, h = F.split(x, axis=-1, num_outputs=4)
        hw = w / 2
        hh = h / 2
        xmin = x - hw
        ymin = y - hh
        xmax = x + hw
        ymax = y + hh
        if not self._split:
            return F.concat(xmin, ymin, xmax, ymax, dim=-1)
        else:
            return xmin, ymin, xmax, ymax
