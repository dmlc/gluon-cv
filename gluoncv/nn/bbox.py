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
    axis : int, default is -1
        Effective axis of the bounding box. Default is -1(the last dimension).

    Returns
    -------
     A BxNx4 NDArray if split is False, or 4 BxNx1 NDArray if split is True
    """
    def __init__(self, axis=-1, split=False):
        super(BBoxCornerToCenter, self).__init__()
        self._split = split
        self._axis = axis

    def hybrid_forward(self, F, x):
        xmin, ymin, xmax, ymax = F.split(x, axis=self._axis, num_outputs=4)
        width = xmax - xmin
        height = ymax - ymin
        x = xmin + width / 2
        y = ymin + height / 2
        if not self._split:
            return F.concat(x, y, width, height, dim=self._axis)
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
    axis : int, default is -1
        Effective axis of the bounding box. Default is -1(the last dimension).

    Returns
    -------
     A BxNx4 NDArray if split is False, or 4 BxNx1 NDArray if split is True.
    """
    def __init__(self, axis=-1, split=False):
        super(BBoxCenterToCorner, self).__init__()
        self._split = split
        self._axis = axis

    def hybrid_forward(self, F, x):
        """Hybrid forward"""
        x, y, w, h = F.split(x, axis=self._axis, num_outputs=4)
        hw = w / 2
        hh = h / 2
        xmin = x - hw
        ymin = y - hh
        xmax = x + hw
        ymax = y + hh
        if not self._split:
            return F.concat(xmin, ymin, xmax, ymax, dim=self._axis)
        else:
            return xmin, ymin, xmax, ymax


class BBoxSplit(gluon.HybridBlock):
    """Split bounding boxes into 4 columns.

    Parameters
    ----------
    axis : int, default is -1
        On which axis to split the bounding box. Default is -1(the last dimension).

    """
    def __init__(self, axis, **kwargs):
        super(BBoxSplit, self).__init__(**kwargs)
        self._axis = axis

    def hybrid_forward(self, F, x):
        return F.split(x, axis=self._axis, num_outputs=4)


class BBoxArea(gluon.HybridBlock):
    """Calculate the area of bounding boxes.

    Parameters
    ----------
    fmt : str, default is corner
        Bounding box format, can be {'center', 'corner'}.
        'center': {x, y, width, height}
        'corner': {xmin, ymin, xmax, ymax}
    axis : int, default is -1
        Effective axis of the bounding box. Default is -1(the last dimension).

    Returns
    -------
    A BxNx1 NDArray

    """
    def __init__(self, axis=-1, fmt='corner', **kwargs):
        super(BBoxArea, self).__init__(**kwargs)
        if fmt.lower() == 'center':
            self._pre = BBoxCornerToCenter(split=True)
        elif fmt.lower() == 'corner':
            self._pre = BBoxSplit(axis=axis)
        else:
            raise ValueError("Unsupported format: {}. Use 'corner' or 'center'.".format(fmt))

    def hybrid_forward(self, F, x):
        _, _, width, height = self._pre(x)
        width = F.where(width > 0, width, F.zeros_like(width))
        height = F.where(height > 0, height, F.zeros_like(height))
        return width * height


class BBoxClipToImage(gluon.HybridBlock):
    """Clip bounding box coordinates to image boundaries.
    If multiple images are supplied and padded, must have additional inputs
    of accurate image shape.
    """
    def __init__(self, **kwargs):
        super(BBoxClipToImage, self).__init__(**kwargs)

    def hybrid_forward(self, F, x, img):
        """If images are padded, must have additional inputs for clipping

        Parameters
        ----------
        x: (B, N, 4) Bounding box coordinates.
        img: (B, C, H, W) Image tensor.

        Returns
        -------
        (B, N, 4) Bounding box coordinates.

        """
        x = F.maximum(x, 0.0)
        # window [B, 2] -> reverse hw -> tile [B, 4] -> [B, 1, 4], boxes [B, N, 4]
        window = F.shape_array(img).slice_axis(axis=0, begin=2, end=None).expand_dims(0)
        m = F.tile(F.reverse(window, axis=1), reps=(2,)).reshape((0, -4, 1, -1))
        return F.broadcast_minimum(x, F.cast(m, dtype='float32'))
