"""CenterNet training target generator."""
from __future__ import absolute_import

import numpy as np

from mxnet import nd
from mxnet import gluon


class CenterNetTargetGenerator(gluon.Block):
    """Target generator for CenterNet.

    Parameters
    ----------
    num_class : int
        Number of categories.
    output_width : int
        Width of the network output.
    output_height : int
        Height of the network output.

    """
    def __init__(self, num_class, output_width, output_height):
        super(CenterNetTargetGenerator, self).__init__()
        self._num_class = num_class
        self._output_width = int(output_width)
        self._output_height = int(output_height)

    def forward(self, gt_boxes, gt_ids):
        """Target generation"""
        # pylint: disable=arguments-differ
        h_scale = 1.0  # already scaled in affinetransform
        w_scale = 1.0  # already scaled in affinetransform
        heatmap = np.zeros((self._num_class, self._output_height, self._output_width),
                           dtype=np.float32)
        wh_target = np.zeros((2, self._output_height, self._output_width), dtype=np.float32)
        wh_mask = np.zeros((2, self._output_height, self._output_width), dtype=np.float32)
        center_reg = np.zeros((2, self._output_height, self._output_width), dtype=np.float32)
        center_reg_mask = np.zeros((2, self._output_height, self._output_width), dtype=np.float32)
        for bbox, cid in zip(gt_boxes, gt_ids):
            cid = int(cid)
            box_h, box_w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if box_h > 0 and box_w > 0:
                radius = _gaussian_radius((np.ceil(box_h), np.ceil(box_w)))
                radius = max(0, int(radius))
                center = np.array(
                    [(bbox[0] + bbox[2]) / 2 * w_scale, (bbox[1] + bbox[3]) / 2 * h_scale],
                    dtype=np.float32)
                center_int = center.astype(np.int32)
                center_x, center_y = center_int
                assert center_x < self._output_width, \
                    'center_x: {} > output_width: {}'.format(center_x, self._output_width)
                assert center_y < self._output_height, \
                    'center_y: {} > output_height: {}'.format(center_y, self._output_height)
                _draw_umich_gaussian(heatmap[cid], center_int, radius)
                wh_target[0, center_y, center_x] = box_w * w_scale
                wh_target[1, center_y, center_x] = box_h * h_scale
                wh_mask[:, center_y, center_x] = 1.0
                center_reg[:, center_y, center_x] = center - center_int
                center_reg_mask[:, center_y, center_x] = 1.0
        return tuple([nd.array(x) for x in \
            (heatmap, wh_target, wh_mask, center_reg, center_reg_mask)])


def _gaussian_radius(det_size, min_overlap=0.7):
    """Calculate gaussian radius for foreground objects.

    Parameters
    ----------
    det_size : tuple of int
        Object size (h, w).
    min_overlap : float
        Minimal overlap between objects.

    Returns
    -------
    float
        Gaussian radius.

    """
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)

def _gaussian_2d(shape, sigma=1):
    """Generate 2d gaussian.

    Parameters
    ----------
    shape : tuple of int
        The shape of the gaussian.
    sigma : float
        Sigma for gaussian.

    Returns
    -------
    float
        2D gaussian kernel.

    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def _draw_umich_gaussian(heatmap, center, radius, k=1):
    """Draw a 2D gaussian heatmap.

    Parameters
    ----------
    heatmap : numpy.ndarray
        Heatmap to be write inplace.
    center : tuple of int
        Center of object (h, w).
    radius : type
        The radius of gaussian.

    Returns
    -------
    numpy.ndarray
        Drawn gaussian heatmap.

    """
    diameter = 2 * radius + 1
    gaussian = _gaussian_2d((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap
