"""RPN anchors."""
from __future__ import absolute_import

import numpy as np
from mxnet import gluon


class RPNAnchorGenerator(gluon.HybridBlock):
    r"""Anchor generator for Region Proposal Networks.

    Parameters
    ----------
    stride : int
        Feature map stride with respect to original image.
        This is usually the ratio between original image size and feature map size.
    base_size : int
        The width(and height) of reference anchor box.
    ratios : iterable of float
        The aspect ratios of anchor boxes. We expect it to be a list or tuple.
    scales : iterable of float
        The areas of anchor boxes.
        We use the following form to compute the shapes of anchors:

        .. math::

            width_{anchor} = size_{base} \times scale \times \sqrt{ 1 / ratio}
            height_{anchor} = size_{base} \times scale \times \sqrt{ratio}

    alloc_size : tuple of int
        Allocate size for the anchor boxes as (H, W).
        Usually we generate enough anchors for large feature map, e.g. 128x128.
        Later in inference we can have variable input sizes,
        at which time we can crop corresponding anchors from this large
        anchor map so we can skip re-generating anchors for each input.

    """

    def __init__(self, stride, base_size, ratios, scales, alloc_size, **kwargs):
        super(RPNAnchorGenerator, self).__init__(**kwargs)
        if not base_size:
            raise ValueError("Invalid base_size: {}.".format(base_size))
        if not isinstance(ratios, (tuple, list)):
            ratios = [ratios]
        if not isinstance(scales, (tuple, list)):
            scales = [scales]

        anchors = self._generate_anchors(stride, base_size, ratios, scales, alloc_size)
        self._num_depth = len(ratios) * len(scales)
        self.anchors = self.params.get_constant('anchor_', anchors)

    @property
    def num_depth(self):
        """Number of anchors at each pixel."""
        return self._num_depth

    def _generate_anchors(self, stride, base_size, ratios, scales, alloc_size):
        """Pre-generate all anchors."""
        # generate same shapes on every location
        base_sizes = generate_base_anchors(stride=stride,
                                           sizes=tuple(s * base_size for s in scales),
                                           aspect_ratios=ratios)

        # propagete to all locations by shifting offsets
        height, width = alloc_size
        offset_x = np.arange(0, width * stride, stride)
        offset_y = np.arange(0, height * stride, stride)
        offset_x, offset_y = np.meshgrid(offset_x, offset_y)
        offsets = np.stack((offset_x.ravel(), offset_y.ravel(),
                            offset_x.ravel(), offset_y.ravel()), axis=1)
        # broadcast_add (1, N, 4) + (M, 1, 4)
        anchors = (base_sizes.reshape((1, -1, 4)) + offsets.reshape((-1, 1, 4)))
        anchors = anchors.reshape((1, 1, height, width, -1)).astype(np.float32)
        return anchors

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, x, anchors):
        """Slice anchors given the input image shape.

        Inputs:
            - **x**: input tensor with (1 x C x H x W) shape.
        Outputs:
            - **out**: output anchor with (1, N, 4) shape. N is the number of anchors.

        """
        a = F.slice_like(anchors, x, axes=(2, 3))
        return a.reshape((1, -1, 4))


def generate_base_anchors(stride=16, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2)):
    """Generates a matrix of anchor boxes in (x1, y1, x2, y2) format. Anchors
    are centered on stride / 2, have (approximate) sqrt areas of the specified
    sizes, and aspect ratios as given.
    """
    return _generate_base_anchors(
        stride,
        np.array(sizes, dtype=np.float) / stride,
        np.array(aspect_ratios, dtype=np.float)
    )


def _generate_base_anchors(base_size, scales, aspect_ratios):
    """Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, base_size - 1, base_size - 1) window.
    """
    anchor = np.array([1, 1, base_size, base_size], dtype=np.float) - 1
    anchors = _ratio_enum(anchor, aspect_ratios)
    anchors = np.vstack(
        [_scale_enum(anchors[i, :], scales) for i in range(anchors.shape[0])]
    )
    return anchors


def _whctrs(anchor):
    """Return width, height, x center, and y center for an anchor (window)."""
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack(
        (
            x_ctr - 0.5 * (ws - 1),
            y_ctr - 0.5 * (hs - 1),
            x_ctr + 0.5 * (ws - 1),
            y_ctr + 0.5 * (hs - 1)
        )
    )
    return anchors


def _ratio_enum(anchor, ratios):
    """Enumerate a set of anchors for each aspect ratio wrt an anchor."""
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """Enumerate a set of anchors for each scale wrt an anchor."""
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors
