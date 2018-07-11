"""RPN anchors."""
from __future__ import absolute_import

from mxnet import gluon
import numpy as np


class RPNAnchorGenerator(gluon.HybridBlock):
    r"""Anchor generator for Region Proposal Netoworks.

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
        px, py = (base_size - 1) * 0.5, (base_size - 1) * 0.5
        base_sizes = []
        for r in ratios:
            for s in scales:
                size = base_size * base_size / r
                ws = np.round(np.sqrt(size))
                w = (ws * s - 1) * 0.5
                h = (np.round(ws * r) * s - 1) * 0.5
                base_sizes.append([px - w, py - h, px + w, py + h])
        base_sizes = np.array(base_sizes)  # (N, 4)

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
        a = F.slice_like(anchors, x * 0, axes=(2, 3))
        return a.reshape((1, -1, 4))
