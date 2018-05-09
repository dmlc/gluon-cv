"""RPN anchors."""
from __future__ import absolute_import

from mxnet import gluon
import numpy as np


class RPNAnchorGenerator(gluon.HybridBlock):
    """Anchor generator for Region Proposal Netoworks.

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
    def __init__(self, stride, base_size=16, ratios=(0.5, 1, 2), scales=(8, 16, 32),
                 alloc_size=(128, 128), **kwargs):
        super(RPNAnchorGenerator, self).__init__(**kwargs)
        if not base_size:
            raise ValueError("Invalid base_size: {}.".format(base_size))
        if not isinstance(ratios, (tuple, list)):
            ratios = [ratios]
        if not isinstance(scales, (tuple, list)):
            scales = [scales]

        anchors = self._generate_anchors(base_size, ratios, scales, alloc_size)
        self.anchors = self.params.get_constant('anchor_', anchors)

    def _generate_anchors(self, stride, base_size, ratios, scales, alloc_size):
        # generate same shapes on every location
        px, py = base_size / 2., base_size / 2.
        base_sizes = []
        for i, r in enumerate(ratios):
            for j, s in enumerate(scales):
                h = base_size * s * np.sqrt(r) / 2.
                w = base_size * s * np.sqrt(1. / r) / 2.
                base_size.append([px - w, py - h, px + w, py + h])
        base_sizes = np.array(base_sizes)  # (N, 4)

        # propagete to all locations by shift offsets
        height, width = alloc_size
        offset_x = np.arange(0, width * stride, stride)
        offset_y = np.arange(0, height * stride, stride)
        offset_x, offset_y = np.meshgrid(offset_x, offset_y)
        offsets = np.stack((offset_x.ravel(), offset_y.ravel(),
                            offset_x.ravel(), offset_y.ravel()), axis=1)
        # broadcast_add (1, N, 4) + (M, 1, 4)
        anchors = (base_sizes.reshape((1, -1, 4))
            + offsets.reshape((1, -1, 4)).transpose((1, 0, 2)))
        anchors = anchors.reshape((-1, 4)).astype(np.float32)
        return anchors

    def hybrid_forward(self, F, x, anchors):
        a = F.slice_like(anchors, x * 0, axes=(2, 3))
        return a.reshape((1, -1, 4))
