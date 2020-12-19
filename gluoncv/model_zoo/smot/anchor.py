# pylint: disable=unused-import
"""Anchor box generator for SSD detector."""
from __future__ import absolute_import

import numpy as np
from mxnet import gluon


class SSDAnchorGenerator(gluon.HybridBlock):
    """Bounding box anchor generator for Single-shot Object Detection.

    Parameters
    ----------
    index : int
        Index of this generator in SSD models, this is required for naming.
    sizes : iterable of floats
        Sizes of anchor boxes.
    ratios : iterable of floats
        Aspect ratios of anchor boxes.
    step : int or float
        Step size of anchor boxes.
    alloc_size : tuple of int
        Allocate size for the anchor boxes as (H, W).
        Usually we generate enough anchors for large feature map, e.g. 128x128.
        Later in inference we can have variable input sizes,
        at which time we can crop corresponding anchors from this large
        anchor map so we can skip re-generating anchors for each input.
    offsets : tuple of float
        Center offsets of anchor boxes as (h, w) in range(0, 1).

    """
    def __init__(self, index, im_size, sizes, ratios, step, alloc_size=(128, 128),
                 offsets=(0.5, 0.5), clip=False, **kwargs):
        super(SSDAnchorGenerator, self).__init__(**kwargs)
        assert len(im_size) == 2
        self._im_size = im_size
        self._clip = clip
        self._sizes = sizes
        self._ratios = ratios
        anchors = self._generate_anchors(self._sizes, self._ratios, step, alloc_size, offsets)
        self._num_anchors = np.size(anchors) / 4
        self.anchors = self.params.get_constant('anchor_%d'%(index), anchors)

    def _generate_anchors(self, sizes, ratios, step, alloc_size, offsets):
        # pylint: disable=unused-argument,too-many-function-args
        """Generate anchors for once. Anchors are stored with (center_x, center_y, w, h) format."""
        anchors = []
        for i in range(alloc_size[0]):
            for j in range(alloc_size[1]):
                cy = (i + offsets[0]) * step
                cx = (j + offsets[1]) * step

                for sz in self._sizes:
                    for r in ratios:
                        sr = np.sqrt(r)
                        w = sz * sr
                        h = sz / sr
                        anchors.append([cx, cy, w, h])
        return np.array(anchors).reshape(1, 1, alloc_size[0], alloc_size[1], -1)

    @property
    def num_depth(self):
        """Number of anchors at each pixel."""
        return len(self._sizes) * len(self._ratios)

    @property
    def num_anchors(self):
        """Number of anchors at each pixel."""
        return self._num_anchors

    # pylint: disable=arguments-differ
    def hybrid_forward(self, F, x, anchors):
        a = F.slice_like(anchors, x * 0, axes=(2, 3))
        a = a.reshape((1, -1, 4))
        if self._clip:
            cx, cy, cw, ch = a.split(axis=-1, num_outputs=4)
            H, W = self._im_size
            a = F.concat(*[cx.clip(0, W), cy.clip(0, H), cw.clip(0, W), ch.clip(0, H)], dim=-1)
        return a.reshape((1, -1, 4))
