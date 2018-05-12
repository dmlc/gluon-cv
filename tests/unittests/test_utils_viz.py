from __future__ import print_function
from __future__ import division

import mxnet as mx
import numpy as np

import gluoncv as gcv

def test_viz_bbox():
    img = mx.nd.zeros((300, 300, 3), dtype=np.uint8)
    bbox = mx.nd.array([[10, 20, 200, 500], [150, 200, 400, 300]])
    scores = mx.nd.array([0.8, 0.001])
    labels = mx.nd.array([1, 3])
    class_names = ['a', 'b', 'c']
    ax = gcv.utils.viz.plot_bbox(img, bbox, scores=scores, labels=labels, class_names=class_names)
    ax = gcv.utils.viz.plot_bbox(img, bbox, ax=ax, reverse_rgb=True)
    ax = gcv.utils.viz.plot_bbox(img, bbox / 500, ax=ax, reverse_rgb=True, absolute_coordinates=False)

def test_viz_image():
    img = mx.nd.zeros((300, 300, 3), dtype=np.uint8)
    ax = gcv.utils.viz.plot_image(img)
    ax = gcv.utils.viz.plot_image(img, ax=ax, reverse_rgb=True)

if __name__ == '__main__':
    import nose
    nose.runmodule()
