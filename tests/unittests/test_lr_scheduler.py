from __future__ import print_function
from __future__ import division

import os.path as osp
import mxnet as mx
import numpy as np
import gluoncv as gcv

from mxnet import autograd, gluon
from math import pi, cos
from gluoncv.utils import LRScheduler, Compose

def test_sanity():
    N = 1000
    constant = LRScheduler('constant', baselr=0, targetlr=1, niters=N)
    linear = LRScheduler('linear', baselr=1, targetlr=2, niters=N)
    cosine = LRScheduler('cosine', baselr=3, targetlr=1, niters=N)
    poly = LRScheduler('poly', baselr=1, targetlr=0, niters=N, power=2)

    np.testing.assert_allclose(0, constant.__call__(0))
    np.testing.assert_allclose(0, constant.__call__(N-1))
    np.testing.assert_allclose(1, linear.__call__(0))
    np.testing.assert_allclose(2, linear.__call__(N-1))
    np.testing.assert_allclose(3, cosine.__call__(0))
    np.testing.assert_allclose(1, cosine.__call__(N-1))
    np.testing.assert_allclose(1, poly.__call__(0))
    np.testing.assert_allclose(0, poly.__call__(N-1))

def test_single_method():
    N = 1000
    constant = LRScheduler('constant', baselr=0, targetlr=1, niters=N)
    linear = LRScheduler('linear', baselr=1, targetlr=2, niters=N)
    cosine = LRScheduler('cosine', baselr=3, targetlr=1, niters=N)
    poly = LRScheduler('poly', baselr=1, targetlr=0, niters=N, power=2)

    # Test numerical value
    for i in range(N):
        np.testing.assert_allclose(constant.__call__(i), 0)
        expect_linear = 2 + (1 - 2) * (1 - i / (N - 1))
        np.testing.assert_allclose(linear.__call__(i), expect_linear)
        expect_cosine = 1 + (3 - 1) * ((1 + cos(pi * i / (N-1))) / 2)
        np.testing.assert_allclose(cosine.__call__(i), expect_cosine)
        expect_poly = 0 + (1 - 0) * (pow(1 - i / (N-1), 2))
        np.testing.assert_allclose(poly.__call__(i), expect_poly)

    # Test out-of-range updates
    for i in range(10):
        constant.update(i - 3)
        linear.update(i - 3)
        cosine.update(i - 3)
        poly.update(i - 3)

def test_composed_method():
    N = 1000
    constant = LRScheduler('constant', baselr=0, targetlr=1, niters=N)
    linear = LRScheduler('linear', baselr=1, targetlr=2, niters=N)
    cosine = LRScheduler('cosine', baselr=3, targetlr=1, niters=N)
    poly = LRScheduler('poly', baselr=1, targetlr=0, niters=N, power=2)
    # components with niters=0 will be ignored
    null_cosine = LRScheduler('cosine', baselr=3, targetlr=1, niters=0)
    null_poly = LRScheduler('cosine', baselr=3, targetlr=1, niters=0)
    arr = Compose([constant, null_cosine, linear, cosine, null_poly, poly])
    # constant
    for i in range(N):
        np.testing.assert_allclose(arr.__call__(i), 0)
    # linear
    for i in range(N, 2*N):
        expect_linear = 2 + (1 - 2) * (1 - (i - N) / (N - 1))
        np.testing.assert_allclose(arr.__call__(i), expect_linear)
    # cosine
    for i in range(2*N, 3*N):
        expect_cosine = 1 + (3 - 1) * ((1 + cos(pi * (i - 2*N) / (N - 1))) / 2)
        np.testing.assert_allclose(arr.__call__(i), expect_cosine)
    # poly
    for i in range(3*N, 4*N):
        expect_poly = 0 + (1 - 0) * (pow(1 - (i - 3*N) / (N - 1), 2))
        np.testing.assert_allclose(arr.__call__(i), expect_poly)

if __name__ == '__main__':
    import nose
    nose.runmodule()
