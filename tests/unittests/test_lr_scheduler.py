from __future__ import print_function
from __future__ import division

import os.path as osp
import mxnet as mx
import numpy as np
import gluoncv as gcv

from mxnet import autograd, gluon
from math import pi, cos
from gluoncv.utils import LRScheduler, LRCompose

def compare(obj, niter, expect):
    np.testing.assert_allclose(expect, obj.__call__(niter))


def test_sanity():
    N = 1000
    constant = LRScheduler('constant', baselr=0, targetlr=1, niters=N)
    linear = LRScheduler('linear', baselr=1, targetlr=2, niters=N)
    cosine = LRScheduler('cosine', baselr=3, targetlr=1, niters=N)
    poly = LRScheduler('poly', baselr=1, targetlr=0, niters=N, power=2)
    step = LRScheduler('step', baselr=1, targetlr=0, niters=N,
                       step=[100, 500], step_factor=0.1)

    compare(constant, 0, 0)
    compare(constant, N-1, 0)
    compare(linear, 0, 1)
    compare(linear, N-1, 2)
    compare(cosine, 0, 3)
    compare(cosine, N-1, 1)
    compare(poly, 0, 1)
    compare(poly, N-1, 0)
    compare(step, 0, 1)
    compare(step, 100, 0.1)
    compare(step, 500, 0.01)
    compare(step, N-1, 0.01)

def test_single_method():
    N = 1000
    constant = LRScheduler('constant', baselr=0, targetlr=1, niters=N)
    linear = LRScheduler('linear', baselr=1, targetlr=2, niters=N)
    cosine = LRScheduler('cosine', baselr=3, targetlr=1, niters=N)
    poly = LRScheduler('poly', baselr=1, targetlr=0, niters=N, power=2)
    step = LRScheduler('step', baselr=1, targetlr=0, niters=N,
                       step=[100, 500], step_factor=0.1)

    # Test numerical value
    for i in range(N):
        compare(constant, i, 0)

        expect_linear = 2 + (1 - 2) * (1 - i / (N - 1))
        compare(linear, i, expect_linear)

        expect_cosine = 1 + (3 - 1) * ((1 + cos(pi * i / (N-1))) / 2)
        compare(cosine, i, expect_cosine)

        expect_poly = 0 + (1 - 0) * (pow(1 - i / (N-1), 2))
        compare(poly, i, expect_poly)

        if i < 100:
            expect_step = 1
        elif i < 500:
            expect_step = 0.1
        else:
            expect_step = 0.01
        compare(step, i, expect_step)

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
    step = LRScheduler('step', baselr=1, targetlr=0, niters=N,
                       step=[100, 500], step_factor=0.1)
    arr = Compose([constant, null_cosine, linear, cosine, null_poly, poly, step])
    # constant
    for i in range(N):
        compare(constant, i, 0)
    # linear
    for i in range(N, 2*N):
        expect_linear = 2 + (1 - 2) * (1 - (i - N) / (N - 1))
        compare(linear, i, expect_linear)
    # cosine
    for i in range(2*N, 3*N):
        expect_cosine = 1 + (3 - 1) * ((1 + cos(pi * (i - 2*N) / (N - 1))) / 2)
        compare(cosine, i, expect_cosine)
    # poly
    for i in range(3*N, 4*N):
        expect_poly = 0 + (1 - 0) * (pow(1 - (i - 3*N) / (N - 1), 2))
        compare(poly, i, expect_poly)
    for i in range(4*N, 5*N):
        if i - 4*N < 100:
            expect_step = 1
        elif i - 4*N < 500:
            expect_step = 0.1
        else:
            expect_step = 0.01
        compare(step, i, expect_step)

if __name__ == '__main__':
    import nose
    nose.runmodule()
