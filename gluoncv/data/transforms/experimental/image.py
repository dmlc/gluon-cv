"""Experimental image transformations."""
from __future__ import division
import random
import numpy as np
import mxnet as mx
from mxnet import nd

def random_color_distort(src, brightness_delta=32, contrast_low=0.5, contrast_high=1.5,
                         saturation_low=0.5, saturation_high=1.5, hue_delta=18):
    """Randomly distort image color space.
    Note that input image should in original range [0, 255].

    Parameters
    ----------
    src : mxnet.nd.NDArray
        Input image as HWC format.
    brightness_delta : int
        Maximum brightness delta. Defaults to 32.
    contrast_low : float
        Lowest contrast. Defaults to 0.5.
    contrast_high : float
        Highest contrast. Defaults to 1.5.
    saturation_low : float
        Lowest saturation. Defaults to 0.5.
    saturation_high : float
        Highest saturation. Defaults to 1.5.
    hue_delta : int
        Maximum hue delta. Defaults to 18.

    Returns
    -------
    mxnet.nd.NDArray
        Distorted image in HWC format.

    """
    def brightness(src, delta, p=0.5):
        """Brightness distortion."""
        if np.random.uniform(0, 1) > p:
            delta = np.random.uniform(-delta, delta)
            src += delta
            return src
        return src

    def contrast(src, low, high, p=0.5):
        """Contrast distortion"""
        if np.random.uniform(0, 1) > p:
            alpha = np.random.uniform(low, high)
            src *= alpha
            return src
        return src

    def saturation(src, low, high, p=0.5):
        """Saturation distortion."""
        if np.random.uniform(0, 1) > p:
            alpha = np.random.uniform(low, high)
            gray = src * nd.array([[[0.299, 0.587, 0.114]]], ctx=src.context)
            gray = mx.nd.sum(gray, axis=2, keepdims=True)
            gray *= (1.0 - alpha)
            src *= alpha
            src += gray
            return src
        return src

    def hue(src, delta, p=0.5):
        """Hue distortion"""
        if np.random.uniform(0, 1) > p:
            alpha = random.uniform(-delta, delta)
            u = np.cos(alpha * np.pi)
            w = np.sin(alpha * np.pi)
            bt = np.array([[1.0, 0.0, 0.0],
                           [0.0, u, -w],
                           [0.0, w, u]])
            tyiq = np.array([[0.299, 0.587, 0.114],
                             [0.596, -0.274, -0.321],
                             [0.211, -0.523, 0.311]])
            ityiq = np.array([[1.0, 0.956, 0.621],
                              [1.0, -0.272, -0.647],
                              [1.0, -1.107, 1.705]])
            t = np.dot(np.dot(ityiq, bt), tyiq).T
            src = nd.dot(src, nd.array(t, ctx=src.context))
            return src
        return src

    src = src.astype('float32')

    # brightness
    src = brightness(src, brightness_delta)

    # color jitter
    if np.random.randint(0, 2):
        src = contrast(src, contrast_low, contrast_high)
        src = saturation(src, saturation_low, saturation_high)
        src = hue(src, hue_delta)
    else:
        src = saturation(src, saturation_low, saturation_high)
        src = hue(src, hue_delta)
        src = contrast(src, contrast_low, contrast_high)
    return src
