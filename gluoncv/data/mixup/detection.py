# pylint: disable=keyword-arg-before-vararg
"""Mixup detection dataset wrapper."""
from __future__ import absolute_import
import numpy as np
import mxnet as mx
from mxnet.gluon.data import Dataset


class MixupDetection(Dataset):
    """Detection dataset wrapper that performs mixup for normal dataset.

    Parameters
    ----------
    dataset : mx.gluon.data.Dataset
        Gluon dataset object.
    mixup : callable random generator, e.g. np.random.uniform
        A random mixup ratio sampler, preferably a random generator from numpy.random
        A random float will be sampled each time with mixup(*args).
        Use None to disable.
    *args : list
        Additional arguments for mixup random sampler.

    """
    def __init__(self, dataset, mixup=None, *args):
        self._dataset = dataset
        self._mixup = mixup
        self._mixup_args = args

    def set_mixup(self, mixup=None, *args):
        """Set mixup random sampler, use None to disable.

        Parameters
        ----------
        mixup : callable random generator, e.g. np.random.uniform
            A random mixup ratio sampler, preferably a random generator from numpy.random
            A random float will be sampled each time with mixup(*args)
        *args : list
            Additional arguments for mixup random sampler.

        """
        self._mixup = mixup
        self._mixup_args = args

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        # first image
        img1, label1 = self._dataset[idx]
        lambd = 1

        # draw a random lambda ratio from distribution
        if self._mixup is not None:
            lambd = max(0, min(1, self._mixup(*self._mixup_args)))

        if lambd >= 1:
            weights1 = np.ones((label1.shape[0], 1))
            label1 = np.hstack((label1, weights1))
            return img1, label1

        # second image
        idx2 = np.random.choice(np.delete(np.arange(len(self)), idx))
        img2, label2 = self._dataset[idx2]

        # mixup two images
        height = max(img1.shape[0], img2.shape[0])
        width = max(img1.shape[1], img2.shape[1])
        mix_img = mx.nd.zeros(shape=(height, width, 3), dtype='float32')
        mix_img[:img1.shape[0], :img1.shape[1], :] = img1.astype('float32') * lambd
        mix_img[:img2.shape[0], :img2.shape[1], :] += img2.astype('float32') * (1. - lambd)
        mix_img = mix_img.astype('uint8')
        y1 = np.hstack((label1, np.full((label1.shape[0], 1), lambd)))
        y2 = np.hstack((label2, np.full((label2.shape[0], 1), 1. - lambd)))
        mix_label = np.vstack((y1, y2))
        return mix_img, mix_label
