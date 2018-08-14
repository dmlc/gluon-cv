"""Bounding box visualization functions."""
from __future__ import absolute_import, division

import mxnet as mx
import numpy as np

from ...data.transforms.mask import fill

def expand_mask(masks, bboxes, im_shape):
    """Expand instance segmentation mask to full image size.

    Parameters
    ----------
    masks : numpy.ndarray or mxnet.nd.NDArray
        Binary images with shape `N, M, M`
    bboxes : numpy.ndarray or mxnet.nd.NDArray
        Bounding boxes with shape `N, 4`. Where `N` is the number of boxes
    im_shape : tuple
        Tuple of length 2: (width, height)

    Returns
    -------
    numpy.ndarray
        Binary images with shape `N, height, width`

    """
    if len(masks) != len(bboxes):
        raise ValueError('The length of bboxes and masks mismatch, {} vs {}'
                         .format(len(bboxes), len(masks)))

    if isinstance(masks, mx.nd.NDArray):
        masks = masks.asnumpy()
    if isinstance(bboxes, mx.nd.NDArray):
        bboxes = bboxes.asnumpy()

    full_masks = []
    for mask, bbox in zip(masks, bboxes):
        full_masks.append(fill(mask, bbox, im_shape))
    full_masks = np.array(full_masks)
    return full_masks


def plot_mask(img, masks, scores=None, thresh=0.5, alpha=0.5):
    """Visualize segmentation mask.

    Parameters
    ----------
    img : numpy.ndarray or mxnet.nd.NDArray
        Image with shape `H, W, 3`.
    masks : numpy.ndarray or mxnet.nd.NDArray
        Binary images with shape `N, H, W`.
    scores : numpy.ndarray or mxnet.nd.NDArray, optional
        Confidence scores of the provided `bboxes` with shape `N`.
    thresh : float, optional, default 0.5
        Display threshold if `scores` is provided. Scores with less than `thresh`
        will be ignored in display, this is visually more elegant if you have
        a large number of bounding boxes with very small scores.
    alpha : float, optional, default 0.5
        Transparency of plotted mask

    Returns
    -------
    numpy.ndarray
        The image plotted with segmentation masks

    """
    if scores is not None and not len(masks) == len(scores):
        raise ValueError('The length of scores and masks mismatch, {} vs {}'
                         .format(len(scores), len(masks)))

    if isinstance(img, mx.nd.NDArray):
        img = img.asnumpy()
    if isinstance(masks, mx.nd.NDArray):
        masks = masks.asnumpy()
    if isinstance(scores, mx.nd.NDArray):
        scores = scores.asnumpy()

    for i, mask in enumerate(masks):
        if scores is not None and scores.flat[i] < thresh:
            continue

        color = np.random.random(3) * 255
        mask = np.repeat((mask > 0)[:, :, np.newaxis], repeats=3, axis=2)
        img = np.where(mask, img * (1 - alpha) + color * alpha, img)
    return img.astype('uint8')
