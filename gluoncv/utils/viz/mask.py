"""Bounding box visualization functions."""
from __future__ import absolute_import, division

import numpy as np
import mxnet as mx

from ...data.transforms.mask import fill

def expand_mask(masks, bboxes, im_shape, scores=None, thresh=0.5):
    """Expand instance segmentation mask to full image size.

    Parameters
    ----------
    masks : numpy.ndarray or mxnet.nd.NDArray
        Binary images with shape `N, M, M`
    bboxes : numpy.ndarray or mxnet.nd.NDArray
        Bounding boxes with shape `N, 4`. Where `N` is the number of boxes
    im_shape : tuple
        Tuple of length 2: (width, height)
    scores : numpy.ndarray or mxnet.nd.NDArray, optional
        Confidence scores of the provided `bboxes` with shape `N`.
    thresh : float, optional, default 0.5
        Display threshold if `scores` is provided. Scores with less than `thresh`
        will be ignored in display, this is visually more elegant if you have
        a large number of bounding boxes with very small scores.

    Returns
    -------
    numpy.ndarray
        Binary images with shape `N, height, width`

    """
    if len(masks) != len(bboxes):
        raise ValueError('The length of bboxes and masks mismatch, {} vs {}'
                         .format(len(bboxes), len(masks)))
    if scores is not None and len(masks) != len(scores):
        raise ValueError('The length of scores and masks mismatch, {} vs {}'
                         .format(len(scores), len(masks)))

    if isinstance(masks, mx.nd.NDArray):
        masks = masks.asnumpy()
    if isinstance(bboxes, mx.nd.NDArray):
        bboxes = bboxes.asnumpy()
    if isinstance(scores, mx.nd.NDArray):
        scores = scores.asnumpy()

    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    sorted_inds = np.argsort(-areas)

    full_masks = []
    for i in sorted_inds:
        if scores is not None and scores[i] < thresh:
            continue
        mask = masks[i]
        bbox = bboxes[i]
        full_masks.append(fill(mask, bbox, im_shape))
    full_masks = np.array(full_masks)
    return full_masks


def plot_mask(img, masks, alpha=0.5):
    """Visualize segmentation mask.

    Parameters
    ----------
    img : numpy.ndarray or mxnet.nd.NDArray
        Image with shape `H, W, 3`.
    masks : numpy.ndarray or mxnet.nd.NDArray
        Binary images with shape `N, H, W`.
    alpha : float, optional, default 0.5
        Transparency of plotted mask

    Returns
    -------
    numpy.ndarray
        The image plotted with segmentation masks

    """
    if isinstance(img, mx.nd.NDArray):
        img = img.asnumpy()
    if isinstance(masks, mx.nd.NDArray):
        masks = masks.asnumpy()

    for mask in masks:
        color = np.random.random(3) * 255
        mask = np.repeat((mask > 0)[:, :, np.newaxis], repeats=3, axis=2)
        img = np.where(mask, img * (1 - alpha) + color * alpha, img)
    return img.astype('uint8')
