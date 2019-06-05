"""Bounding box visualization functions."""
from __future__ import absolute_import, division

import numpy as np
import mxnet as mx

from ...data.transforms.mask import fill

def expand_mask(masks, bboxes, im_shape, scores=None, thresh=0.5, scale=1.0, sortby=None):
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
    sortby : str, optional, default None
        If not None, sort the color palette for masks by the given attributes of each bounding box.
        Valid inputs are 'area', 'xmin', 'ymin', 'xmax', 'ymax'.
    scale : float
        The scale of output image, which may affect the positions of boxes

    Returns
    -------
    numpy.ndarray
        Binary images with shape `N, height, width`
    numpy.ndarray
        Index array of sorted masks
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

    if sortby is not None:
        if sortby == 'area':
            areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            sorted_inds = np.argsort(-areas)
        elif sortby == 'xmin':
            sorted_inds = np.argsort(-bboxes[:, 0])
        elif sortby == 'ymin':
            sorted_inds = np.argsort(-bboxes[:, 1])
        elif sortby == 'xmax':
            sorted_inds = np.argsort(-bboxes[:, 2])
        elif sortby == 'ymax':
            sorted_inds = np.argsort(-bboxes[:, 3])
        else:
            raise ValueError('argument sortby cannot take value {}'
                             .format(sortby))
    else:
        sorted_inds = np.argsort(range(len(masks)))

    full_masks = []
    bboxes *= scale
    for i in sorted_inds:
        if scores is not None and scores[i] < thresh:
            continue
        mask = masks[i]
        bbox = bboxes[i]
        full_masks.append(fill(mask, bbox, im_shape))
    full_masks = np.array(full_masks)
    return full_masks, sorted_inds


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

def cv_merge_two_images(img1, img2, alpha=0.5, size=None):
    """Merge two images with OpoenCV.

    Parameters
    ----------
    img1 : numpy.ndarray or mxnet.nd.NDArray
        Image with shape `H, W, 3`.
    img2 : numpy.ndarray or mxnet.nd.NDArray
        Image with shape `H, W, 3`.
    alpha : float, optional, default 0.5
        Transparency of `img2`
    size : list, optional, default None
        The output size of the merged image

    Returns
    -------
    numpy.ndarray
        The merged image

    """
    from ..filesystem import try_import_cv2
    cv2 = try_import_cv2()

    if isinstance(img1, mx.nd.NDArray):
        img1 = img1.asnumpy()
    if isinstance(img2, mx.nd.NDArray):
        img2 = img2.asnumpy()
    img = cv2.addWeighted(img1, 1-alpha, img2, alpha, 0)
    if size is not None:
        img = cv2.resize(img, (int(size[1]), int(size[0])))
    return img
