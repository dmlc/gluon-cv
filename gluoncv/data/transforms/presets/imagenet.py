"""Transforms for ImageNet series."""
from __future__ import absolute_import
import mxnet as mx

from mxnet.gluon.data.vision import transforms

__all__ = ['transform_eval']

def transform_eval(imgs, resize_short=256, crop_size=224,
                   mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """A util function to transform all images to tensors as network input by applying
    normalizations. This function support 1 NDArray or iterable of NDArrays.

    Parameters
    ----------
    imgs : NDArray or iterable of NDArray
        Image(s) to be transformed.
    resize_short : int, default=256
        Resize image short side to this value and keep aspect ratio.
    crop_size : int, default=224
        After resize, crop the center square of size `crop_size`
    mean : iterable of float
        Mean pixel values.
    std : iterable of float
        Standard deviations of pixel values.

    Returns
    -------
    mxnet.NDArray or list of such tuple
        A (1, 3, H, W) mxnet NDArray as input to network
        If multiple image names are supplied, return a list.
    """
    if isinstance(imgs, mx.nd.NDArray):
        imgs = [imgs]
    for im in imgs:
        assert isinstance(im, mx.nd.NDArray), "Expect NDArray, got {}".format(type(im))

    transform_fn = transforms.Compose([
        transforms.Resize(resize_short, keep_ratio=True),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    res = [transform_fn(img).expand_dims(0) for img in imgs]

    if len(res) == 1:
        return res[0]
    return res
