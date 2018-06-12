"""Preset Transforms for Testing Faster RCNN Model"""
from PIL import Image
import numpy as np
import mxnet as mx
from mxnet import cpu
import mxnet.ndarray as F

__all__ = ['load_image', 'subtract_imagenet_mean_batch']

def load_image(filename, ctx=cpu(), short_size=800):
    """Util Function of Loading Images for Faster RCNN Model"""
    img = Image.open(filename).convert('RGB')
    w, h = img.size[0], img.size[1]
    if h < w:
        nh = short_size
        scale = 1.0 * short_size / h
        nw = int(1.0 * short_size / h * w)
    else:
        nw = short_size
        scale = 1.0 * short_size / w
        nh = int(1.0 * short_size / w * h)

    img = img.resize((nw, nh), Image.ANTIALIAS)
    img = np.array(img).transpose(2, 0, 1).astype(float)
    img = F.expand_dims(mx.nd.array(img, ctx=ctx), 0)
    img = subtract_imagenet_mean_batch(img)
    return img, scale, w, h


def subtract_imagenet_mean_batch(batch):
    """Subtract ImageNet mean pixel-wise from a BGR image."""
    batch = F.swapaxes(batch, 0, 1)
    (r, g, b) = F.split(batch, num_outputs=3, axis=0)
    r = r - 122.7717
    g = g - 115.9465
    b = b - 102.9801
    batch = F.concat(r, g, b, dim=0)
    batch = F.swapaxes(batch, 0, 1)
    return batch
