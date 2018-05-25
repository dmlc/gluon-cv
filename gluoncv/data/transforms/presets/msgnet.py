"""Preset Transforms for Testing Faster RCNN Model"""
from PIL import Image
import numpy as np
import mxnet as mx
from mxnet import cpu
import mxnet.ndarray as F

__all__ = ['load_image', 'subtract_imagenet_mean_batch', 'save_rgbimage',
           'save_bgrimage', 'preprocess_batch']

def load_image(filename, ctx=cpu(), short_size=800):
    """Util Function of Loading Images for Faster RCNN Model"""
    img = Image.open(filename).convert('RGB')
    w, h = img.size[0], img.size[1]
    if h < w:
        nh = short_size
        nw = int(1.0 * short_size / h * w)
    else:
        nw = short_size
        nh = int(1.0 * short_size / w * h)

    img = img.resize((nw, nh), Image.ANTIALIAS)
    img = np.array(img).transpose(2, 0, 1).astype(float)
    img = mx.nd.array(img, ctx=ctx).expand_dims(0)
    return img


def preprocess_batch(batch):
    batch = F.swapaxes(batch, 0, 1)
    (r, g, b) = F.split(batch, num_outputs=3, axis=0)
    batch = F.concat(b, g, r, dim=0)
    batch = F.swapaxes(batch, 0, 1)
    return batch


def subtract_imagenet_mean_batch(batch):
    """Subtract ImageNet mean pixel-wise from a RBG image."""
    batch = F.swapaxes(batch, 0, 1)
    (r, g, b) = F.split(batch, num_outputs=3, axis=0)
    r = r - 123.680
    g = g - 116.779
    b = b - 103.939
    batch = F.concat(r, g, b, dim=0)
    batch = F.swapaxes(batch, 0, 1)
    return batch


def add_imagenet_mean_batch(batch):
    """Add ImageNet mean pixel-wise from a BGR image."""
    batch = F.swapaxes(batch, 0, 1)
    (b, g, r) = F.split(batch, num_outputs=3, axis=0)
    r = r + 123.680
    g = g + 116.779
    b = b + 103.939
    batch = F.concat(b, g, r, dim=0)
    batch = F.swapaxes(batch, 0, 1)
    return batch


def save_rgbimage(img, filename):
    """Save rgb image"""
    img = F.clip(img, 0, 255).asnumpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img.save(filename)


def save_bgrimage(tensor, filename):
    """Save bgr image"""
    (b, g, r) = F.split(tensor, num_outputs=3, axis=0)
    tensor = F.concat(r, g, b, dim=0)
    save_rgbimage(tensor, filename)
