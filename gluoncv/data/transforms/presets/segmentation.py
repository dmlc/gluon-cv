"""Transforms for Segmentation models."""
from __future__ import absolute_import

from mxnet.gluon.data.vision import transforms

def test_transform(img, ctx):
    transform_fn = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    ])
    img = transform_fn(img)
    img = img.expand_dims(0).as_in_context(ctx)
    return img
