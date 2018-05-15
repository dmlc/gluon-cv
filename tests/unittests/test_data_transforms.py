from __future__ import print_function
from __future__ import division

import mxnet as mx
import numpy as np

import gluoncv as gcv
from gluoncv.data import transforms

def test_bbox_crop():
    bbox = np.array([[10, 20, 200, 500], [150, 200, 400, 300]])
    np.testing.assert_allclose(transforms.bbox.crop(bbox, None), bbox)
    np.testing.assert_allclose(
        transforms.bbox.crop(bbox, (20, 30, 200, 200), allow_outside_center=True),
        np.array([[  0,   0, 180, 200], [130, 170, 200, 200]]))
    np.testing.assert_allclose(
        transforms.bbox.crop(bbox, (20, 30, 200, 300), allow_outside_center=False),
        np.array([[  0,   0, 180, 300]]))

def test_bbox_flip():
    bbox = np.array([[10, 20, 200, 500], [150, 200, 400, 300]])
    size = (500, 1000)
    np.testing.assert_allclose(
        transforms.bbox.flip(bbox, size, False, False),
        bbox)
    np.testing.assert_allclose(
        transforms.bbox.flip(bbox, size, False, True),
        np.array([[ 10, 500, 200, 980], [150, 700, 400, 800]]))
    np.testing.assert_allclose(
        transforms.bbox.flip(bbox, size, True, False),
        np.array([[300,  20, 490, 500], [100, 200, 350, 300]]))
    np.testing.assert_allclose(
        transforms.bbox.flip(bbox, size, True, True),
        np.array([[300, 500, 490, 980], [100, 700, 350, 800]]))

def test_bbox_resize():
    bbox = np.array([[10, 20, 200, 500], [150, 200, 400, 300]], dtype=np.float32)
    in_size = (600, 1000)
    out_size = (200, 300)
    np.testing.assert_allclose(
        transforms.bbox.resize(bbox, in_size, out_size),
        np.array([[  3.333333,   6.,  66.66667 , 150.], [ 50.,  60.000004, 133.33334 ,  90.]]),
        rtol=1e-3)

def test_bbox_translate():
    bbox = np.array([[10, 20, 200, 500], [150, 200, 400, 300]], dtype=np.float32)
    xoff = np.random.randint(-100, 100)
    yoff = np.random.randint(-100, 100)
    expected = bbox.copy()
    expected[:, (0, 2)] += xoff
    expected[:, (1, 3)] += yoff
    np.testing.assert_allclose(transforms.bbox.translate(bbox, xoff, yoff), expected)

def test_image_imresize():
    image = mx.random.normal(shape=(240, 480, 3)).astype(np.uint8)
    out = transforms.image.imresize(image, 300, 300)
    np.testing.assert_allclose(out.shape, (300, 300, 3))

def test_image_resize_long():
    image = mx.random.normal(shape=(240, 480, 3)).astype(np.uint8)
    out = transforms.image.resize_long(image, 300)
    np.testing.assert_allclose(out.shape, (150, 300, 3))

def test_image_random_pca():
    image = mx.random.normal(shape=(240, 120, 3)).astype(np.float32)
    out = transforms.image.random_pca_lighting(image, 0.1)
    np.testing.assert_allclose(out.shape, image.shape)  # no value check

def test_image_random_expand():
    image = mx.random.normal(shape=(240, 120, 3)).astype(np.uint8)
    # no expand when ratio <= 1
    out, _ = transforms.image.random_expand(image, max_ratio=0.1, keep_ratio=True)
    np.testing.assert_allclose(out.asnumpy(), image.asnumpy())
    # check ratio
    out, _ = transforms.image.random_expand(image, 4, keep_ratio=True)
    np.testing.assert_allclose(out.shape[0] / out.shape[1], image.shape[0] / image.shape[1], rtol=1e-2, atol=1e-3)
    # #
    out, _ = transforms.image.random_expand(image, 4, keep_ratio=False)
    np.testing.assert_((np.array(out.shape[:2]) - np.array(image.shape[:2])).all())

def test_image_random_flip():
    image = mx.random.normal(shape=(240, 120, 3)).astype(np.uint8)
    # no flip
    out, f = transforms.image.random_flip(image, 0, 0)
    np.testing.assert_allclose(image.asnumpy(), out.asnumpy())
    assert(f == (False, False))
    # horizontal
    out, f = transforms.image.random_flip(image, 1, 0)
    np.testing.assert_allclose(image.asnumpy()[:, ::-1, :], out.asnumpy())
    assert(f == (True, False))
    # vertical
    out, f = transforms.image.random_flip(image, 0, 1)
    np.testing.assert_allclose(image.asnumpy()[::-1, :, :], out.asnumpy())
    assert(f == (False, True))
    # both
    out, f = transforms.image.random_flip(image, 1, 1)
    np.testing.assert_allclose(image.asnumpy()[::-1, ::-1, :], out.asnumpy())
    assert(f == (True, True))

def test_image_resize_contrain():
    image = mx.random.normal(shape=(240, 120, 3)).astype(np.uint8)
    size = (300, 300)
    out, _ = transforms.image.resize_contain(image, size)
    np.testing.assert_allclose(out.shape, (300, 300, 3))
    size = (100, 100)
    out, _ = transforms.image.resize_contain(image, size)
    np.testing.assert_allclose(out.shape, (100, 100, 3))

def test_image_ten_crop():
    image = mx.random.normal(shape=(240, 120, 3)).astype(np.uint8)
    size = (24, 24)
    crops = transforms.image.ten_crop(image, size)
    assert len(crops) == 10
    im = image.asnumpy()[:24, :24, :]
    np.testing.assert_allclose(im, crops[1].asnumpy())
    np.testing.assert_allclose(im[:, ::-1, :], crops[6].asnumpy())

def test_experimental_bbox_random_crop_with_constraints():
    bbox = np.array([[10, 20, 200, 500], [150, 200, 400, 300]])
    size = (640, 480)
    for _ in range(10):
        min_scale = np.random.uniform(0, 0.9)
        max_scale = np.random.uniform(min_scale, 1)
        max_aspect_ratio = np.random.uniform(1, 3)
        out, crop = transforms.experimental.bbox.random_crop_with_constraints(
            bbox, size, min_scale=min_scale, max_scale=max_scale,
            max_aspect_ratio=max_aspect_ratio, max_trial=20)
    assert out.size >= 4

def test_experimental_image_random_color_distort():
    image = mx.random.normal(shape=(240, 120, 3)).astype(np.float32)
    for _ in range(10):
        brightness_delta = np.random.randint(0, 64)
        contrast_low = np.random.uniform(0, 1)
        contrast_high = np.random.uniform(1, 2)
        saturation_low = np.random.uniform(0, 1)
        saturation_high = np.random.uniform(1, 2)
        hue_delta = np.random.randint(0, 36)
        out = transforms.experimental.image.random_color_distort(
            image, brightness_delta=brightness_delta, contrast_low=contrast_low,
            contrast_high=contrast_high, saturation_low=saturation_low,
            saturation_high=saturation_high, hue_delta=hue_delta)
        np.testing.assert_allclose(out.shape, image.shape)

if __name__ == '__main__':
    import nose
    nose.runmodule()
