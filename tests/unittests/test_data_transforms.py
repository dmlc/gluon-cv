from __future__ import print_function
from __future__ import division

import os.path as osp
import mxnet as mx
import numpy as np
from mxnet import autograd, gluon
import gluoncv as gcv
from gluoncv.data import transforms
from gluoncv.data import batchify
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets import ssd
from gluoncv.data.transforms.presets import rcnn
from gluoncv.data.transforms.presets import yolo

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
    np.testing.assert_((np.array(out.shape[:2]) - np.array(image.shape[:2]) + 1).all())

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

def test_transforms_presets_ssd():
    im_fname = gcv.utils.download('https://github.com/dmlc/web-data/blob/master/' +
                                  'gluoncv/detection/biking.jpg?raw=true', path='biking.jpg')
    x, orig_img = ssd.load_test(im_fname, short=512)
    if not osp.isdir(osp.expanduser('~/.mxnet/datasets/voc')):
        return
    train_dataset = gcv.data.VOCDetection(splits=((2007, 'trainval'), (2012, 'trainval')))
    val_dataset = gcv.data.VOCDetection(splits=[(2007, 'test')])
    width, height = (512, 512)
    net = gcv.model_zoo.get_model('ssd_512_resnet50_v1_voc', pretrained=False, pretrained_base=False)
    net.initialize()
    num_workers = 0
    batch_size = 4
    with autograd.train_mode():
        _, _, anchors = net(mx.nd.zeros((1, 3, height, width)))
    batchify_fn = Tuple(Stack(), Stack(), Stack())  # stack image, cls_targets, box_targets
    train_loader = gluon.data.DataLoader(
        train_dataset.transform(ssd.SSDDefaultTrainTransform(width, height, anchors)),
        batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)
    val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_loader = gluon.data.DataLoader(
        val_dataset.transform(ssd.SSDDefaultValTransform(width, height)),
        batch_size, False, batchify_fn=val_batchify_fn, last_batch='keep', num_workers=num_workers)
    train_loader2 = gluon.data.DataLoader(
        train_dataset.transform(ssd.SSDDefaultTrainTransform(width, height)),
        batch_size, True, batchify_fn=val_batchify_fn, last_batch='rollover', num_workers=num_workers)

    for loader in [train_loader, val_loader, train_loader2]:
        for i, batch in enumerate(loader):
            if i > 1:
                break
            pass

def test_transforms_presets_rcnn():
    im_fname = gcv.utils.download('https://github.com/dmlc/web-data/blob/master/' +
                                  'gluoncv/detection/biking.jpg?raw=true', path='biking.jpg')
    x, orig_img = rcnn.load_test(im_fname, short=600, max_size=1000)
    if not osp.isdir(osp.expanduser('~/.mxnet/datasets/voc')):
        return
    train_dataset = gcv.data.VOCDetection(splits=((2007, 'trainval'), (2012, 'trainval')))
    val_dataset = gcv.data.VOCDetection(splits=[(2007, 'test')])
    width, height = (512, 512)
    net = gcv.model_zoo.get_model('faster_rcnn_resnet50_v1b_voc', pretrained=False, pretrained_base=False)
    net.initialize()
    num_workers = 0
    short, max_size = 600, 1000
    batch_size = 4
    train_bfn = batchify.Tuple(*[batchify.Append() for _ in range(5)])
    train_loader = mx.gluon.data.DataLoader(
        train_dataset.transform(rcnn.FasterRCNNDefaultTrainTransform(short, max_size, net)),
        batch_size, True, batchify_fn=train_bfn, last_batch='rollover', num_workers=num_workers)
    val_bfn = batchify.Tuple(*[batchify.Append() for _ in range(3)])
    val_loader = mx.gluon.data.DataLoader(
        val_dataset.transform(rcnn.FasterRCNNDefaultValTransform(short, max_size)),
        batch_size, False, batchify_fn=val_bfn, last_batch='keep', num_workers=num_workers)
    train_loader2 = gluon.data.DataLoader(
        train_dataset.transform(rcnn.FasterRCNNDefaultTrainTransform(short, max_size)),
        batch_size, True, batchify_fn=batchify.Tuple(*[batchify.Append() for _ in range(2)]),
        last_batch='rollover', num_workers=num_workers)

    for loader in [train_loader, val_loader, train_loader2]:
        for i, batch in enumerate(loader):
            if i > 1:
                break
            pass

def test_transforms_presets_mask_rcnn():
    # use valid only, loading training split is very slow
    train_dataset = gcv.data.COCOInstance(splits=('instances_val2017',), skip_empty=True)
    val_dataset = gcv.data.COCOInstance(splits=('instances_val2017',))
    net = gcv.model_zoo.get_model('mask_rcnn_resnet50_v1b_coco', pretrained=False, pretrained_base=False)
    net.initialize()
    num_workers = 0
    short, max_size = 800, 1333
    batch_size = 8
    train_bfn = batchify.Tuple(*[batchify.Append() for _ in range(6)])
    train_loader = mx.gluon.data.DataLoader(
        train_dataset.transform(rcnn.MaskRCNNDefaultTrainTransform(short, max_size, net)),
        batch_size, True, batchify_fn=train_bfn, last_batch='rollover', num_workers=num_workers)
    val_bfn = batchify.Tuple(*[batchify.Append() for _ in range(2)])
    val_loader = mx.gluon.data.DataLoader(
        val_dataset.transform(rcnn.MaskRCNNDefaultValTransform(short, max_size)),
        batch_size, False, batchify_fn=val_bfn, last_batch='keep', num_workers=num_workers)

    for loader in [train_loader, val_loader]:
        for i, batch in enumerate(loader):
            if i > 1:
                break
            pass

def test_transforms_presets_yolo():
    im_fname = gcv.utils.download('https://github.com/dmlc/web-data/blob/master/' +
                                  'gluoncv/detection/biking.jpg?raw=true', path='biking.jpg')
    x, orig_img = yolo.load_test(im_fname, short=512)
    if not osp.isdir(osp.expanduser('~/.mxnet/datasets/voc')):
        return
    train_dataset = gcv.data.VOCDetection(splits=((2007, 'trainval'), (2012, 'trainval')))
    val_dataset = gcv.data.VOCDetection(splits=[(2007, 'test')])
    width, height = (512, 512)
    net = gcv.model_zoo.get_model('yolo3_darknet53_voc', pretrained=False, pretrained_base=False)
    net.initialize()
    num_workers = 0
    batch_size = 4
    batchify_fn = Tuple(*([Stack() for _ in range(6)] + [Pad(axis=0, pad_val=-1) for _ in range(1)]))
    train_loader = gluon.data.DataLoader(
        train_dataset.transform(yolo.YOLO3DefaultTrainTransform(width, height, net)),
        batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)
    val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_loader = gluon.data.DataLoader(
        val_dataset.transform(yolo.YOLO3DefaultValTransform(width, height)),
        batch_size, False, batchify_fn=val_batchify_fn, last_batch='keep', num_workers=num_workers)
    train_loader2 = gluon.data.DataLoader(
        train_dataset.transform(yolo.YOLO3DefaultTrainTransform(width, height)),
        batch_size, True, batchify_fn=val_batchify_fn, last_batch='rollover', num_workers=num_workers)

    for loader in [train_loader, val_loader, train_loader2]:
        for i, batch in enumerate(loader):
            if i > 1:
                break
            pass

if __name__ == '__main__':
    import nose
    nose.runmodule()
