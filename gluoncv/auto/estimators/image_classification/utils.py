"""Utils for auto classification estimator"""
# pylint: disable=bad-whitespace,missing-function-docstring
import os
import math

import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet.gluon.data.vision import transforms
from ....data import imagenet

def rec_batch_fn(batch, ctx):
    data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0, even_split=False)
    label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0, even_split=False)
    return data, label

def get_data_rec(rec_train, rec_train_idx, rec_val, rec_val_idx, batch_size, num_workers, input_size, crop_ratio):
    rec_train = os.path.expanduser(rec_train)
    rec_train_idx = os.path.expanduser(rec_train_idx)
    rec_val = os.path.expanduser(rec_val)
    rec_val_idx = os.path.expanduser(rec_val_idx)
    jitter_param = 0.4
    lighting_param = 0.1
    input_size = input_size
    crop_ratio = crop_ratio if crop_ratio > 0 else 0.875
    resize = int(math.ceil(input_size / crop_ratio))
    mean_rgb = [123.68, 116.779, 103.939]
    std_rgb = [58.393, 57.12, 57.375]

    train_data = mx.io.ImageRecordIter(
        path_imgrec         = rec_train,
        path_imgidx         = rec_train_idx,
        preprocess_threads  = num_workers,
        shuffle             = True,
        batch_size          = batch_size,

        data_shape          = (3, input_size, input_size),
        mean_r              = mean_rgb[0],
        mean_g              = mean_rgb[1],
        mean_b              = mean_rgb[2],
        std_r               = std_rgb[0],
        std_g               = std_rgb[1],
        std_b               = std_rgb[2],
        rand_mirror         = True,
        random_resized_crop = True,
        max_aspect_ratio    = 4. / 3.,
        min_aspect_ratio    = 3. / 4.,
        max_random_area     = 1,
        min_random_area     = 0.08,
        brightness          = jitter_param,
        saturation          = jitter_param,
        contrast            = jitter_param,
        pca_noise           = lighting_param,
    )
    val_data = mx.io.ImageRecordIter(
        path_imgrec         = rec_val,
        path_imgidx         = rec_val_idx,
        preprocess_threads  = num_workers,
        shuffle             = False,
        batch_size          = batch_size,

        resize              = resize,
        data_shape          = (3, input_size, input_size),
        mean_r              = mean_rgb[0],
        mean_g              = mean_rgb[1],
        mean_b              = mean_rgb[2],
        std_r               = std_rgb[0],
        std_g               = std_rgb[1],
        std_b               = std_rgb[2],
    )
    return train_data, val_data, rec_batch_fn

def loader_batch_fn(batch, ctx):
    data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
    label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
    return data, label

def get_data_loader(data_dir, batch_size, num_workers, input_size, crop_ratio, train_dataset=None, val_dataset=None):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    jitter_param = 0.4
    lighting_param = 0.1
    crop_ratio = crop_ratio if crop_ratio > 0 else 0.875
    resize = int(math.ceil(input_size / crop_ratio))

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomFlipLeftRight(),
        transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param,
                                     saturation=jitter_param),
        transforms.RandomLighting(lighting_param),
        transforms.ToTensor(),
        normalize
    ])
    transform_test = transforms.Compose([
        transforms.Resize(resize, keep_ratio=True),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize
    ])

    if not train_dataset:
        train_dataset = imagenet.classification.ImageNet(data_dir, train=True)
    if not val_dataset:
        val_dataset = imagenet.classification.ImageNet(data_dir, train=False)

    train_data = gluon.data.DataLoader(
        train_dataset.transform_first(transform_train),
        batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=num_workers)
    val_data = gluon.data.DataLoader(
        val_dataset.transform_first(transform_test),
        batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_data, val_data, loader_batch_fn

def mixup_transform(label, classes, lam=1, eta=0.0):
    if isinstance(label, nd.NDArray):
        label = [label]
    res = []
    for l in label:
        y1 = l.one_hot(classes, on_value=1 - eta + eta/classes, off_value=eta/classes)
        y2 = l[::-1].one_hot(classes, on_value=1 - eta + eta/classes, off_value=eta/classes)
        res.append(lam*y1 + (1-lam)*y2)
    return res

def smooth(label, classes, eta=0.1):
    if isinstance(label, nd.NDArray):
        label = [label]
    smoothed = []
    for l in label:
        res = l.one_hot(classes, on_value=1 - eta + eta/classes, off_value=eta/classes)
        smoothed.append(res)
    return smoothed
