from __future__ import division

import argparse
import logging
logging.basicConfig(level=logging.INFO)
import time
import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from gluonvision import data as gdata
from gluonvision.model_zoo import get_model
from gluonvision.data.transforms.presets.ssd import SSDDefaultValTransform
from gluonvision.utils.metrics.voc_detection import VOC07MApMetric

def parse_args():
    parser = argparse.ArgumentParser(description='Train SSD networks.')
    parser.add_argument('--network', type=str, default='vgg16_atrous',
                        help="Base network name")
    parser.add_argument('--data-shape', type=int, default=300,
                        help="Input data shape")
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Training mini-batch size')
    parser.add_argument('--dataset', type=str, default='voc',
                        help='Training dataset.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=4, help='Number of data workers')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--pretrained', type=str, required=True,
                        help='Load weights from previously saved parameters.')
    args = parser.parse_args()
    return args

def get_dataset(dataset):
    if dataset.lower() == 'voc':
        val_dataset = gdata.VOCDetection(
            splits=[(2007, 'test')])
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    return val_dataset

def get_dataloader(val_dataset, data_shape, batch_size, num_workers):
    """Get dataloader."""
    width, height = data_shape, data_shape
    val_loader = gdata.DetectionDataLoader(
        val_dataset.transform(SSDDefaultValTransform(width, height)),
        batch_size, False, last_batch='keep', num_workers=num_workers)
    return val_loader

def validate(net, val_data, ctx, classes):
    """Test on validation dataset."""
    net.collect_params().reset_ctx(ctx)
    metric = VOC07MApMetric(iou_thresh=0.5, class_names=classes)
    net.set_nms(nms_thresh=0.45, nms_topk=-1, force_nms=False)
    net.hybridize()
    for ib, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        for x, y in zip(data, label):
            ids, scores, bboxes = net(x)
            bboxes = bboxes.clip(0, batch[0].shape[2])
            gt_ids = y.slice_axis(axis=-1, begin=4, end=5)
            gt_bboxes = y.slice_axis(axis=-1, begin=0, end=4)
            gt_difficults = y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None
            metric.update(bboxes, ids, scores, gt_bboxes, gt_ids, gt_difficults)
        logging.info("[Batch %d] [Finished %d]", ib, (ib + 1) * batch[0].shape[0])
    return metric.get()

if __name__ == '__main__':
    args = parse_args()

    # training contexts
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]

    # training data
    val_dataset = get_dataset(args.dataset)
    val_data = get_dataloader(
        val_dataset, args.data_shape, args.batch_size, args.num_workers)
    classes = val_dataset.classes  # class names

    # network
    net_name = '_'.join(('ssd', str(args.data_shape), args.network))
    net = get_model(net_name, classes=len(classes), pretrained=0)  # load pretrained base network
    net.load_params(args.pretrained.strip())

    # training
    names, values = validate(net, val_data, ctx, classes)
    for k, v in zip(names, values):
        print(k, v)
