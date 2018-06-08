from __future__ import division

import argparse
import logging
logging.basicConfig(level=logging.INFO)
import time
import numpy as np
import mxnet as mx
from tqdm import tqdm
from mxnet import nd
from mxnet import gluon
import gluoncv as gcv
from gluoncv import data as gdata
from gluoncv.data import batchify
from gluoncv.data.transforms.presets.rcnn import FasterRCNNDefaultValTransform
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric

def parse_args():
    parser = argparse.ArgumentParser(description='Validate Faster-RCNN networks.')
    parser.add_argument('--network', type=str, default='resnet50_v1b',
                        help="Base feature extraction network name")
    parser.add_argument('--dataset', type=str, default='voc',
                        help='Training dataset.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=4, help='Number of data workers')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--pretrained', type=str, default='True',
                        help='Load weights from previously saved parameters.')
    args = parser.parse_args()
    return args

def get_dataset(dataset, args):
    if dataset.lower() == 'voc':
        val_dataset = gdata.VOCDetection(
            splits=[(2007, 'test')])
        val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    elif dataset.lower() == 'coco':
        val_dataset = gdata.COCODetection(splits='instances_val2017')
        val_metric = COCODetectionMetric(val_dataset, args.save_prefix + '_eval', cleanup=True)
        # coco validation is slow, consider increase the validation interval
        if args.val_interval == 1:
            args.val_interval = 10
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    return val_dataset, val_metric

def get_dataloader(net, val_dataset, batch_size, num_workers):
    """Get dataloader."""
    short, max_size = 600, 1000
    val_bfn = batchify.Tuple(*[batchify.Append() for _ in range(2)])
    val_loader = mx.gluon.data.DataLoader(
        val_dataset.transform(FasterRCNNDefaultValTransform(short, max_size)),
        batch_size, False, batchify_fn=val_bfn, last_batch='keep', num_workers=num_workers)
    return val_loader

def split_and_load(batch, ctx_list):
    """Split data to 1 batch each device."""
    num_ctx = len(ctx_list)
    new_batch = []
    for i, data in enumerate(batch):
        new_data = [x.as_in_context(ctx) for x, ctx in zip(data, ctx_list)]
        new_batch.append(new_data)
    return new_batch

def validate(net, val_data, ctx, eval_metric, size):
    """Test on validation dataset."""
    net.collect_params().reset_ctx(ctx)
    metric = VOC07MApMetric(iou_thresh=0.5, class_names=net.classes)
    net.set_nms(nms_thresh=0.3, nms_topk=400)
    # net.hybridize()
    with tqdm(total=size) as pbar:
        for ib, batch in enumerate(val_data):
            batch = split_and_load(batch, ctx_list=ctx)
            det_bboxes = []
            det_ids = []
            det_scores = []
            gt_bboxes = []
            gt_ids = []
            gt_difficults = []
            for x, y in zip(*batch):
                # get prediction results
                ids, scores, bboxes = net(x)
                det_ids.append(ids.expand_dims(0))
                det_scores.append(scores.expand_dims(0))
                # clip to image size
                det_bboxes.append(mx.nd.Custom(bboxes, x, op_type='bbox_clip_to_image').expand_dims(0))
                # split ground truths
                gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
                gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
                gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)
            # update metric
            for det_bbox, det_id, det_score, gt_bbox, gt_id, gt_diff in zip(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults):
                eval_metric.update(det_bbox, det_id, det_score, gt_bbox, gt_id, gt_diff)
            pbar.update(len(ctx))
    return metric.get()

if __name__ == '__main__':
    args = parse_args()

    # training contexts
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]
    args.batch_size = len(ctx)  # 1 batch per device

    # network
    net_name = '_'.join(('faster_rcnn', args.network, args.dataset))
    if args.pretrained.lower() in ['true', '1', 'yes', 't']:
        net = gcv.model_zoo.get_model(net_name, pretrained=True)
    else:
        net = gcv.model_zoo.get_model(net_name, pretrained=False)
        net.load_params(args.pretrained.strip())

    # training data
    val_dataset, eval_metric = get_dataset(args.dataset, args)
    val_data = get_dataloader(
        net, val_dataset, args.batch_size, args.num_workers)

    # validation
    names, values = validate(net, val_data, ctx, eval_metric, len(val_dataset))
    for k, v in zip(names, values):
        print(k, v)
