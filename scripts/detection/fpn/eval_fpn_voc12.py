"""Validate FPN end to end."""
from __future__ import division
from __future__ import print_function

import os
# disable autotune
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
import argparse
import glob
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
from gluoncv.data.transforms.presets.rcnn import FPNDefaultValTransform
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric

def parse_args():
    parser = argparse.ArgumentParser(description='Validate FPN networks.')
    parser.add_argument('--network', type=str, default='resnet50_v1d',
                        help="Base feature extraction network name")
    parser.add_argument('--dataset', type=str, default='voc',
                        help='Training dataset.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=8, help='Number of data workers')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--pretrained', type=str, default='./20181026_fpn_resnet50_v1d_voc_best.params',
                        help='Load weights from previously saved parameters.')
    parser.add_argument('--save-prefix', type=str, default='',
                        help='Saving parameter prefix')
    parser.add_argument('--save-json', action='store_true',
                        help='Save coco output json')
    parser.add_argument('--eval-all', action='store_true',
                        help='Eval all models begins with save prefix. Use with pretrained.')
    args = parser.parse_args()
    return args

def get_dataset(dataset, args):
    if dataset.lower() == 'voc':
        val_dataset = gdata.VOC12Detection(
            splits=[(2012, 'test')])
        val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    elif dataset.lower() == 'coco':
        val_dataset = gdata.COCODetection(splits='instances_val2017', skip_empty=False)
        val_metric = COCODetectionMetric(val_dataset, args.save_prefix + '_eval',
                                         cleanup=not args.save_json)
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    return val_dataset, val_metric

def get_dataloader(net, val_dataset, batch_size, num_workers):
    """Get dataloader."""
    val_bfn = batchify.Tuple(*[batchify.Append() for _ in range(3)])
    val_loader = mx.gluon.data.DataLoader(
        val_dataset.transform(FPNDefaultValTransform(net.short, net.max_size)),
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

def validate(net, val_data, val_items, val_shapes, ctx, size, classes):
    """Test on validation dataset."""
    clipper = gcv.nn.bbox.BBoxClipToImage()
    net.hybridize(static_alloc=True)
    print("---Detect Total {:d} Image Start.---".format(len(val_items))) 

    result_dict = {}
    for ib, (batch, item) in enumerate(zip(val_data, val_items)):
        batch = split_and_load(batch, ctx_list=ctx)
        for x, y, im_scale in zip(*batch):
            ids, scores, bboxes = net(x)
            bboxes = clipper(bboxes, x)
            im_scale = im_scale.reshape((-1)).asscalar()
            bboxes *= im_scale
            inds = nd.argsort(nd.squeeze(ids, axis=(0, 2)), is_ascend=False)
            ids = nd.squeeze(ids, axis=(0, 2)).asnumpy().astype(np.int8).tolist()
            valid_ids = [id for id in ids if id is not -1]
            valid_len = len(valid_ids)
            inds = nd.slice_axis(inds, begin=0, end=valid_len, axis=0)
            scores = nd.take(scores, inds, axis=1)
            bboxes = nd.take(bboxes, inds, axis=1)
            scores = scores.asnumpy()
            bboxes = bboxes.asnumpy()
            for i, id in enumerate(valid_ids):
                score = scores[:, i, 0][0]
                xmin, ymin, xmax, ymax = bboxes[:, i, 0][0], bboxes[:, i, 1][0], bboxes[:, i, 2][0], bboxes[:, i, 3][0] 
                result_dict[id] = result_dict.get(id, []) + [[item, score, xmin, ymin, xmax, ymax]]
            print("Detect Image {:s} Done.".format(item))
    print("---Detect Total {:d} Image Done.---".format(len(val_items)))
    return result_dict

if __name__ == '__main__':
    args = parse_args()

    # training contexts
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]
    args.batch_size = len(ctx)  # 1 batch per device

    # network
    net_name = '_'.join(('fpn', args.network, args.dataset))

    net_name = 'fpn_resnet50_v1d_voc12'

    args.save_prefix += net_name
    if args.pretrained.lower() in ['true', '1', 'yes', 't']:
        net = gcv.model_zoo.get_model(net_name, pretrained=True)
    else:
        net = gcv.model_zoo.get_model(net_name, pretrained=False)
        net.load_parameters(args.pretrained.strip())
    net.collect_params().reset_ctx(ctx)

    # training data
    val_dataset, eval_metric = get_dataset(args.dataset, args)
    val_data = get_dataloader(
        net, val_dataset, args.batch_size, args.num_workers)

    classes = val_dataset.classes  # class names
    val_items = [item[1] for item in val_dataset.items] # image names for each image
    val_shapes = val_dataset.im_shapes # image shapes for each image 

    # validation
    result_dict = validate(net, val_data, val_items, val_shapes, ctx, classes, len(val_dataset))
    
    # Write Result 
    index_map = {16:'sheep', 12:'horse', 1:'bicycle', 0:'aeroplane', 9:'cow', 17:'sofa', 5:'bus', 11:'dog', 
    7:'cat', 14:'person', 18:'train', 10:'diningtable', 4:'bottle', 6:'car', 15:'pottedplant', 
    19:'tvmonitor', 8:'chair', 2:'bird', 3:'boat', 13:'motorbike'}
    
    for k in result_dict:
        with open("comp4_det_{:s}_{:s}.txt".format(args.competition, index_map[k]), "wt") as f:
            for res in result_dict[k]:
                f.write('{:s} {:f} {:f} {:f} {:f} {:f}\n'.format(res[0], res[1], res[2], res[3], res[4], res[5]))
            f.close()
            print("Write Result File {:s} done.".format("comp4_det_test_{:s}.txt".format(index_map[k])))