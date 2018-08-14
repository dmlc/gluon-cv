from __future__ import division
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
from gluoncv.data.transforms.presets.rcnn import MaskRCNNDefaultValTransform
from gluoncv.utils.metrics.coco_instance import COCOInstanceMetric

def parse_args():
    parser = argparse.ArgumentParser(description='Validate Mask RCNN networks.')
    parser.add_argument('--network', type=str, default='resnet50_v1b',
                        help="Base feature extraction network name")
    parser.add_argument('--dataset', type=str, default='coco',
                        help='Training dataset.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=4, help='Number of data workers')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--pretrained', type=str, default='True',
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
    if dataset.lower() == 'coco':
        val_dataset = gdata.COCOInstance(splits='instances_val2017', skip_empty=False)
        val_metric = COCOInstanceMetric(val_dataset, args.save_prefix + '_eval',
                                        cleanup=not args.save_json)
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    return val_dataset, val_metric

def get_dataloader(net, val_dataset, batch_size, num_workers):
    """Get dataloader."""
    val_bfn = batchify.Tuple(*[batchify.Append() for _ in range(2)])
    val_loader = mx.gluon.data.DataLoader(
        val_dataset.transform(MaskRCNNDefaultValTransform(net.short, net.max_size)),
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
    clipper = gcv.nn.bbox.BBoxClipToImage()
    eval_metric.reset()
    net.hybridize(static_alloc=True)
    with tqdm(total=size) as pbar:
        for ib, batch in enumerate(val_data):
            batch = split_and_load(batch, ctx_list=ctx)
            det_bboxes = []
            det_ids = []
            det_scores = []
            det_masks = []
            det_infos = []
            for x, im_info in zip(*batch):
                # get prediction results
                ids, scores, bboxes, masks = net(x)
                det_bboxes.append(clipper(bboxes, x))
                det_ids.append(ids)
                det_scores.append(scores)
                det_masks.append(masks)
                det_infos.append(im_info)
            # update metric
            for det_bbox, det_id, det_score, det_mask, det_info in zip(det_bboxes, det_ids, det_scores, det_masks, det_infos):
                for i in range(det_info.shape[0]):
                    # numpy everything
                    det_bbox = det_bbox[i].asnumpy()
                    det_id = det_id[i].asnumpy()
                    det_score = det_score[i].asnumpy()
                    det_mask = det_mask[i].asnumpy()
                    det_info = det_info[i].asnumpy()
                    # filter by conf threshold
                    im_height, im_width, im_scale = det_info
                    valid = np.where(((det_id >= 0) & (det_score >= 0.001)))[0]
                    det_id = det_id[valid]
                    det_score = det_score[valid]
                    det_bbox = det_bbox[valid] / im_scale
                    det_mask = det_mask[valid]
                    # fill full mask
                    im_height, im_width = int(round(im_height / im_scale)), int(round(im_width / im_scale))
                    full_masks = []
                    for bbox, mask in zip(det_bbox, det_mask):
                        full_masks.append(gcv.data.transforms.mask.fill(mask, bbox, (im_width, im_height)))
                    full_masks = np.array(full_masks)
                    eval_metric.update(det_bbox, det_id, det_score, full_masks)
            pbar.update(len(ctx))
    return eval_metric.get()

if __name__ == '__main__':
    args = parse_args()

    # training contexts
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]
    args.batch_size = len(ctx)  # 1 batch per device

    # network
    net_name = '_'.join(('mask_rcnn', args.network, args.dataset))
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

    # validation
    if not args.eval_all:
        names, values = validate(net, val_data, ctx, eval_metric, len(val_dataset))
        for k, v in zip(names, values):
            print(k, v)
    else:
        saved_models = glob.glob(args.save_prefix + '*.params')
        for epoch, saved_model in enumerate(sorted(saved_models)):
            print('[Epoch {}] Validating from {}'.format(epoch, saved_model))
            net.load_parameters(saved_model)
            net.collect_params().reset_ctx(ctx)
            map_name, mean_ap = validate(net, val_data, ctx, eval_metric, len(val_dataset))
            val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
            print('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
            current_map = float(mean_ap[-1])
            with open(args.save_prefix+'_best_map.log', 'a') as f:
                f.write('\n{:04d}:\t{:.4f}'.format(epoch, current_map))
