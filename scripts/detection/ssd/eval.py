from __future__ import division

import argparse
import logging
logging.basicConfig(level=logging.INFO)
import time
import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
from gluonvision import data as gdata
from gluonvision import utils as gutils
from gluonvision.model_zoo import get_model
from gluonvision.data.transforms.presets.ssd import SSDDefaultTrainTransform
from gluonvision.data.transforms.presets.ssd import SSDDefaultValTransform
from gluonvision.utils.metrics.voc_detection import VOC07MApMetric
from gluonvision.utils.metrics.accuracy import Accuracy

def parse_args():
    parser = argparse.ArgumentParser(description='Train SSD networks.')
    parser.add_argument('--network', type=str, default='resnet50_v1',
                        help="Base network name")
    parser.add_argument('--data-shape', type=int, default=512,
                        help="Input data shape")
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training mini-batch size')
    parser.add_argument('--dataset', type=str, default='voc',
                        help='Training dataset.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=0, help='Number of data workers')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--epochs', type=int, default=240,
                        help='Training epochs.')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from previously saved parameters.')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Starting epoch for resuming, default is 0 for new training.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate, default is 0.001')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-epoch', type=str, default='160,200',
                        help='epoches at which learning rate decays. default is 160,200.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum, default is 0.9')
    parser.add_argument('--wd', type=float, default=0.0005,
                        help='Weight decay, default is 1e-4')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Logging mini-batch interval.')
    parser.add_argument('--seed', type=int, default=233,
                        help='Random seed to be fixed.')
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

def validate(net, val_data, ctx, classes, val_dataset):
    """Test on validation dataset."""
    net.collect_params().reset_ctx(ctx)
    metric = VOC07MApMetric(iou_thresh=0.5, class_names=classes)
    net.set_nms(nms_thresh=0.45, nms_topk=-1, force_nms=False)
    # net.hybridize()
    all_boxes = [[[] for _ in range(4952)]
                 for _ in range(20+1)]
    image_count = 0
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
            ids, scores, bboxes = [xxx.asnumpy() for xxx in [ids, scores, bboxes]]
            for idd, score, bbox in zip(ids, scores, bboxes):
                img = val_dataset[image_count]
                H, W, _ = img[0].shape
                # each image
                for j in range(0, 20):
                    mask = np.where(idd == j)[0]
                    if mask.size < 1:
                        continue
                    ss = score[mask]
                    bb = bbox[mask, :]
                    bb[:, (0, 2)] /= 300 / W
                    bb[:, (1, 3)] /= 300 / H
                    cls_dets = np.hstack((bb, ss)).astype(np.float32, copy=False)
                    # if j == 0:
                    # print(j, image_count, cls_dets)
                    # raise
                    all_boxes[j+1][image_count] = cls_dets


                image_count += 1

        logging.info("[Batch %d] [Finished %d]", ib, (ib + 1) * batch[0].shape[0])
    import pickle
    with open('/home/joshua/Dev/Cache/gluon_det.pkl', 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
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
    net.load_params('/home/joshua/Dev/Cache/ssd_vgg16_converted.params')

    # training
    names, values = validate(net, val_data, ctx, classes, val_dataset)
    for k, v in zip(names, values):
        print(k, v)
