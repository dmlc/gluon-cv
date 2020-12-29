from __future__ import division
from __future__ import print_function

import argparse
import logging
logging.basicConfig(level=logging.INFO)
import time
import sys
import numpy as np
import mxnet as mx
from tqdm import tqdm
from mxnet import nd
from mxnet import gluon
import gluoncv as gcv
gcv.utils.check_version('0.6.0')
from gluoncv import data as gdata
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.ssd import SSDDefaultValTransform
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric
from mxnet.contrib.quantization import *

def parse_args():
    parser = argparse.ArgumentParser(description='Eval SSD networks.')
    parser.add_argument('--network', type=str, default='vgg16_atrous',
                        help="Base network name")
    parser.add_argument('--deploy', action='store_true',
                        help='whether load static model for deployment')
    parser.add_argument('--model-prefix', type=str, required=False,
                        help='load static model as hybridblock.')
    parser.add_argument('--quantized', action='store_true',
                        help='use int8 pretrained model')
    parser.add_argument('--data-shape', type=int, default=300,
                        help="Input data shape")
    parser.add_argument('--batch-size', type=int, default=64,
                        help='eval mini-batch size')
    parser.add_argument('--benchmark', action='store_true',
                        help="run dummy-data based benchmarking")
    parser.add_argument('--num-iterations', type=int, default=100,
                        help="number of benchmarking iterations.")
    parser.add_argument('--dataset', type=str, default='voc',
                        help='eval dataset.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=4, help='Number of data workers')
    parser.add_argument('--num-gpus', type=int, default=0,
                        help='number of gpus to use.')
    parser.add_argument('--pretrained', type=str, default='True',
                        help='Load weights from previously saved parameters.')
    parser.add_argument('--save-prefix', type=str, default='',
                        help='Saving parameter prefix')
    parser.add_argument('--calibration', action='store_true',
                        help='quantize model')
    parser.add_argument('--num-calib-batches', type=int, default=5,
                        help='number of batches for calibration')
    parser.add_argument('--quantized-dtype', type=str, default='auto',
                        choices=['auto', 'int8', 'uint8'],
                        help='quantization destination data type for input data')
    parser.add_argument('--calib-mode', type=str, default='naive',
                        help='calibration mode used for generating calibration table for the quantized symbol; supports'
                             ' 1. none: no calibration will be used. The thresholds for quantization will be calculated'
                             ' on the fly. This will result in inference speed slowdown and loss of accuracy'
                             ' in general.'
                             ' 2. naive: simply take min and max values of layer outputs as thresholds for'
                             ' quantization. In general, the inference accuracy worsens with more examples used in'
                             ' calibration. It is recommended to use `entropy` mode as it produces more accurate'
                             ' inference results.'
                             ' 3. entropy: calculate KL divergence of the fp32 output and quantized output for optimal'
                             ' thresholds. This mode is expected to produce the best inference accuracy of all three'
                             ' kinds of quantized models if the calibration dataset is representative enough of the'
                             ' inference dataset.')
    args = parser.parse_args()
    return args

def get_dataset(dataset, data_shape):
    if dataset.lower() == 'voc':
        val_dataset = gdata.VOCDetection(splits=[(2007, 'test')])
        val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    elif dataset.lower() == 'coco':
        val_dataset = gdata.COCODetection(splits='instances_val2017', skip_empty=False)
        val_metric = COCODetectionMetric(
            val_dataset, args.save_prefix + '_eval', cleanup=True,
            data_shape=(data_shape, data_shape))
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    return val_dataset, val_metric

def get_dataloader(val_dataset, data_shape, batch_size, num_workers):
    """Get dataloader."""
    width, height = data_shape, data_shape
    batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_loader = gluon.data.DataLoader(
        val_dataset.transform(SSDDefaultValTransform(width, height)), batchify_fn=batchify_fn,
        batch_size=batch_size, shuffle=False, last_batch='rollover', num_workers=num_workers)
    return val_loader

def benchmarking(net, ctx, num_iteration, datashape=300, batch_size=64):
    input_shape = (batch_size, 3) + (datashape, datashape)
    data = mx.random.uniform(-1.0, 1.0, shape=input_shape, ctx=ctx, dtype='float32')
    dryrun = 5
    for i in range(dryrun + num_iteration):
        if i == dryrun:
            tic = time.time()
        ids, scores, bboxes = net(data)
        ids.asnumpy()
        scores.asnumpy()
        bboxes.asnumpy()
    toc = time.time() - tic
    return toc

def validate(net, val_data, ctx, classes, size, metric):
    """Test on validation dataset."""
    net.collect_params().reset_ctx(ctx)
    metric.reset()
    with tqdm(total=size) as pbar:
        start = time.time()
        for ib, batch in enumerate(val_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
            det_bboxes = []
            det_ids = []
            det_scores = []
            gt_bboxes = []
            gt_ids = []
            gt_difficults = []
            for x, y in zip(data, label):
                ids, scores, bboxes = net(x)
                det_ids.append(ids)
                det_scores.append(scores)
                # clip to image size
                det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
                # split ground truths
                gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
                gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
                gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

            metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
            pbar.update(batch[0].shape[0])
        end = time.time()
        speed = size / (end - start)
        print('Throughput is %f img/sec.'% speed)
    return metric.get()

if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig()
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)
    logging.info(args)

    # eval contexts
    num_gpus = args.num_gpus
    if num_gpus > 0:
        args.batch_size *= num_gpus
    ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]

    # network
    net_name = '_'.join(('ssd', str(args.data_shape), args.network, args.dataset))
    if args.quantized:
        net_name = '_'.join((net_name, 'int8'))
    args.save_prefix += net_name
    if not args.deploy:
        if args.pretrained.lower() in ['true', '1', 'yes', 't']:
            net = gcv.model_zoo.get_model(net_name, pretrained=True)
        else:
            net = gcv.model_zoo.get_model(net_name, pretrained=False)
            net.load_parameters(args.pretrained.strip())
        net.set_nms(nms_thresh=0.45, nms_topk=400)
        net.hybridize()
    else:
        net_name = 'deploy'
        net = mx.gluon.SymbolBlock.imports('{}-symbol.json'.format(args.model_prefix),
              ['data'], '{}-0000.params'.format(args.model_prefix))
        net.hybridize(static_alloc=True, static_shape=True)

    if args.benchmark:
        print('-----benchmarking on %s -----'%net_name)
        #input_shape = (args.batch_size, 3) + (args.data_shape, args.data_shape)
        #data = mx.random.uniform(-1.0, 1.0, shape=input_shape, ctx=ctx[0], dtype='float32')
        speed = (args.batch_size*args.num_iterations)/benchmarking(net, ctx=ctx[0], num_iteration=args.num_iterations,
                datashape=args.data_shape, batch_size=args.batch_size)
        print('Inference speed on %s, with batchsize %d is %.2f img/sec'%(net_name, args.batch_size, speed))
        sys.exit()

    # eval data
    val_dataset, val_metric = get_dataset(args.dataset, args.data_shape)
    val_data = get_dataloader(
        val_dataset, args.data_shape, args.batch_size, args.num_workers)
    classes = val_dataset.classes  # class names

    # calibration
    if args.calibration and not args.quantized:
        exclude_layers = []
        exclude_layers_match = ['flatten', 'concat']
        if num_gpus > 0:
            raise ValueError('currently only supports CPU with MKL-DNN backend')
        net = quantize_net(
            net, quantized_dtype='auto', exclude_layers=exclude_layers,
            exclude_layers_match=exclude_layers_match, calib_data=val_data,
            calib_mode=args.calib_mode, num_calib_examples=args.batch_size * args.num_calib_batches, ctx=ctx[0],
            logger=logger)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        dst_dir = os.path.join(dir_path, 'model')
        if not os.path.isdir(dst_dir):
            os.mkdir(dst_dir)
        prefix = os.path.join(dst_dir, net_name +
                              '-quantized-' + args.calib_mode)
        logger.info('Saving quantized model at %s' % dst_dir)
        net.export(prefix, epoch=0)
        sys.exit()

    # eval
    names, values = validate(net, val_data, ctx, classes, len(val_dataset), val_metric)
    for k, v in zip(names, values):
        print(k, v)
