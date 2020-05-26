import os
import logging
from tqdm import tqdm
import numpy as np
import argparse
import time
import sys

import mxnet as mx
from mxnet import gluon, ndarray as nd
from mxnet.gluon.data.vision import transforms
from mxnet.contrib.quantization import *

import gluoncv
gluoncv.utils.check_version('0.6.0')
from gluoncv.model_zoo.segbase import *
from gluoncv.model_zoo import get_model
from gluoncv.data import get_segmentation_dataset, ms_batchify_fn
from gluoncv.utils.viz import get_color_pallete
from gluoncv.utils.parallel import *

def parse_args():
    parser = argparse.ArgumentParser(description='Validation on Semantic Segmentation model')
    parser.add_argument('--model-zoo', type=str, default=None,
                        help='evaluating on model zoo model')
    parser.add_argument('--model', type=str, default='fcn',
                        help='model name (default: fcn)')
    parser.add_argument('--backbone', type=str, default='resnet101',
                        help='base network')
    parser.add_argument('--deploy', action='store_true',
                        help='whether load static model for deployment')
    parser.add_argument('--model-prefix', type=str, required=False,
                        help='load static model as hybridblock.')
    parser.add_argument('--image-shape', type=int, default=480,
                        help='image shape')
    parser.add_argument('--base-size', type=int, default=520,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=480,
                        help='crop image size')
    parser.add_argument('--height', type=int, default=None,
                        help='height of original image size')
    parser.add_argument('--width', type=int, default=None,
                        help='width of original image size')
    parser.add_argument('--mode', type=str, default='val',
                        help='val, testval')
    parser.add_argument('--dataset', type=str, default='pascal_voc',
                        help='dataset used for validation [pascal_voc, pascal_aug, coco, ade20k]')
    parser.add_argument('--quantized', action='store_true',
                        help='whether to use int8 pretrained  model')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-iterations', type=int, default=100,
                        help='number of benchmarking iterations.')
    parser.add_argument('--workers', type=int, default=4,
                        help='number of workers for data loading')
    parser.add_argument('--pretrained', action="store_true",
                        help='whether to use pretrained params')
    parser.add_argument('--ngpus', type=int,
                        default=len(mx.test_utils.list_gpus()),
                        help='number of GPUs (default: 4)')
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')
    parser.add_argument('--syncbn', action='store_true', default=False,
                        help='using Synchronized Cross-GPU BatchNorm')
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluation only')
    # dummy benchmark
    parser.add_argument('--benchmark', action='store_true', default=False,
                        help='whether to use dummy data for benchmark')
    # calibration
    parser.add_argument('--calibration', action='store_true',
                        help='quantize model')
    parser.add_argument('--num-calib-batches', type=int, default=5,
                        help='number of batches for calibration')
    parser.add_argument('--quantized-dtype', type=str, default='auto',
                        choices=['auto', 'int8', 'uint8'],
                        help='quantization destination data type for input data')
    parser.add_argument('--calib-mode', type=str, default='entropy',
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

    args.ctx = [mx.cpu(0)]
    args.ctx = [mx.gpu(i) for i in range(args.ngpus)] if args.ngpus > 0 else args.ctx

    args.norm_layer = mx.gluon.contrib.nn.SyncBatchNorm if args.syncbn \
        else mx.gluon.nn.BatchNorm
    args.norm_kwargs = {'num_devices': args.ngpus} if args.syncbn else {}
    return args

def test(model, args, input_transform):
    # DO NOT modify!!! Only support batch_size=ngus
    batch_size = args.ngpus

    # output folder
    outdir = 'outdir'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # get dataset
    if args.eval:
        testset = get_segmentation_dataset(
            args.dataset, split='val', mode='testval', transform=input_transform)
    else:
        testset = get_segmentation_dataset(
            args.dataset, split='test', mode='test', transform=input_transform)

    if 'icnet' or 'fastscnn' in args.model:
        test_data = gluon.data.DataLoader(
            testset, batch_size, shuffle=False, last_batch='rollover',
            num_workers=args.workers)
    else:
        test_data = gluon.data.DataLoader(
            testset, batch_size, shuffle=False, last_batch='rollover',
            batchify_fn=ms_batchify_fn, num_workers=args.workers)
    print(model)

    if 'icnet' or 'fastscnn' in args.model:
        evaluator = DataParallelModel(SegEvalModel(model, use_predict=True), ctx_list=args.ctx)
    else:
        evaluator = MultiEvalModel(model, testset.num_class, ctx_list=args.ctx)

    metric = gluoncv.utils.metrics.SegmentationMetric(testset.num_class)

    if 'icnet' or 'fastscnn' in args.model:
        tbar = tqdm(test_data)
        t_gpu = 0
        num = 0
        for i, (data, dsts) in enumerate(tbar):
            tic = time.time()
            outputs = evaluator(data.astype('float32', copy=False))
            # outputs = evaluator(data.astype('float32', copy=False))
            t_gpu += time.time() - tic
            num += 1

            outputs = [x[0] for x in outputs]
            targets = mx.gluon.utils.split_and_load(dsts, ctx_list=args.ctx, even_split=False)
            metric.update(targets, outputs)

            pixAcc, mIoU = metric.get()
            gpu_time = t_gpu / num
            tbar.set_description('pixAcc: %.4f, mIoU: %.4f, t_gpu: %.2fms' % (pixAcc, mIoU, gpu_time*1000))
    else:
        tbar = tqdm(test_data)
        for i, (data, dsts) in enumerate(tbar):
            if args.eval:
                predicts = [pred[0] for pred in evaluator.parallel_forward(data)]
                targets = [target.as_in_context(predicts[0].context) \
                           for target in dsts]
                metric.update(targets, predicts)
                pixAcc, mIoU = metric.get()
                tbar.set_description('pixAcc: %.4f, mIoU: %.4f' % (pixAcc, mIoU))
            else:
                im_paths = dsts
                predicts = evaluator.parallel_forward(data)
                for predict, impath in zip(predicts, im_paths):
                    predict = mx.nd.squeeze(mx.nd.argmax(predict[0], 1)).asnumpy() + \
                        testset.pred_offset
                    mask = get_color_pallete(predict, args.dataset)
                    outname = os.path.splitext(impath)[0] + '.png'
                    mask.save(os.path.join(outdir, outname))

def test_quantization(model, args, test_data, size, num_class, pred_offset):
    # output folder
    outdir = 'outdir_int8'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    print(model)
    metric = gluoncv.utils.metrics.SegmentationMetric(num_class)

    tbar = tqdm(test_data)
    metric.reset()
    tic = time.time()
    for i, (batch, dsts) in enumerate(tbar):
        if args.eval:
            targets = mx.gluon.utils.split_and_load(dsts, ctx_list=args.ctx, even_split=False)
            data = mx.gluon.utils.split_and_load(batch, ctx_list=args.ctx, batch_axis=0, even_split=False)
            outputs = None
            for x in data:
                output = model(x)
                outputs = output if outputs is None else nd.concat(outputs, output, axis=0)
            metric.update(targets, outputs)
            pixAcc, mIoU = metric.get()
            tbar.set_description('pixAcc: %.4f, mIoU: %.4f' % (pixAcc, mIoU))
        else:
            for data, impath in zip(batch, dsts):
                data = data.as_in_context(args.ctx[0])
                if len(data.shape) < 4:
                    data = nd.expand_dims(data, axis=0)
                predict = model(data)[0]
                predict = mx.nd.squeeze(mx.nd.argmax(predict, 1)).asnumpy() + pred_offset
                mask = get_color_pallete(predict, args.dataset)
                outname = os.path.splitext(impath)[0] + '.png'
                mask.save(os.path.join(outdir, outname))
    speed = size / (time.time() - tic)
    print('Inference speed with batchsize %d is %.2f img/sec' % (args.batch_size, speed))

def benchmarking(model, args):
    bs = args.batch_size
    num_iterations = args.num_iterations
    input_shape = (bs, 3, args.image_shape, args.image_shape)
    size = num_iterations * bs
    data = mx.random.uniform(-1.0, 1.0, shape=input_shape, ctx=args.ctx[0], dtype='float32')
    dry_run = 5

    with tqdm(total=size+dry_run*bs) as pbar:
        for n in range(dry_run + num_iterations):
            if n == dry_run:
                tic = time.time()
            outputs = model(data)
            for output in outputs:
                output.wait_to_read()
            pbar.update(bs)
    speed = size / (time.time() - tic)
    print('With batch size %d , %d batches, throughput is %f imgs/sec' % (bs, num_iterations, speed))

if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig()
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)
    logging.info(args)

    withQuantization = False
    model_prefix = args.model + '_' + args.backbone
    if 'pascal' in args.dataset:
        model_prefix += '_voc'
        withQuantization = True if (args.backbone in ['resnet101']) else withQuantization
    elif args.dataset == 'coco':
        model_prefix += '_coco'
        withQuantization = True if (args.backbone in ['resnet101']) else withQuantization
    elif args.dataset == 'ade20k':
        model_prefix += '_ade'
    elif args.dataset == 'citys':
        model_prefix += '_citys'
    elif args.dataset == 'mhpv1':
        model_prefix += '_mhpv1'
    else:
        raise ValueError('Unsupported dataset {} used'.format(args.dataset))

    if args.ngpus > 0:
        withQuantization = False

    if withQuantization and args.quantized:
        model_prefix += '_int8'

    if not args.deploy:
        if args.calibration:
            args.pretrained = True
        # create network
        if args.model_zoo is not None:
            model = get_model(args.model_zoo, norm_layer=args.norm_layer,
                              norm_kwargs=args.norm_kwargs, aux=args.aux,
                              base_size=args.base_size, crop_size=args.crop_size,
                              ctx=args.ctx, pretrained=True)
        elif args.pretrained:
            if 'icnet' or 'fastscnn' in model_prefix:
                model = get_model(model_prefix, pretrained=True, height=args.height, width=args.width)
            else:
                model = get_model(model_prefix, pretrained=True)
            model.collect_params().reset_ctx(ctx=args.ctx)
        else:
            assert "_in8" not in model_prefix, "Currently, Int8 models are not supported when pretrained=False"
            model = get_segmentation_model(model=args.model, dataset=args.dataset, ctx=args.ctx,
                                           backbone=args.backbone, norm_layer=args.norm_layer,
                                           norm_kwargs=args.norm_kwargs, aux=args.aux,
                                           base_size=args.base_size, crop_size=args.crop_size,
                                           height=args.height, width=args.width)
            # load local pretrained weight
            assert args.resume is not None, '=> Please provide the checkpoint using --resume'
            if os.path.isfile(args.resume):
                model.load_parameters(args.resume, ctx=args.ctx)
            else:
                raise RuntimeError("=> no checkpoint found at '{}'" \
                    .format(args.resume))
        if args.quantized:
            model.hybridize(static_alloc=True, static_shape=True)
    else:
        model_prefix = 'deploy_int8' if args.quantized else 'deploy'
        model = mx.gluon.SymbolBlock.imports('{}-symbol.json'.format(args.model_prefix),
              ['data'], '{}-0000.params'.format(args.model_prefix))
        model.hybridize(static_alloc=True, static_shape=True)

    logger.info('Successfully loaded %s model' % model_prefix)
    logger.info('Testing model: %s' % args.resume)

    # benchmark
    if args.benchmark:
        logger.info('------benchmarking on %s model------' % model_prefix)
        benchmarking(model, args)
        sys.exit()

    # image transform
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])

    if args.calibration or '_int8' in model_prefix:
        # get dataset
        if args.eval:
            testset = get_segmentation_dataset(
                args.dataset, split='val', mode=args.mode, transform=input_transform)
        else:
            testset = get_segmentation_dataset(
                args.dataset, split='test', mode=args.mode, transform=input_transform)
        size = len(testset)
        batchify_fn = ms_batchify_fn if testset.mode == 'test' else None
        # get dataloader
        test_data = gluon.data.DataLoader(
                testset, args.batch_size, batchify_fn=batchify_fn, last_batch='rollover',
                shuffle=False, num_workers=args.workers)

        # calibration
        if not args.quantized:
            assert args.eval and args.mode == 'val', "Only val dataset can used for calibration."
            exclude_sym_layer = []
            exclude_match_layer = []
            if args.ngpus > 0:
                raise ValueError('currently only supports CPU with MKL-DNN backend')
            model = quantize_net(model, calib_data=test_data, quantized_dtype=args.quantized_dtype, calib_mode=args.calib_mode,
                                 exclude_layers=exclude_sym_layer, num_calib_examples=args.batch_size * args.num_calib_batches,
                                 exclude_layers_match=exclude_match_layer, ctx=args.ctx[0], logger=logger)
            dir_path = os.path.dirname(os.path.realpath(__file__))
            dst_dir = os.path.join(dir_path, 'model')
            if not os.path.isdir(dst_dir):
                os.mkdir(dst_dir)
            prefix = os.path.join(dst_dir, model_prefix + '-quantized-' + args.calib_mode)
            logger.info('Saving quantized model at %s' % dst_dir)
            model.export(prefix, epoch=0)
            sys.exit()

    # validation
    if '_int8' in model_prefix:
        test_quantization(model, args, test_data, size, testset.num_class, testset.pred_offset)
    else:
        test(model, args, input_transform)
