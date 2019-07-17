import os
from tqdm import tqdm
import numpy as np
import argparse
import time
import sys

import mxnet as mx
from mxnet import gluon
from mxnet.gluon.data.vision import transforms

import gluoncv
from gluoncv.model_zoo.segbase import *
from gluoncv.model_zoo import get_model
from gluoncv.data import get_segmentation_dataset, ms_batchify_fn
from gluoncv.utils.viz import get_color_pallete

def parse_args():
    parser = argparse.ArgumentParser(description='Validation on Segmentation model')
    # model and dataset
    parser.add_argument('--model', type=str, default='fcn',
                        help='model name (default: fcn)')
    parser.add_argument('--backbone', type=str, default='resnet101',
                        help='base network')
    parser.add_argument('--base-size', type=int, default=520,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=480,
                        help='crop image size')
    parser.add_argument('--mode', type=str, default='val',
                        help='val, testval')
    parser.add_argument('--dataset', type=str, default='pascal_voc',
                        help='dataset used for validation [pascal_voc, pascal_aug, coco, ade20k]')
    parser.add_argument('--quantized', action='store_true', 
                        help='whether to use quantized model')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-batches', type=int, default=100)
    parser.add_argument('--workers', type=int, default=4,
                        help='number of workers for data loading')
    parser.add_argument('--pretrained', action="store_true",
                        help='whether to use pretrained params')
    parser.add_argument('--ngpus', type=int, default=0)
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')
    # synchronized Batch Normalization
    parser.add_argument('--syncbn', action='store_true', default=False,
                        help='using Synchronized Cross-GPU BatchNorm')
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    # evaluation only
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluation only')

    args = parser.parse_args()
    
    args.ctx = [mx.cpu(0)]
    args.ctx = [mx.gpu(i) for i in range(args.ngpus)] if args.ngpus > 0 else args.ctx

    args.norm_layer = mx.gluon.contrib.nn.SyncBatchNorm if args.syncbn \
        else mx.gluon.nn.BatchNorm
    args.norm_kwargs = {'num_devices': args.ngpus} if args.syncbn else {}
    return args
    

def test(args, model):
    # output folder
    outdir = 'outdir'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # image transform
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    data_kwargs = {'transform': input_transform, 'base_size': args.base_size,
                   'crop_size': args.crop_size}
    # get dataset
    if args.eval:
        testset = get_segmentation_dataset(
            args.dataset, split='val', mode=args.mode, **data_kwargs)
    else:
        testset = get_segmentation_dataset(
            args.dataset, split='test', mode=args.mode, **data_kwargs)
    size = len(testset)
    # get dataloader
    test_data = gluon.data.DataLoader(
            testset, args.batch_size, last_batch='keep', shuffle=False, num_workers=args.workers)

    print(model)
    if not args.eval:
        evaluator = MultiEvalModel(model, testset.num_class, ctx_list=args.ctx)
    metric = gluoncv.utils.metrics.SegmentationMetric(testset.num_class)

    tbar = tqdm(test_data)
    metric.reset()
    tic = time.time()
    for i, (batch, dsts) in enumerate(tbar):
        if args.eval:
            targets = mx.gluon.utils.split_and_load(dsts, ctx_list=args.ctx, even_split=False)
            data = mx.gluon.utils.split_and_load(batch, ctx_list=args.ctx, batch_axis=0, even_split=False)
            outputs = None
            for x in data:
                output = model.forward(x)[0]
                outputs = output if outputs is None else nd.concat(outputs, output, axis=0)
            outputs = [outputs]
            metric.update(targets, outputs)
            pixAcc, mIoU = metric.get()
            tbar.set_description( 'pixAcc: %.4f, mIoU: %.4f' % (pixAcc, mIoU))
        else:
            im_paths = dsts
            predicts = evaluator.parallel_forward(batch)
            for predict, impath in zip(predicts, im_paths):
                predict = mx.nd.squeeze(mx.nd.argmax(predict[0], 1)).asnumpy() + \
                    testset.pred_offset
                mask = get_color_pallete(predict, args.dataset)
                outname = os.path.splitext(impath)[0] + '.png'
                mask.save(os.path.join(outdir, outname))
    speed = size / (time.time() - tic)
    print('Inference speed with batchsize %d is %.2f img/sec' % (args.batch_size, speed))


if __name__ == "__main__":
    args = parse_args()

    withQuantization = False
    model_prefix = args.model + '_' + args.backbone
    if 'pascal' in args.dataset:
        model_prefix += '_voc'
        withQuantization = True if (args.backbone in ['resnet101'] and args.ngpus == 0) else withQuantization
    elif args.dataset == 'coco':
        model_prefix += '_coco'
        withQuantization = True if args.backbone in ['resnet101'] else withQuantization
    elif args.dataset == 'ade20k':
        model_prefix += 'ade'
    elif args.dataset == 'citys':
        model_prefix += 'citys'
    else:
        raise ValueError('Unsupported dataset {} used'.format(args.dataset))

    if withQuantization and args.quantized:
        model_prefix += '_int8'

    if args.quantized and args.mode != 'val':
        raise ValueError("Currently, %s mode or is not supported by quantized model." % args.mode)

    if args.quantized and args.eval == False:
        raise ValueError("Currently, only evaluation is supported by quantized model.")

     # create network
    if args.pretrained:
        model = get_model(model_prefix, pretrained=True)
        model.collect_params().reset_ctx(ctx = args.ctx)
    else:
        model = get_segmentation_model(model=args.model, dataset=args.dataset, ctx=args.ctx,
                                       backbone=args.backbone, norm_layer=args.norm_layer,
                                       norm_kwargs=args.norm_kwargs, aux=args.aux,
                                       base_size=args.base_size, crop_size=args.crop_size)
        # load local pretrained weight
        assert args.resume is not None, '=> Please provide the checkpoint using --resume'
        if os.path.isfile(args.resume):
            model.load_parameters(args.resume, ctx=args.ctx)
        else:
            raise RuntimeError("=> no checkpoint found at '{}'" \
                .format(args.resume))

    print("Successfully loaded %s model" % model_prefix)
    if '_int8' in model_prefix:
        model.hybridize(static_alloc=True, static_shape=True)
    else:
        model.hybridize()

    print('Testing model: ', args.resume)
    test(args, model)
