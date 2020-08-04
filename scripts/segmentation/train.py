import os
import logging
import shutil
import argparse
import numpy as np
from tqdm import tqdm

import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon.data.vision import transforms

import gluoncv
gluoncv.utils.check_version('0.6.0')
from gluoncv.loss import *
from gluoncv.utils import makedirs, LRScheduler, LRSequential
from gluoncv.model_zoo.segbase import *
from gluoncv.model_zoo import get_model
from gluoncv.utils.parallel import *
from gluoncv.data import get_segmentation_dataset

def parse_args():
    """Training Options for Semantic Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='MXNet Gluon Semantic Segmentation')
    # model and dataset
    parser.add_argument('--model', type=str, default='fcn',
                        help='model name (default: fcn)')
    parser.add_argument('--model-zoo', type=str, default=None,
                        help='evaluating on model zoo model')
    parser.add_argument('--pretrained', action="store_true",
                        help='whether to use pretrained params')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        help='backbone name (default: resnet50)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        help='dataset name (default: pascal)')
    parser.add_argument('--workers', type=int, default=16,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=520,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=480,
                        help='crop image size')
    parser.add_argument('--train-split', type=str, default='train',
                        help='dataset train split (default: train)')
    # training hyper params
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')
    parser.add_argument('--aux-weight', type=float, default=0.5,
                        help='auxiliary loss weight')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=16,
                        metavar='N', help='input batch size for \
                        training (default: 16)')
    parser.add_argument('--test-batch-size', type=int, default=16,
                        metavar='N', help='input batch size for \
                        testing (default: 16)')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help='optimizer (default: sgd)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--warmup-epochs', type=int, default=0,
                        help='number of warmup epochs.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        metavar='M', help='w-decay (default: 1e-4)')
    parser.add_argument('--no-wd', action='store_true',
                        help='whether to remove weight decay on bias, \
                        and beta/gamma for batchnorm layers.')
    parser.add_argument('--mode', type=str, default=None,
                        help='whether to turn on model hybridization')
    # cuda and distribute
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--ngpus', type=int,
                        default=len(mx.test_utils.list_gpus()),
                        help='number of GPUs (default: 4)')
    parser.add_argument('--kvstore', type=str, default='device',
                        help='kvstore to use for trainer/module.')
    parser.add_argument('--dtype', type=str, default='float32',
                        help='data type for training. default is float32')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default='default',
                        help='set the checkpoint name')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='directory of saved models')
    parser.add_argument('--log-interval', type=int, default=20,
                        help='Number of batches to wait before logging.')
    parser.add_argument('--logging-file', type=str, default='train.log',
                        help='name of training log file')
    # evaluation only
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluation only')
    parser.add_argument('--no-val', action='store_true', default=False,
                            help='skip validation during training')
    # synchronized Batch Normalization
    parser.add_argument('--syncbn', action='store_true', default=False,
                        help='using Synchronized Cross-GPU BatchNorm')
    # the parser
    args = parser.parse_args()

    # handle contexts
    if args.no_cuda:
        print('Using CPU')
        args.kvstore = 'local'
        args.ctx = [mx.cpu(0)]
    else:
        print('Number of GPUs:', args.ngpus)
        assert args.ngpus > 0, 'No GPUs found, please enable --no-cuda for CPU mode.'
        args.ctx = [mx.gpu(i) for i in range(args.ngpus)]

    if 'psp' in args.model or 'deeplab' in args.model:
        assert args.crop_size % 8 == 0, ('For PSPNet and DeepLabV3 model families, '
        'we only support input crop size as multiples of 8.')

    # logging and checkpoint saving
    if args.save_dir is None:
        args.save_dir = "runs/%s/%s/%s/" % (args.dataset, args.model, args.backbone)
    makedirs(args.save_dir)

    # Synchronized BatchNorm
    args.norm_layer = mx.gluon.contrib.nn.SyncBatchNorm if args.syncbn \
        else mx.gluon.nn.BatchNorm
    args.norm_kwargs = {'num_devices': args.ngpus} if args.syncbn else {}
    return args

class Trainer(object):
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

        # dataset and dataloader
        data_kwargs = {'transform': input_transform,
                       'base_size': args.base_size,
                       'crop_size': args.crop_size}
        trainset = get_segmentation_dataset(args.dataset,
                                            split=args.train_split,
                                            mode='train',
                                            **data_kwargs)
        valset = get_segmentation_dataset(args.dataset,
                                          split='val',
                                          mode='val',
                                          **data_kwargs)
        self.train_data = gluon.data.DataLoader(trainset,
                                                args.batch_size,
                                                shuffle=True,
                                                last_batch='rollover',
                                                num_workers=args.workers)
        self.eval_data = gluon.data.DataLoader(valset,
                                               args.test_batch_size,
                                               last_batch='rollover',
                                               num_workers=args.workers)

        # create network
        if args.model_zoo is not None:
            model = get_model(args.model_zoo, norm_layer=args.norm_layer,
                              norm_kwargs=args.norm_kwargs, aux=args.aux,
                              base_size=args.base_size, crop_size=args.crop_size,
                              pretrained=args.pretrained)
        else:
            model = get_segmentation_model(model=args.model, dataset=args.dataset,
                                           backbone=args.backbone, norm_layer=args.norm_layer,
                                           norm_kwargs=args.norm_kwargs, aux=args.aux,
                                           base_size=args.base_size, crop_size=args.crop_size)
        # for resnest use only
        from gluoncv.nn.dropblock import set_drop_prob
        from functools import partial
        apply_drop_prob = partial(set_drop_prob, 0.0)
        model.apply(apply_drop_prob)

        model.cast(args.dtype)
        logger.info(model)

        self.net = DataParallelModel(model, args.ctx, args.syncbn)
        self.evaluator = DataParallelModel(SegEvalModel(model), args.ctx)
        # resume checkpoint if needed
        if args.resume is not None:
            if os.path.isfile(args.resume):
                model.load_parameters(args.resume, ctx=args.ctx)
            else:
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))

        # create criterion
        if 'icnet' in args.model:
            criterion = ICNetLoss(crop_size=args.crop_size)
        elif 'danet' in args.model or (args.model_zoo and 'danet' in args.model_zoo):
            criterion = SegmentationMultiLosses()
        else:
            criterion = MixSoftmaxCrossEntropyLoss(args.aux, aux_weight=args.aux_weight)
        self.criterion = DataParallelCriterion(criterion, args.ctx, args.syncbn)

        # optimizer and lr scheduling
        self.lr_scheduler = LRSequential([
                LRScheduler('linear', base_lr=0, target_lr=args.lr,
                            nepochs=args.warmup_epochs, iters_per_epoch=len(self.train_data)),
                LRScheduler(mode='poly', base_lr=args.lr,
                            nepochs=args.epochs-args.warmup_epochs,
                            iters_per_epoch=len(self.train_data),
                            power=0.9)
            ])
        kv = mx.kv.create(args.kvstore)

        if args.optimizer == 'sgd':
            optimizer_params = {'lr_scheduler': self.lr_scheduler,
                                'wd': args.weight_decay,
                                'momentum': args.momentum,
                                'learning_rate': args.lr}
        elif args.optimizer == 'adam':
            optimizer_params = {'lr_scheduler': self.lr_scheduler,
                                'wd': args.weight_decay,
                                'learning_rate': args.lr}
        else:
            raise ValueError('Unsupported optimizer {} used'.format(args.optimizer))

        if args.dtype == 'float16':
            optimizer_params['multi_precision'] = True

        if args.no_wd:
            for k, v in self.net.module.collect_params('.*beta|.*gamma|.*bias').items():
                v.wd_mult = 0.0

        self.optimizer = gluon.Trainer(self.net.module.collect_params(), args.optimizer,
                                       optimizer_params, kvstore=kv)
        # evaluation metrics
        self.metric = gluoncv.utils.metrics.SegmentationMetric(trainset.num_class)

    def training(self, epoch):
        tbar = tqdm(self.train_data)
        train_loss = 0.0
        for i, (data, target) in enumerate(tbar):
            with autograd.record(True):
                outputs = self.net(data.astype(args.dtype, copy=False))
                losses = self.criterion(outputs, target)
                mx.nd.waitall()
                autograd.backward(losses)
            self.optimizer.step(self.args.batch_size)
            for loss in losses:
                train_loss += np.mean(loss.asnumpy()) / len(losses)
            tbar.set_description('Epoch %d, training loss %.3f' % \
                (epoch, train_loss/(i+1)))
            if i != 0 and i % self.args.log_interval == 0:
                self.logger.info('Epoch %d iteration %04d/%04d: training loss %.3f' % \
                    (epoch, i, len(self.train_data), train_loss/(i+1)))
            mx.nd.waitall()

        # save every epoch
        if self.args.no_val:
            save_checkpoint(self.net.module, self.args, epoch, 0, False)

    def validation(self, epoch):
        self.metric.reset()
        tbar = tqdm(self.eval_data)
        for i, (data, target) in enumerate(tbar):
            outputs = self.evaluator(data.astype(args.dtype, copy=False))
            outputs = [x[0] for x in outputs]
            targets = mx.gluon.utils.split_and_load(target, args.ctx, even_split=False)
            self.metric.update(targets, outputs)
            pixAcc, mIoU = self.metric.get()
            tbar.set_description('Epoch %d, validation pixAcc: %.3f, mIoU: %.3f' % \
                (epoch, pixAcc, mIoU))
            mx.nd.waitall()
        self.logger.info('Epoch %d validation pixAcc: %.3f, mIoU: %.3f' % (epoch, pixAcc, mIoU))
        save_checkpoint(self.net.module, self.args, epoch, mIoU, False)

def save_checkpoint(net, args, epoch, mIoU, is_best=False):
    """Save Checkpoint"""
    filename = 'epoch_%04d_mIoU_%2.4f.params' % (epoch, mIoU)
    filepath = os.path.join(args.save_dir, filename)
    net.save_parameters(filepath)
    if is_best:
        shutil.copyfile(filename, os.path.join(args.save_dir, 'model_best.params'))

if __name__ == "__main__":
    args = parse_args()

    # build logger
    filehandler = logging.FileHandler(os.path.join(args.save_dir, args.logging_file))
    streamhandler = logging.StreamHandler()
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)
    logger.info(args)

    trainer = Trainer(args, logger)
    if args.eval:
        logger.info('Evaluating model: %s' % args.resume)
        trainer.validation(args.start_epoch)
    else:
        logger.info('Starting Epoch: %d' % args.start_epoch)
        logger.info('Total Epochs: %d' % args.epochs)
        for epoch in range(args.start_epoch, args.epochs):
            trainer.training(epoch)
            if not trainer.args.no_val:
                trainer.validation(epoch)