import os
import shutil
import argparse
import numpy as np
from tqdm import tqdm

import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon.data.vision import transforms

from gluoncv.utils import LRScheduler
from gluoncv.model_zoo.segbase import *
from gluoncv.utils.parallel import *
from gluoncv.data import get_segmentation_dataset


def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='MXNet Gluon \
                                     Segmentation')
    # model and dataset 
    parser.add_argument('--model', type=str, default='fcn',
                        help='model name (default: fcn)')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        help='backbone name (default: resnet50)')
    parser.add_argument('--dataset', type=str, default='pascalaug',
                        help='dataset name (default: pascal)')
    parser.add_argument('--workers', type=int, default=16,
                        metavar='N', help='dataloader threads')
    # training hyper params
    parser.add_argument('--aux', action='store_true', default= False,
                        help='Auxilary loss')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=16,
                        metavar='N', help='input batch size for \
                        training (default: 16)')
    parser.add_argument('--test-batch-size', type=int, default=16,
                        metavar='N', help='input batch size for \
                        testing (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        metavar='M', help='w-decay (default: 1e-4)')
    # cuda and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--ngpus', type=int,
                        default=len(mx.test_utils.list_gpus()),
                        help='number of GPUs (default: 4)')
    parser.add_argument('--kvstore', type=str, default='device',
                        help='kvstore to use for trainer/module.')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default='default',
                        help='set the checkpoint name')
    parser.add_argument('--model-zoo', type=str, default=None,
                        help='evaluating on model zoo model')
    # evaluation only
    parser.add_argument('--eval', action='store_true', default= False,
                        help='evaluation only')
    # synchronized Batch Normalization
    parser.add_argument('--syncbn', action='store_true', default= False,
                        help='using Synchronized Cross-GPU BatchNorm')
    # the parser
    args = parser.parse_args()
    # handle contexts
    if args.no_cuda:
        print('Using CPU')
        args.kvstore = 'local'
        args.ctx = mx.cpu(0)
    else:
        print('Number of GPUs:', args.ngpus)
        args.ctx = [mx.gpu(i) for i in range(args.ngpus)]
    # Synchronized BatchNorm
    if args.syncbn:
        from gluoncv.model_zoo.syncbn import BatchNorm
        args.norm_layer = BatchNorm
    else:
        args.norm_layer = mx.gluon.nn.BatchNorm
    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args
        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        # dataset and dataloader
        trainset = get_segmentation_dataset(
            args.dataset, split='train', transform=input_transform)
        valset = get_segmentation_dataset(
            args.dataset, split='val', transform=input_transform)
        self.train_data = gluon.data.DataLoader(
            trainset, args.batch_size, shuffle=True, last_batch='rollover',
            num_workers=args.workers)
        self.eval_data = gluon.data.DataLoader(valset, args.test_batch_size,
            last_batch='keep', num_workers=args.workers)
        # create network
        model = get_segmentation_model(model=args.model, dataset=args.dataset,
                                       backbone=args.backbone, norm_layer=args.norm_layer,
                                       aux=args.aux)
        print(model)
        self.net = DataParallelModel(model, args.ctx, args.syncbn)
        self.evaluator = DataParallelModel(SegEvalModel(model), args.ctx)
        # resume checkpoint if needed
        if args.resume is not None:
            if os.path.isfile(args.resume):
                model.load_params(args.resume, ctx=args.ctx)
            else:
                raise RuntimeError("=> no checkpoint found at '{}'" \
                    .format(args.resume))
        # create criterion
        criterion = SoftmaxCrossEntropyLossWithAux(args.aux)
        self.criterion = DataParallelCriterion(criterion, args.ctx, args.syncbn)
        # optimizer and lr scheduling
        self.lr_scheduler = LRScheduler(mode='poly', baselr=args.lr, niters=len(self.train_data), 
                                        nepochs=args.epochs)
        kv = mx.kv.create(args.kvstore)
        self.optimizer = gluon.Trainer(self.net.module.collect_params(), 'sgd',
                                       {'lr_scheduler': self.lr_scheduler,
                                        'wd':args.weight_decay,
                                        'momentum': args.momentum,
                                        'multi_precision': True},
                                        kvstore = kv)

    def training(self, epoch):
        tbar = tqdm(self.train_data)
        train_loss = 0.0
        for i, (data, target) in enumerate(tbar):
            self.lr_scheduler.update(i, epoch)
            with autograd.record(True):
                outputs = self.net(data)
                losses = self.criterion(outputs, target)
                mx.nd.waitall()
                autograd.backward(losses)
            self.optimizer.step(self.args.batch_size)
            for loss in losses:
                train_loss += loss.asnumpy()[0] / len(losses)
            tbar.set_description('Epoch %d, training loss %.3f'%\
                (epoch, train_loss/(i+1)))
            mx.nd.waitall()

        # save every epoch
        save_checkpoint(self.net.module, self.args, False)

    def validation(self, epoch):
        total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        tbar = tqdm(self.eval_data)
        for i, (data, target) in enumerate(tbar):
            outputs = self.evaluator(data, target)
            for (correct, labeled, inter, union) in outputs:
                total_correct += correct
                total_label += labeled
                total_inter += inter
                total_union += union
            pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
            IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
            mIoU = IoU.mean()
            tbar.set_description('Epoch %d, validation pixAcc: %.3f, mIoU: %.3f'%\
                (epoch, pixAcc, mIoU))
            mx.nd.waitall()


def save_checkpoint(net, args, is_best=False):
    """Save Checkpoint"""
    directory = "runs/%s/%s/%s/" % (args.dataset, args.model, args.checkname)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename='checkpoint.params'
    filename = directory + filename
    net.save_params(filename)
    if is_best:
        shutil.copyfile(filename, directory + 'model_best.params')


if __name__ == "__main__":
    args = parse_args()
    trainer = Trainer(args)
    if args.eval:
        print('Evaluating model: ', args.resume)
        trainer.validation(args.start_epoch)
    else:
        print('Starting Epoch:', args.start_epoch)
        print('Total Epoches:', args.epochs)
        for epoch in range(args.start_epoch, args.epochs):
            trainer.training(epoch)
            trainer.validation(epoch)
