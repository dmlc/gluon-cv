import os
import shutil
import numpy as np
from tqdm import tqdm

import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon.data.vision import transforms

import gluoncv

gluoncv.utils.check_version('0.6.0')
from gluoncv.utils import LRScheduler

from gluoncv.model_zoo.segbase import *
from gluoncv.model_zoo.icnet_hybridize import get_icnet_resnet50_citys
from gluoncv.loss import ICNetLoss

from gluoncv.utils.parallel import *
from gluoncv.data import get_segmentation_dataset


class Trainer(object):
    def __init__(self, ngpus=8, base_size=2048, crop_size=768, batch_size=16,
                 test_batch_size=16, num_workers=48, syncbn=True, epochs=120):
        # gpus
        self.ngpus = ngpus
        self.ctx = [mx.cpu(0)]
        self.ctx = [mx.gpu(i) for i in range(ngpus)] if ngpus > 0 else self.ctx

        # best results
        self.best_mIoU = 0.0

        ######################### Dataset and DataLoader ###############################
        self.base_size, self.crop_size = base_size, crop_size
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.syncbn = syncbn

        self.norm_layer = mx.gluon.contrib.nn.SyncBatchNorm if self.syncbn \
            else mx.gluon.nn.BatchNorm
        self.norm_kwargs = {'num_devices': self.ngpus} if self.syncbn else {}

        # image transform
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])
        data_kwargs = {'transform': input_transform, 'base_size': self.base_size,
                       'crop_size': self.crop_size}
        # get dataset
        trainset = get_segmentation_dataset('citys', split='train', mode='train', **data_kwargs)
        self.train_data = gluon.data.DataLoader(
            dataset=trainset, batch_size=self.batch_size,
            shuffle=True, last_batch='rollover', num_workers=self.num_workers
        )

        valset = get_segmentation_dataset('citys', split='val', mode='val', **data_kwargs)
        self.eval_data = gluon.data.DataLoader(
            dataset=valset, batch_size=self.test_batch_size,
            shuffle=False, last_batch='rollover', num_workers=self.num_workers
        )

        ######################### Training Model ###############################
        ## get the model

        model = get_icnet_resnet50_citys(pretrained=False, ctx=self.ctx,
                                         norm_layer=self.norm_layer, norm_kwargs=self.norm_kwargs,
                                         base_size=base_size, crop_size=crop_size)
        model.cast('float32')
        # model.hybridize()

        self.net = DataParallelModel(model, ctx_list=self.ctx, sync=self.syncbn)
        print('Training model ICNet-PSP50')

        self.evaluator = DataParallelModel(SegEvalModel(model), ctx_list=self.ctx)

        ######################### Training Setting ###############################
        weights = (0.4, 0.4, 1.0)
        lr = 0.01
        weight_decay = 1e-4
        momentum = 0.9

        criterion = ICNetLoss(weights=weights)

        self.criterion = DataParallelCriterion(criterion, ctx_list=self.ctx, sync=True)
        # optimizer and lr scheduling
        self.lr_scheduler = LRScheduler(mode='poly', base_lr=lr, nepochs=epochs,
                                        iters_per_epoch=len(self.train_data), power=0.9)
        kv = mx.kv.create('device')

        optimizer_params = {
            'lr_scheduler': self.lr_scheduler,
            'wd': weight_decay,
            'momentum': momentum,
            'learning_rate': lr
        }

        self.optimizer = gluon.Trainer(self.net.module.collect_params(), 'sgd',
                                       optimizer_params, kvstore=kv)
        # evaluation metrics
        self.metric = gluoncv.utils.metrics.SegmentationMetric(trainset.num_class)

    def training(self, epoch):
        tbar = tqdm(self.train_data)
        train_loss = 0.0

        for i, (data, target) in enumerate(tbar):
            with autograd.record(True):
                outputs = self.net(data.astype('float32', copy=False))
                losses = self.criterion(outputs, target)
                mx.nd.waitall()
                autograd.backward(losses)
            self.optimizer.step(self.batch_size)
            for loss in losses:
                train_loss += np.mean(loss.asnumpy()) / len(losses)
            tbar.set_description('Epoch %d, training loss %.3f' % (epoch, train_loss / (i + 1)))
            mx.nd.waitall()

        # save every epoch
        save_checkpoint(self.net.module, False)

    def validation(self, epoch):
        self.metric.reset()
        tbar = tqdm(self.eval_data)
        pixAcc, mIoU, num = 0, 0, 0

        for i, (data, target) in enumerate(tbar):
            outputs = self.evaluator(data.astype('float32', copy=False))
            outputs = [x[0] for x in outputs]
            targets = mx.gluon.utils.split_and_load(target, ctx_list=self.ctx, even_split=False)
            self.metric.update(targets, outputs)
            pixAcc, mIoU = self.metric.get()
            tbar.set_description('Epoch %d, validation pixAcc: %.3f, mIoU: %.3f' % (epoch, pixAcc, mIoU))
            mx.nd.waitall()

        if mIoU > self.best_mIoU:
            save_checkpoint(self.net.module, is_best=True)
            self.best_mIoU = mIoU

        return pixAcc, mIoU


def save_checkpoint(net, is_best=False):
    """Save Checkpoint"""
    directory = "./training/icnet_psp50/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = 'icnet_psp50.params'
    filename = directory + filename
    net.save_parameters(filename)

    if is_best:
        shutil.copyfile(filename, directory + 'icnet_psp50_best.params')


if __name__ == '__main__':
    pixAcc, mIoU = 0, 0

    # training
    epochs = 240
    trainer = Trainer(epochs=epochs)

    for epoch in range(epochs):
        trainer.training(epoch)
        pixAcc, mIoU = trainer.validation(epoch)

    # training log
    model_prefix = 'icnet_psp50_citys_'

    output_directory = './training/icnet_psp50/'

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    output_filename = model_prefix
    test_txt = os.path.join(output_directory, output_filename + 'test.txt')

    # record accuracy
    with open(test_txt, 'w') as txtfile:
        txtfile.write("pixAcc={:.3f}\nmIoU={:.3f}\n".format(pixAcc, mIoU))
