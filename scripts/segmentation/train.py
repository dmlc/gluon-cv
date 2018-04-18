import numpy as np
from tqdm import tqdm

import mxnet as mx
from mxnet import gluon, autograd
import gluonvision.utils as utils
from gluonvision.model_zoo.segbase import SegEvalModel
from gluonvision.utils.parallel import ModelDataParallel

from option import Options
from utils import save_checkpoint, get_data_loader, get_model_criterion


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.net, self.criterion = get_model_criterion(args)
        if args.test:
            self.test_data = get_data_loader(args)
        else:
            self.train_data, self.eval_data = get_data_loader(args)
            self.lr_scheduler = utils.PolyLRScheduler(args.lr, niters=len(self.train_data), 
                                                      nepochs=args.epochs)
            kv = mx.kv.create(args.kvstore)
            self.optimizer = gluon.Trainer(self.net.module.collect_params(), 'sgd',
                                           {'lr_scheduler': self.lr_scheduler,
                                            'wd':args.weight_decay,
                                            'momentum': args.momentum,
                                            'multi_precision': True},
                                            kvstore = kv)
        self.evaluator = ModelDataParallel(SegEvalModel(self.net.module, args.bg), args.ctx)

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
                train_loss += loss.asnumpy()[0] / self.args.batch_size * len(losses)
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


def main(args):
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


if __name__ == "__main__":
    args = Options().parse()
    main(args)
