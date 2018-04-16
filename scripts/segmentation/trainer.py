import os
import threading
import numpy as np
from tqdm import tqdm, trange

import mxnet as mx
from mxnet import gluon
import mxnet.ndarray as F
from mxnet import autograd as ag

import gluonvision.utils as utils
from data_utils import get_data_loader
from model_utils import get_model_criterion

from evaluator import Evaluator
from utils import save_checkpoint, get_mask

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
        self.evaluator = Evaluator(args)
        self.outdir = 'test'

    def training(self, epoch):
        tbar = tqdm(self.train_data)
        train_loss = 0.0
        for i, (data, target) in enumerate(tbar):
            self.lr_scheduler.update(i, epoch)
            inputs = gluon.utils.split_and_load(data, ctx_list=self.args.ctx)
            targets = gluon.utils.split_and_load(target, ctx_list=self.args.ctx)
            with ag.record():
                outputs = self.net(inputs)
                losses = self.criterion(outputs, targets)
                mx.nd.waitall()
                ag.backward(losses)
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
            inputs = gluon.utils.split_and_load(data, ctx_list=self.args.ctx)
            targets = gluon.utils.split_and_load(target, ctx_list=self.args.ctx)
            outputs = self.net(inputs)
            correct, labeled, inter, union = self.evaluator.test_batch(outputs, targets)
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


    def test(self):
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        global count
        count = 0
        lock = threading.Lock()
        def _worker(i, model, image, impath, args):
            if self.args.cuda:
                image = image.as_in_context(args.ctx[i])
            image = image.expand_dims(0)
            output = self.evaluator. multi_eval_batch(image, model)
            predict = F.squeeze(F.argmax(output, 1)).asnumpy()
            # visualize format
            newmask = get_mask(predict, self.args.dataset)
            outname = os.path.splitext(impath)[0] + '.png'
            newmask.save(os.path.join(self.outdir, outname))
            global count
            with lock:
                count += 1

        ngpus = len(self.args.ctx)
        print('len(test_data)', len(self.test_data))
        tbar = trange(0, len(self.test_data), ngpus)
        for j in tbar:
            images = []
            paths = []
            for i in range(ngpus):
                idx = j + i
                if idx >= len(self.test_data):
                    break
                img, path = self.test_data[idx]
                images.append(img)
                paths.append(path)
            threads = [threading.Thread(target=_worker,
                         args=(m, self.net, image, path, self.args),
                                    )
                   for m, (image, path) in
                   enumerate(zip(images, paths))]

            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            tbar.set_description('Generating image count: %d'%(count))
