""" SiamRPN train"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import os
import time

import numpy as np
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
import mxnet as mx

from mxnet import gluon, nd, autograd
from mxnet.contrib import amp

from gluoncv.utils import LRScheduler, LRSequential, split_and_load
from gluoncv.data.tracking_data.track import TrkDataset
from gluoncv.model_zoo import get_model
from gluoncv.loss import SiamRPNLoss

def parse_args():
    """parameter test."""
    parser = argparse.ArgumentParser(description='siamrpn tracking test result')
    parser.add_argument('--model-name', type=str, default='siamrpn_alexnet_v2_otb15',
                        help='name of model.')
    parser.add_argument('--use-pretrained', action='store_true', default=False,
                        help='enable using pretrained model from gluon.')
    parser.add_argument('--dtype', type=str, default='float32',
                        help='data type for training. default is float32')
    parser.add_argument('--save-dir', type=str, default='params',
                        help='directory of saved models')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='training batch size per device (CPU/GPU).')
    parser.add_argument('--ngpus', type=int, default=8,
                        help='number of gpus to use.')
    parser.add_argument('--resume-params', type=str, default=None,
                        help='path of parameters to load from.')
    parser.add_argument('--logging-file', type=str, default='train.log',
                        help='name of training log file')
    parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=32, type=int,
                        help='number of preprocessing workers')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of training epochs.')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='training start epochs.')
    parser.add_argument('--lr-mode', type=str, default='step',
                        help='lr mode')
    parser.add_argument('--base-lr', type=float, default=0.005,
                        help='base lr')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='warmup epochs number')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        metavar='M', help='w-decay (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--no-wd', action='store_true',
                        help='whether to remove weight decay on bias, \
                        and beta/gamma for batchnorm layers.')
    parser.add_argument('--cls-weight', type=float, default=1.0,
                        help='cls weight (default: 1e-3)')
    parser.add_argument('--loc-weight', type=float, default=1.2,
                        help='loc weight (default: 1e-3)')
    parser.add_argument('--log-interval', type=int, default=128,
                        help='Logging mini-batch interval. Default is 128.')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--syncbn', action='store_true', default=False,
                        help='using Synchronized Cross-GPU BatchNorm')
    parser.add_argument('--accumulate', type=int, default=1,
                        help='new step to accumulate gradient. If >1, the batch size is enlarged.')
    parser.add_argument('--use-amp', action='store_true',
                        help='whether to use automatic mixed precision.')
    parser.add_argument('--no-val', action='store_true', default=True,
                        help='skip validation during training')
    parser.add_argument('--mode', type=str, default='hybrid',
                        help='mode in which to train the model. options are symbolic, hybrid')
    parser.add_argument('--is-train', type=str, default=True,
                        help='whether to train the model. options are True, False')
    opt = parser.parse_args()

    if opt.no_cuda:
        print('Using CPU')
        opt.ctx = [mx.cpu(0)]
    else:
        print('Number of GPUs:', opt.ngpus)
        assert opt.ngpus > 0, 'No GPUs found, please enable --no-cuda for CPU mode.'
        opt.ctx = [mx.gpu(i) for i in range(opt.ngpus)]

    # logging and checkpoint saving
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)

    # Synchronized BatchNorm
    opt.norm_layer = mx.gluon.contrib.nn.SyncBatchNorm if opt.syncbn \
        else mx.gluon.nn.BatchNorm
    opt.norm_kwargs = {'num_devices': opt.ngpus} if opt.syncbn else {}
    return opt

def train_batch_fn(data, opt):
    """split and load data in GPU"""
    template = split_and_load(data[0], ctx_list=opt.ctx, batch_axis=0)
    search = split_and_load(data[1], ctx_list=opt.ctx, batch_axis=0)
    label_cls = split_and_load(data[2], ctx_list=opt.ctx, batch_axis=0)
    label_loc = split_and_load(data[3], ctx_list=opt.ctx, batch_axis=0)
    label_loc_weight = split_and_load(data[4], ctx_list=opt.ctx, batch_axis=0)
    return template, search, label_cls, label_loc, label_loc_weight


def build_data_loader(batch_size):
    """bulid dataset and dataloader"""
    logger.info("build train dataset")
    # train_dataset
    train_dataset = TrkDataset(train_epoch=opt.epochs)
    logger.info("build dataset done")

    train_loader = gluon.data.DataLoader(train_dataset,
                                         batch_size=batch_size,
                                         last_batch='discard',
                                         num_workers=opt.num_workers)
    return train_loader

def main(logger, opt):
    """train model"""
    filehandler = logging.FileHandler(os.path.join(opt.save_dir, opt.logging_file))
    streamhandler = logging.StreamHandler()
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)
    logger.info(opt)

    if opt.use_amp:
        amp.init()

    num_gpus = opt.ngpus
    batch_size = opt.batch_size*max(1, num_gpus)
    logger.info('Total batch size is set to %d on %d GPUs' % (batch_size, num_gpus))
    train_loader = build_data_loader(batch_size)

    # create model
    net = get_model(opt.model_name, bz=opt.batch_size, is_train=opt.is_train, ctx=opt.ctx)
    net.cast(opt.dtype)
    logger.info(net)
    net.collect_params().reset_ctx(opt.ctx)
    if opt.resume_params is not None:
        if os.path.isfile(opt.resume_params):
            net.load_parameters(opt.resume_params, ctx=opt.ctx)
            print('Continue training from model %s.' % (opt.resume_params))
        else:
            raise RuntimeError("=> no checkpoint found at '{}'".format(opt.resume_params))

    # create criterion
    criterion = SiamRPNLoss(opt.batch_size)
    # optimizer and lr scheduling
    step_epoch = [10, 20, 30, 40, 50]
    num_batches = len(train_loader)
    lr_scheduler = LRSequential([LRScheduler(mode='step',
                                             base_lr=0.005,
                                             target_lr=0.01,
                                             nepochs=opt.warmup_epochs,
                                             iters_per_epoch=num_batches,
                                             step_epoch=step_epoch,
                                            ),
                                 LRScheduler(mode='poly',
                                             base_lr=0.01,
                                             target_lr=0.005,
                                             nepochs=opt.epochs-opt.warmup_epochs,
                                             iters_per_epoch=num_batches,
                                             step_epoch=[e - opt.warmup_epochs for e in step_epoch],
                                             power=0.02)])

    optimizer_params = {'lr_scheduler': lr_scheduler,
                        'wd': opt.weight_decay,
                        'momentum': opt.momentum,
                        'learning_rate': opt.lr}

    if opt.dtype == 'float32':
        optimizer_params['multi_precision'] = True

    if opt.use_amp:
        amp.init_trainer(optimizer_params)

    if opt.no_wd:
        for k, v in net.module.collect_params('.*beta|.*gamma|.*bias').items():
            v.wd_mult = 0.0

    if opt.mode == 'hybrid':
        net.hybridize(static_alloc=True, static_shape=True)

    optimizer = gluon.Trainer(net.collect_params(), 'sgd', optimizer_params)

    if opt.accumulate > 1:
        params = [p for p in net.collect_params().values() if p.grad_req != 'null']
        for p in params:
            p.grad_req = 'add'

    train(opt, net, train_loader, criterion, optimizer, batch_size, logger)

def save_checkpoint(net, opt, epoch, is_best=False):
    """Save Checkpoint"""
    filename = 'epoch_%d.params'%(epoch)
    filepath = os.path.join(opt.save_dir, filename)
    net.save_parameters(filepath)
    if is_best:
        shutil.copyfile(filename, os.path.join(opt.save_dir, 'model_best.params'))

def train(opt, net, train_loader, criterion, trainer, batch_size, logger):
    """train model"""
    for epoch in range(opt.start_epoch, opt.epochs):
        loss_total_val = 0
        loss_loc_val = 0
        loss_cls_val = 0
        batch_time = time.time()
        for i, data in enumerate(train_loader):
            template, search, label_cls, label_loc, label_loc_weight = train_batch_fn(data, opt)
            cls_losses = []
            loc_losses = []
            total_losses = []
            with autograd.record():
                for j in range(len(opt.ctx)):
                    cls, loc = net(template[j], search[j])
                    label_cls_temp = label_cls[j].reshape(-1).asnumpy()
                    pos_index = np.argwhere(label_cls_temp == 1).reshape(-1)
                    neg_index = np.argwhere(label_cls_temp == 0).reshape(-1)
                    if len(pos_index):
                        pos_index = nd.array(pos_index, ctx=opt.ctx[j])
                    else:
                        pos_index = nd.array(np.array([]), ctx=opt.ctx[j])
                    if len(neg_index):
                        neg_index = nd.array(neg_index, ctx=opt.ctx[j])
                    else:
                        neg_index = nd.array(np.array([]), ctx=opt.ctx[j])
                    cls_loss, loc_loss = criterion(cls, loc, label_cls[j], pos_index, neg_index,
                                                   label_loc[j], label_loc_weight[j])
                    total_loss = opt.cls_weight*cls_loss+opt.loc_weight*loc_loss
                    cls_losses.append(cls_loss)
                    loc_losses.append(loc_loss)
                    total_losses.append(total_loss)

                mx.nd.waitall()
                if opt.use_amp:
                    with amp.scale_loss(total_losses, trainer) as scaled_loss:
                        autograd.backward(scaled_loss)
                else:
                    autograd.backward(total_losses)
            trainer.step(batch_size)
            loss_total_val += sum([l.mean().asscalar() for l in total_losses]) / len(total_losses)
            loss_loc_val += sum([l.mean().asscalar() for l in loc_losses]) / len(loc_losses)
            loss_cls_val += sum([l.mean().asscalar() for l in cls_losses]) / len(cls_losses)
            if i%(opt.log_interval) == 0:
                logger.info('Epoch %d iteration %04d/%04d: loc loss %.3f, cls loss %.3f, \
                             training loss %.3f, batch time %.3f'% \
                            (epoch, i, len(train_loader), loss_loc_val/(i+1), loss_cls_val/(i+1),
                             loss_total_val/(i+1), time.time()-batch_time))
                batch_time = time.time()
            mx.nd.waitall()
        # save every epoch
        if opt.no_val:
            save_checkpoint(net, opt, epoch, False)

if __name__ == '__main__':
    logger = logging.getLogger('global')
    opt = parse_args()
    main(logger, opt)
