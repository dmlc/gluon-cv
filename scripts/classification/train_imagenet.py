"""
Train ImageNet
==============

"""

from __future__ import division

import argparse, time, logging, math

import mxnet as mx
import numpy as np
from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision as models
from mxnet.gluon.data.vision import transforms

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from gluonvision.datasets import imagenet
from gluonvision.utils import makedirs

# CLI
parser = argparse.ArgumentParser(description='Train a model for image classification.')
parser.add_argument('--data-dir', type=str, default='~/.mxnet/datasets/imagenet',
                    help='training and validation pictures to use.')
parser.add_argument('--batch-size', type=int, default=32,
                    help='training batch size per device (CPU/GPU).')
parser.add_argument('--num-gpus', type=int, default=0,
                    help='number of gpus to use.')
parser.add_argument('-j', '--workers', dest='num_workers', default=4, type=int,
                    help='number of preprocessing workers')
parser.add_argument('--epochs', type=int, default=3,
                    help='number of training epochs.')
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate. default is 0.1.')
parser.add_argument('-momentum', type=float, default=0.9,
                    help='momentum value for optimizer, default is 0.9.')
parser.add_argument('--wd', type=float, default=0.0001,
                    help='weight decay rate. default is 0.0001.')
parser.add_argument('--lr-decay', type=float, default=0.1,
                    help='decay rate of learning rate. default is 0.1.')
parser.add_argument('--lr-decay-period', type=int, default=0,
                    help='interval for periodic learning rate decays. default is 0 to disable.')
parser.add_argument('--lr-decay-epoch', type=str, default='40,60',
                    help='epoches at which learning rate decays. default is 40,60.')
parser.add_argument('--mode', type=str,
                    help='mode in which to train the model. options are symbolic, imperative, hybrid')
parser.add_argument('--model', type=str, required=True,
                    help='type of model to use. see vision_model for options.')
parser.add_argument('--use_thumbnail', action='store_true',
                    help='use thumbnail or not in resnet. default is false.')
parser.add_argument('--batch-norm', action='store_true',
                    help='enable batch normalization or not in vgg. default is false.')
parser.add_argument('--use-pretrained', action='store_true',
                    help='enable using pretrained model from gluon.')
parser.add_argument('--log-interval', type=int, default=50, help='Number of batches to wait before logging.')
parser.add_argument('--save-frequency', type=int, default=10,
                    help='frequency of model saving.')
parser.add_argument('--save-dir', type=str, default='params',
                    help='directory of saved models')
parser.add_argument('--logging-dir', type=str, default='logs',
                    help='directory of training logs')
opt = parser.parse_args()

batch_size = opt.batch_size
classes = 1000

num_gpus = opt.num_gpus
batch_size *= max(1, num_gpus)
context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
num_workers = opt.num_workers

lr_decay = opt.lr_decay
lr_decay_period = opt.lr_decay_period
lr_decay_epoch = [int(i) for i in opt.lr_decay_epoch.split(',')] + [1e8]

model_name = opt.model

kwargs = {'ctx': context, 'pretrained': opt.use_pretrained, 'classes': classes}
if model_name.startswith('resnet'):
    kwargs['thumbnail'] = opt.use_thumbnail
elif model_name.startswith('vgg'):
    kwargs['batch_norm'] = opt.batch_norm

optimizer = 'nag'
optimizer_params = {'learning_rate': opt.lr, 'wd': opt.wd, 'momentum': opt.momentum}

net = models.get_model(model_name, **kwargs)

acc_top1 = mx.metric.Accuracy()
acc_top5 = mx.metric.TopKAccuracy(5)

save_frequency = opt.save_frequency
if opt.save_dir and save_frequency:
    save_dir = opt.save_dir
    makedirs(save_dir)
else:
    save_dir = ''
    save_frequency = 0

logging_handlers = [logging.StreamHandler()]
if opt.logging_dir:
    logging_dir = opt.logging_dir
    makedirs(logging_dir)
    logging_handlers.append(logging.FileHandler('%s/train_imagenet_%s.log'%(logging_dir, model_name)))

logging.basicConfig(level=logging.INFO, handlers = logging_handlers)
logging.info(opt)

normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
jitter_param = 0.0 if model_name.startswith('mobilenet') else 0.4
lighting_param = 0.0 if model_name.startswith('mobilenet') else 0.1

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomFlipLeftRight(),
    transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param,
                                 saturation=jitter_param),
    transforms.RandomLighting(lighting_param),
    transforms.ToTensor(),
    normalize
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

def test(ctx, val_data):
    acc_top1.reset()
    acc_top5.reset()
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        outputs = [net(X) for X in data]
        acc_top1.update(label, outputs)
        acc_top5.update(label, outputs)

    _, top1 = acc_top1.get()
    _, top5 = acc_top5.get()
    return (1-top1, 1-top5)

def train(epochs, ctx):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    net.initialize(mx.init.Xavier(magnitude=2), ctx=ctx)

    train_data = gluon.data.DataLoader(
        imagenet.classification.ImageNet(opt.data_dir, train=True).transform_first(transform_train),
        batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=num_workers)
    val_data = gluon.data.DataLoader(
        imagenet.classification.ImageNet(opt.data_dir, train=False).transform_first(transform_test),
        batch_size=batch_size, shuffle=False, num_workers=num_workers)

    trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)
    L = gluon.loss.SoftmaxCrossEntropyLoss()

    lr_decay_count = 0

    for epoch in range(epochs):
        tic = time.time()
        acc_top1.reset()
        acc_top5.reset()
        btic = time.time()
        train_loss = 0
        num_batch = len(train_data)

        if lr_decay_period and epoch and epoch % lr_decay_period == 0:
            trainer.set_learning_rate(trainer.learning_rate*lr_decay)
        elif lr_decay_period == 0 and epoch == lr_decay_epoch[lr_decay_count]:
            trainer.set_learning_rate(trainer.learning_rate*lr_decay)
            lr_decay_count += 1

        for i, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            with ag.record():
                outputs = [net(X) for X in data]
                loss = [L(yhat, y) for yhat, y in zip(outputs, label)]
            for l in loss:
                l.backward()
            trainer.step(batch_size)
            acc_top1.update(label, outputs)
            acc_top5.update(label, outputs)
            train_loss += sum([l.sum().asscalar() for l in loss])
            if opt.log_interval and not (i+1)%opt.log_interval:
                _, top1 = acc_top1.get()
                _, top5 = acc_top5.get()
                err_top1, err_top5 = (1-top1, 1-top5)
                logging.info('Epoch[%d] Batch [%d]\tSpeed: %f samples/sec\ttop1-err=%f\ttop5-err=%f'%(
                             epoch, i, batch_size*opt.log_interval/(time.time()-btic), err_top1, err_top5))
                btic = time.time()

        _, top1 = acc_top1.get()
        _, top5 = acc_top5.get()
        err_top1, err_top5 = (1-top1, 1-top5)
        logging.info('[Epoch %d] training: err-top1=%f err-top5=%f'%(epoch, err_top1, err_top5))
        logging.info('[Epoch %d] time cost: %f'%(epoch, time.time()-tic))
        err_top1_val, err_top5_val = test(ctx, val_data)
        logging.info('[Epoch %d] validation: err-top1=%f err-top5=%f'%(epoch, err_top1_val, err_top5_val))

        if save_frequency and save_dir and (epoch + 1) % save_frequency == 0:
            net.save_params('%s/imagenet-%s-%d.params'%(save_dir, model_name, epoch))

    if save_frequency and save_dir:
        net.save_params('%s/imagenet-%s-%d.params'%(save_dir, model_name, epochs-1))

def main():
    if opt.mode == 'hybrid':
        net.hybridize()
    train(opt.epochs, context)

if __name__ == '__main__':
    main()
