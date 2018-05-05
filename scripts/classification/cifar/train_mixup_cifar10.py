"""
Train on CIFAR-10 with Mixup
============================

"""

from __future__ import division

import matplotlib
matplotlib.use('Agg')

import argparse, time, logging, random, math

import numpy as np
import mxnet as mx

from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

from gluoncv.model_zoo import get_model
from gluoncv.data import transforms as gcv_transforms
from gluoncv.utils import makedirs, TrainingHistory

# CLI
parser = argparse.ArgumentParser(description='Train a model for image classification.')
parser.add_argument('--batch-size', type=int, default=32,
                    help='training batch size per device (CPU/GPU).')
parser.add_argument('--num-gpus', type=int, default=0,
                    help='number of gpus to use.')
parser.add_argument('--model', type=str, default='resnet',
                    help='model to use. options are resnet and wrn. default is resnet.')
parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=4, type=int,
                    help='number of preprocessing workers')
parser.add_argument('--num-epochs', type=int, default=3,
                    help='number of training epochs.')
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate. default is 0.1.')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum value for optimizer, default is 0.9.')
parser.add_argument('--wd', type=float, default=0.0001,
                    help='weight decay rate. default is 0.0001.')
parser.add_argument('--lr-decay', type=float, default=0.1,
                    help='decay rate of learning rate. default is 0.1.')
parser.add_argument('--lr-decay-period', type=int, default=0,
                    help='period in epoch for learning rate decays. default is 0 (has no effect).')
parser.add_argument('--lr-decay-epoch', type=str, default='40,60',
                    help='epoches at which learning rate decays. default is 40,60.')
parser.add_argument('--drop-rate', type=float, default=0.0,
                    help='dropout rate for wide resnet. default is 0.')
parser.add_argument('--mode', type=str,
                    help='mode in which to train the model. options are imperative, hybrid')
parser.add_argument('--save-period', type=int, default=10,
                    help='period in epoch of model saving.')
parser.add_argument('--save-dir', type=str, default='params',
                    help='directory of saved models')
parser.add_argument('--logging-dir', type=str, default='logs',
                    help='directory of training logs')
parser.add_argument('--resume-from', type=str,
                    help='resume training from the model')
parser.add_argument('--save-plot-dir', type=str, default='.',
                    help='the path to save the history plot')
opt = parser.parse_args()

batch_size = opt.batch_size
classes = 10

num_gpus = opt.num_gpus
batch_size *= max(1, num_gpus)
context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
num_workers = opt.num_workers

lr_decay = opt.lr_decay
lr_decay_epoch = [int(i) for i in opt.lr_decay_epoch.split(',')] + [np.inf]

model_name = opt.model
if model_name.startswith('cifar_wideresnet'):
    kwargs = {'classes': classes,
              'drop_rate': opt.drop_rate}
else:
    kwargs = {'classes': classes}
net = get_model(model_name, **kwargs)
model_name += '_mixup'
if opt.resume_from:
    net.load_params(opt.resume_from, ctx = context)
optimizer = 'nag'

save_period = opt.save_period
if opt.save_dir and save_period:
    save_dir = opt.save_dir
    makedirs(save_dir)
else:
    save_dir = ''
    save_period = 0

plot_name = opt.save_plot_dir

logging_handlers = [logging.StreamHandler()]
if opt.logging_dir:
    logging_dir = opt.logging_dir
    makedirs(logging_dir)
    logging_handlers.append(logging.FileHandler('%s/train_cifar10_%s.log'%(logging_dir, model_name)))

logging.basicConfig(level=logging.INFO, handlers = logging_handlers)
logging.info(opt)

transform_train = transforms.Compose([
    gcv_transforms.RandomCrop(32, pad=4),
    transforms.RandomFlipLeftRight(),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

def label_transform(label, classes):
    ind = label.astype('int')
    res = nd.zeros((ind.shape[0], classes), ctx = label.context)
    res[nd.arange(ind.shape[0], ctx = label.context), ind] = 1
    return res

def test(ctx, val_data):
    metric = mx.metric.Accuracy()
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        outputs = [net(X) for X in data]
        metric.update(label, outputs)
    return metric.get()

def train(epochs, ctx):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    net.initialize(mx.init.Xavier(), ctx=ctx)

    train_data = gluon.data.DataLoader(
        gluon.data.vision.CIFAR10(train=True).transform_first(transform_train),
        batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=num_workers)

    val_data = gluon.data.DataLoader(
        gluon.data.vision.CIFAR10(train=False).transform_first(transform_test),
        batch_size=batch_size, shuffle=False, num_workers=num_workers)

    trainer = gluon.Trainer(net.collect_params(), optimizer,
                            {'learning_rate': opt.lr, 'wd': opt.wd, 'momentum': opt.momentum})
    metric = mx.metric.Accuracy()
    train_metric = mx.metric.RMSE()
    loss_fn = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False)
    train_history = TrainingHistory(['training-error', 'validation-error'])

    iteration = 0
    lr_decay_count = 0

    best_val_score = 0

    for epoch in range(epochs):
        tic = time.time()
        train_metric.reset()
        metric.reset()
        train_loss = 0
        num_batch = len(train_data)
        alpha = 1

        if epoch == lr_decay_epoch[lr_decay_count]:
            trainer.set_learning_rate(trainer.learning_rate*lr_decay)
            lr_decay_count += 1

        for i, batch in enumerate(train_data):
            lam = np.random.beta(alpha, alpha)
            if epoch >= epochs - 20:
                lam = 1

            data_1 = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label_1 = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)

            data = [lam*X + (1-lam)*X[::-1] for X in data_1]
            label = []
            for Y in label_1:
                y1 = label_transform(Y, classes)
                y2 = label_transform(Y[::-1], classes)
                label.append(lam*y1 + (1-lam)*y2)

            with ag.record():
                output = [net(X) for X in data]
                loss = [loss_fn(yhat, y) for yhat, y in zip(output, label)]
            for l in loss:
                l.backward()
            trainer.step(batch_size)
            train_loss += sum([l.sum().asscalar() for l in loss])

            output_softmax = [nd.SoftmaxActivation(out) for out in output]
            train_metric.update(label, output_softmax)
            name, acc = train_metric.get()
            iteration += 1

        train_loss /= batch_size * num_batch
        name, acc = train_metric.get()
        name, val_acc = test(ctx, val_data)
        train_history.update([acc, 1-val_acc])
        train_history.plot(save_path='%s/%s_history.png'%(plot_name, model_name))

        if val_acc > best_val_score:
            best_val_score = val_acc
            net.save_params('%s/%.4f-imagenet-%s-%d-best.params'%(save_dir, best_val_score, model_name, epoch))

        name, val_acc = test(ctx, val_data)
        logging.info('[Epoch %d] train=%f val=%f loss=%f time: %f' %
            (epoch, acc, val_acc, train_loss, time.time()-tic))

        if save_period and save_dir and (epoch + 1) % save_period == 0:
            net.save_params('%s/cifar10-%s-%d.params'%(save_dir, model_name, epoch))

    if save_period and save_dir:
        net.save_params('%s/cifar10-%s-%d.params'%(save_dir, model_name, epochs-1))

def main():
    if opt.mode == 'hybrid':
        net.hybridize()
    train(opt.num_epochs, context)

if __name__ == '__main__':
    main()
