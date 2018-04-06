from __future__ import division

import argparse, time, logging, random, math

import numpy as np
import mxnet as mx

from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.test_utils import get_mnist_iterator
from mxnet.gluon.data.vision import transforms

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from gluonvision.models.cifar import resnet, wideresnet
from gluonvision.utils import makedirs

# CLI
parser = argparse.ArgumentParser(description='Train a model for image classification.')
parser.add_argument('--batch-size', type=int, default=32,
                    help='training batch size per device (CPU/GPU).')
parser.add_argument('--num-gpus', type=int, default=0,
                    help='number of gpus to use.')
parser.add_argument('--model', type=str, default='resnet',
                    help='model to use. options are resnet and wrn. default is resnet.')
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
parser.add_argument('--lr-decay-epoch', type=str, default='40,60',
                    help='epoches at which learning rate decays. default is 40,60.')
parser.add_argument('--num-layers', type=int,
                    help='number of layers. need to be 6*n+2 for resnet and 6*n+4 for wrn')
parser.add_argument('--resnet-version', type=int, default=2,
                    help='version of resnet. default is 1.')
parser.add_argument('--width-factor', type=int, default=1,
                    help='width factor for wide resnet. default is 1.')
parser.add_argument('--drop-rate', type=float, default=0.0,
                    help='dropout rate for wide resnet. default is 0.')
parser.add_argument('--mode', type=str,
                    help='mode in which to train the model. options are imperative, hybrid')
parser.add_argument('--mixup', action='store_true',
                    help='enable using mixup training.')
parser.add_argument('--save-frequency', type=int, default=10,
                    help='frequency of model saving.')
parser.add_argument('--save-dir', type=str, default='params',
                    help='directory of saved models')
parser.add_argument('--logging-dir', type=str, default='logs',
                    help='directory of training logs')
opt = parser.parse_args()

batch_size = opt.batch_size
classes = 10

num_gpus = opt.num_gpus
batch_size *= max(1, num_gpus)
context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
num_workers = opt.num_workers

lr_decay = opt.lr_decay
lr_decay_epoch = [int(i) for i in opt.lr_decay_epoch.split(',')] + [np.inf]

num_layers = opt.num_layers
resnet_version = opt.resnet_version

kwargs = {'ctx': context, 'classes': classes}

model = opt.model
if model == 'wrn':
    net = wideresnet.get_wide_resnet(resnet_version, num_layers,
                                     opt.drop_rate, opt.width_factor, **kwargs)
    model_name = 'wrn' + '-' + str(num_layers) + '-' + str(opt.width_factor)
    optimizer = 'nag'
else:
    net = resnet.get_resnet(resnet_version, num_layers, **kwargs)
    model_name = 'resnet' + str(num_layers) + '_v' + str(resnet_version)
    optimizer = 'sgd'

if opt.mixup:
    model_name += '_mixup'

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
    logging_handlers.append(logging.FileHandler('%s/train_cifar10_%s.log'%(logging_dir, model_name)))

logging.basicConfig(level=logging.INFO, handlers = logging_handlers)
logging.info(opt)

transform_train = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomResizedCrop(32),
    transforms.RandomFlipLeftRight(),
    transforms.RandomColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.RandomLighting(0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

transform_test = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

def label_transform(label, classes):
    ind = label.asnumpy().astype('int')
    res = np.zeros((ind.shape[0], classes))
    res[np.arange(ind.shape[0]), ind] = 1
    return nd.array(res, ctx = label.context)

def test(ctx, val_data):
    metric = mx.metric.Accuracy()
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        outputs = [net(X) for X in data]
        metric.update(label, outputs)
    return metric.get()

def train(epochs, mixup, ctx):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    net.initialize(mx.init.Xavier(), ctx=ctx)

    train_data = gluon.data.DataLoader(
        gluon.data.vision.CIFAR10('./data', train=True).transform_first(transform_train),
        batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=num_workers)

    val_data = gluon.data.DataLoader(
        gluon.data.vision.CIFAR10('./data', train=False).transform_first(transform_test),
        batch_size=batch_size, shuffle=False, num_workers=num_workers)

    trainer = gluon.Trainer(net.collect_params(), optimizer,
                            {'learning_rate': opt.lr, 'wd': opt.wd, 'momentum': opt.momentum})
    metric = mx.metric.Accuracy()
    if mixup:
        metric_train = mx.metric.RMSE()
        L = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False)
    else:
        metric_train = mx.metric.Accuracy()
        L = gluon.loss.SoftmaxCrossEntropyLoss()

    iteration = 0
    lr_decay_count = 0

    for epoch in range(epochs):
        tic = time.time()
        metric_train.reset()
        metric.reset()
        train_loss = 0
        num_batch = len(train_data)
        alpha = 1

        if epoch == lr_decay_epoch[lr_decay_count]:
            trainer.set_learning_rate(trainer.learning_rate*lr_decay)
            lr_decay_count += 1

        for i, batch in enumerate(train_data):
            if mixup:
                lam = np.random.beta(alpha, alpha)
                if epoch >= epochs - 50:
                    lam = 1

                data_1 = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
                label_1 = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)

                data = [lam*X + (1-lam)*X[::-1] for X in data_1]
                label = []
                for Y in label_1:
                    y1 = label_transform(Y, classes)
                    y2 = label_transform(Y[::-1], classes)
                    label.append(lam*y1 + (1-lam)*y2)
            else:
                data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
                label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)

            with ag.record():
                output = [net(X) for X in data]
                loss = [L(yhat, y) for yhat, y in zip(output, label)]
            for l in loss:
                l.backward()
            trainer.step(batch_size)
            train_loss += sum([l.sum().asscalar() for l in loss])

            if mixup:
                output_softmax = [nd.SoftmaxActivation(out) for out in output]
                metric_train.update(label, output_softmax)
            else:
                metric_train.update(label, output)
            name, acc = metric_train.get()
            iteration += 1

        name, acc = metric_train.get()
        if val_data is not None:
            name, val_acc = test(ctx, val_data)
            logging.info('[Epoch %d] train=%f val=%f loss=%f time: %f' % 
                (epoch, acc, val_acc, train_loss, time.time()-tic))
        else:
            logging.info('[Epoch %d] train=%f loss=%f time: %f' % 
                (epoch, acc, train_loss, time.time()-tic))

        if save_frequency and save_dir and (epoch + 1) % save_frequency == 0:
            net.save_params('%s/cifar10-%s-%d.params'%(save_dir, model_name, epoch))

    if save_frequency and save_dir:
        net.save_params('%s/cifar10-%s-%d.params'%(save_dir, model_name, epochs-1))

def main():
    if opt.mode == 'hybrid':
        net.hybridize()
    train(opt.epochs, opt.mixup, context)

if __name__ == '__main__':
    main()
