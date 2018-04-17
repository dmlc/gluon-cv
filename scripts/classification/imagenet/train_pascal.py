"""Title
========

"""
import mxnet as mx
import numpy as np
import os, time, logging, math, argparse

from mxnet import gluon, image, init, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision as models

def parse_args():
    parser = argparse.ArgumentParser(description='Gluon for FashionAI Competition',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', required=True, type=str,
                        help='name of the pretrained model from model zoo.')
    parser.add_argument('-j', '--workers', dest='num_workers', default=4, type=int,
                        help='number of preprocessing workers')
    parser.add_argument('--num-gpus', default=0, type=int,
                        help='number of gpus to use, 0 indicates cpu only')
    parser.add_argument('--epochs', default=40, type=int,
                        help='number of training epochs')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        help='mini-batch size')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', dest='wd', default=1e-4, type=float,
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--lr-factor', default=0.75, type=float,
                        help='learning rate decay ratio')
    parser.add_argument('--lr-steps', default='10,20,30', type=str,
                        help='list of learning rate decay epochs as in str')
    args = parser.parse_args()
    return args

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

def train():
    # Initialize the net with pretrained model
    finetune_net = gluon.model_zoo.vision.get_model(model_name, pretrained=True)
    with finetune_net.name_scope():
        finetune_net.output = nn.Dense(classes)
    finetune_net.output.initialize(init.Xavier(), ctx = ctx)
    finetune_net.collect_params().reset_ctx(ctx)
    finetune_net.hybridize()

    # Define DataLoader
    train_data = gluon.data.DataLoader(
        gluon.data.vision.ImageFolderDataset(
            os.path.join('data/train_valid', task, 'train'),
            transform=transform_train),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, last_batch='discard')

    val_data = gluon.data.DataLoader(
        gluon.data.vision.ImageFolderDataset(
            os.path.join('data/train_valid', task, 'val'),
            transform=transform_val),
        batch_size=batch_size, shuffle=False, num_workers = num_workers)

    # Define Trainer
    trainer = gluon.Trainer(finetune_net.collect_params(), 'sgd', {
        'learning_rate': lr, 'momentum': momentum, 'wd': wd})
    metric = mx.metric.Accuracy()
    L = gluon.loss.SoftmaxCrossEntropyLoss()
    lr_counter = 0
    num_batch = len(train_data)

    # Start Training
    for epoch in range(epochs):
        if epoch == lr_steps[lr_counter]:
            trainer.set_learning_rate(trainer.learning_rate*lr_factor)
            lr_counter += 1

        tic = time.time()
        train_loss = 0
        metric.reset()

        for i, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
            with ag.record():
                outputs = [finetune_net(X) for X in data]
                loss = [L(yhat, y) for yhat, y in zip(outputs, label)]
            for l in loss:
                l.backward()

            trainer.step(batch_size)
            train_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)

            metric.update(label, outputs)

            progressbar(i, num_batch-1)

        _, train_acc = metric.get()
        train_loss /= num_batch

        val_acc, val_map, val_loss = validate(finetune_net, val_data, ctx)

        logging.info('[Epoch %d] Train-acc: %.3f, mAP: %.3f, loss: %.3f | Val-acc: %.3f, mAP: %.3f, loss: %.3f | time: %.1f' %
                 (epoch, train_acc, train_map, train_loss, val_acc, val_map, val_loss, time.time() - tic))

    logging.info('\n')
    return (finetune_net)

# Preparation
args = parse_args()

classes = 256

model_name = args.model

epochs = args.epochs
lr = args.lr
batch_size = args.batch_size
momentum = args.momentum
wd = args.wd

lr_factor = args.lr_factor
lr_steps = [int(s) for s in args.lr_steps.split(',')] + [np.inf]

num_gpus = args.num_gpus
num_workers = args.num_workers
ctx = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
batch_size = batch_size * max(num_gpus, 1)

logging.basicConfig(level=logging.INFO,
                    handlers = logging.StreamHandler())

if __name__ == "__main__":
    net = train()

