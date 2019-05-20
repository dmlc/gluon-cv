from __future__ import division

import argparse, datetime, os
import logging
import my_logging

import __init
import mxnet as mx
from mxnet import gluon
from mxnet.gluon.data.vision import transforms
from mxnet import autograd

from networks import resnet50
from networks.resnet import ResNet

from mygluoncv.data.market1501.data_read import ImageTxtDataset
from mygluoncv.data.market1501.label_read import LabelList
from mygluoncv.data.transforms.block import RandomCrop


# CLI
parser = argparse.ArgumentParser(description='Train a model for image classification.')
parser.add_argument('--img-height', type=int, default=384,
                    help='the height of image for input')
parser.add_argument('--img-width', type=int, default=128,
                    help='the width of image for input')
parser.add_argument('--batch-size', type=int, default=32,
                    help='training batch size per device (CPU/GPU).')
parser.add_argument('--num-workers', type=int, default=8,
                    help='the number of workers for data loader')
parser.add_argument('--kvstore', type=str, default='device',
                    help='kvstore to use for trainer/module.')
parser.add_argument('--dataset-root', type=str,
                    default="../../datasets/train_part",
                    help='the number of workers for data loader')
parser.add_argument('--dataset', type=str, default="reid_all_dataset",
                    help='the number of workers for data loader')
parser.add_argument('--train-txt', type=str, default="train.xtx",
                    help='the train txt file')
parser.add_argument('--num-gpus', type=int, default=1,
                    help='number of gpus to use.')
parser.add_argument('--classes', type=int, default=751,
                    help='the classes of the datasets.')
parser.add_argument('--warmup', type=bool, default=True,
                    help='number of training epochs.')
parser.add_argument('--epochs', type=str, default="5,25,50,75")
parser.add_argument('--ratio', type=float, default=1.,
                    help="ratio of training set to all set")
parser.add_argument('--pad', type=int, default=10)
parser.add_argument('--lr', type=float, default=3.5e-4,
                    help='learning rate. default is 0.1.')
parser.add_argument('-momentum', type=float, default=0.9,
                    help='momentum value for optimizer, default is 0.9.')
parser.add_argument('--wd', type=float, default=5e-4,
                    help='weight decay rate. default is 5e-4.')
parser.add_argument('--seed', type=int, default=613,
                    help='random seed to use. Default=613.')
parser.add_argument('--lr-decay', type=int, default=0.1)
parser.add_argument('--hybridize', type=bool, default=True)


def get_data_iters(batch_size):
    train_set, val_set = LabelList(ratio=opt.ratio, root=opt.dataset_root,
                                   name=opt.dataset, train_file=opt.train_txt)
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
        transforms.Resize(size=(opt.img_width, opt.img_height), interpolation=1),
        transforms.RandomFlipLeftRight(),
        RandomCrop(size=(opt.img_width, opt.img_height), pad=opt.pad),
        transforms.ToTensor(),
        normalizer])

    train_imgs = ImageTxtDataset(train_set, transform=transform_train)
    train_data = gluon.data.DataLoader(train_imgs, batch_size, shuffle=True, last_batch='discard', num_workers=opt.num_workers)

    if opt.ratio < 1:
        transform_test = transforms.Compose([
            transforms.Resize(size=(opt.img_width, opt.img_height), interpolation=1),
            transforms.ToTensor(),
            normalizer])

        val_imgs = ImageTxtDataset(val_set, transform=transform_test)
        val_data = gluon.data.DataLoader(val_imgs, batch_size, shuffle=True, last_batch='discard', num_workers=opt.num_workers)
    else:
        val_data = None

    return train_data, val_data


def validate(val_data, net, criterion, ctx):
    loss = 0.0
    for data, label in val_data:
        data_list = gluon.utils.split_and_load(data, ctx)
        label_list = gluon.utils.split_and_load(label, ctx)

        with autograd.predict_mode():
            outpus = [net(X) for X in data_list]
            losses = [criterion(X, y) for X, y in zip(outpus, label_list)]
        accuracy = [(X.argmax(axis=1)==y.astype('float32')).mean.asscalar() for X, y in zip(outpus, label_list)]

        loss_list = [l.mean().asscalar() for l in losses]
        loss += sum(loss_list) / len(loss_list)

    return loss/len(val_data), sum(accuracy)/len(accuracy)


def get_my_ip():
    cmd = "ifconfig | grep 'inet addr:' | grep -v '127.0.0.1' | cut -d: -f2 | awk '{print $1}' | head -1"
    ip = os.popen(cmd).read().strip()
    return ip


def main(net: ResNet, batch_size, epochs, opt, ctx):
    tag_logger = logging.getLogger('tag')
    my_ip = get_my_ip()
    extra = {'tag': my_ip}
    train_data, val_data = get_data_iters(batch_size)
    if opt.hybridize:
        net.hybridize()
    kv = mx.kv.create(opt.kvstore)
    trainer = gluon.Trainer(net.collect_params(), kvstore=kv, optimizer='adam',
                            optimizer_params={'learning_rate': opt.lr, 'wd': opt.wd})
    criterion = gluon.loss.SoftmaxCrossEntropyLoss()

    lr = opt.lr
    if opt.warmup:
        minlr = lr*0.01
        dlr = (lr-minlr)/(epochs[0]-1)

    prev_time = datetime.datetime.now()
    tag_logger.info("start train...", extra=extra)
    for epoch in range(epochs[-1]):
        _loss = 0.
        if opt.warmup:
            if epoch<epochs[0]:
                lr = minlr + dlr*epoch
        if epoch in epochs[1:]:
            lr = lr * opt.lr_decay
        trainer.set_learning_rate(lr)

        n = 0
        for data, label in train_data:
            data_list = gluon.utils.split_and_load(data, ctx)
            label_list = gluon.utils.split_and_load(label, ctx)
            with autograd.record():
                output = [net(X) for X in data_list]
                losses = [criterion(X, y) for X, y in zip(output, label_list)]

            for l in losses:
                l.backward()
            trainer.step(batch_size)
            _loss_list = [l.mean().asscalar() for l in losses]
            _loss += sum(_loss_list) / len(_loss_list)
            n += 1
            if n % 2000 is 0:
                tag_logger.info("current mean loss:{}".format(_loss), extra=extra)

        cur_time = datetime.datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Time %02d:%02d:%02d" % (h, m, s)
        __loss = _loss/len(train_data)

        if val_data is not None:
            val_loss, val_accuracy = validate(val_data, net, criterion, ctx)
            epoch_str = ("Epoch %d. Train loss: %f, Val loss %f, Val accuracy %f, " % (epoch, __loss , val_loss, val_accuracy))
        else:
            epoch_str = ("Epoch %d. Train loss: %f, " % (epoch, __loss))

        prev_time = cur_time
        tag_logger.info(epoch_str + time_str + ', lr ' + str(trainer.learning_rate), extra=extra)

    if not os.path.exists("params"):
        os.mkdir("params")
    net.save_parameters("params/resnet50.params")


if __name__ == '__main__':
    my_logging.init_logging("train2.log")
    opt = parser.parse_args()
    logging.info(opt)
    mx.random.seed(opt.seed)

    batch_size = opt.batch_size
    epochs = [int(i) for i in opt.epochs.split(',')]

    if opt.num_gpus is 0:
        context = [mx.cpu(0)]
    else:
        num_gpus = opt.num_gpus
        batch_size *= max(1, num_gpus)
        context = [mx.gpu(i) for i in range(num_gpus)]

    net = resnet50(ctx=context, num_classes=opt.classes)
    main(net, batch_size, epochs, opt, context)
