import mxnet as mx
import numpy as np
import os, time, logging, argparse, shutil
import gluoncv as gcv
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision as models
from mxnet.gluon.data.vision import transforms
from gluoncv.utils import makedirs
from mxnet import gluon, init, nd
import sys
sys.path.append('..')
from ssd.train_ssd import train as ssdtrain
from faster_rcnn.train_faster_rcnn import train as faster_rcnn_train
from yolo.train_yolo3 import train as yolotrain
from data_utils import get_dataset, ssdget_dataloader, yolo3_get_dataloader, faster_rcnn_get_dataloader


def parse_opts():
    parser = argparse.ArgumentParser(description='Finetune SOTA detection method',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, default='model',
                        help="Base network name which serves as feature extraction base.")
    parser.add_argument('--data-shape', type=int, default=300,
                        help="Input data shape, use 300, 512.")
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training mini-batch size')
    parser.add_argument('--dataset', type=str, default='voc',
                        help='Training dataset. Now support voc.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=4, help='Number of data workers, you can use larger '
                        'number to accelerate data loading, if you CPU and GPUs are powerful.')
    parser.add_argument('--gpus', type=str, default='3',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--epochs', type=int, default=240,
                        help='Training epochs.')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from previously saved parameters if not None. '
                        'For example, you can resume from ./ssd_xxx_0123.params')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Starting epoch for resuming, default is 0 for new training.'
                        'You can specify it to 100 for example to start from 100 epoch.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate, default is 0.001')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-epoch', type=str, default='160,200',
                        help='epoches at which learning rate decays. default is 160,200.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum, default is 0.9')
    parser.add_argument('--wd', type=float, default=0.0005,
                        help='Weight decay, default is 5e-4')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Logging mini-batch interval. Default is 100.')
    parser.add_argument('--save-prefix', type=str, default='',
                        help='Saving parameter prefix')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Saving parameters epoch interval, best model will always be saved.')
    parser.add_argument('--val-interval', type=int, default=1,
                        help='Epoch interval for validation, increase the number will reduce the '
                             'training time if validation is slow.')
    parser.add_argument('--seed', type=int, default=233,
                        help='Random seed to be fixed.')
    parser.add_argument('--classes', type=str, help='finetune-classes')
    opts = parser.parse_args()
    return opts


opts=parse_opts()
lr=opts.lr
epoch = opts.epochs
momentum=opts.momentum
gpus = opts.gpus
ctx = mx.gpu(gpus)
batch_size = opts.batch_size
if opts.model.lower() == 'ssd':
    ### Load the pretrained SSD model ###
    net = gcv.model_zoo.get_model('ssd_300_vgg16_atrous_coco', pretrained=True)
    ### Finetune the model ###
    net.reset_class(opts.classes)
    net.collect_params().reset_ctx(ctx)
    ### Get the dataset ###
    train_dataset, val_dataset, eval_metric = get_dataset(opts.dataset, opts)
    train_data, val_data = ssdget_dataloader(
        net, train_dataset, val_dataset, opts.data_shape, opts.batch_size, opts.num_workers)
    ssdtrain(net, train_data, val_data, eval_metric, ctx, opts)
if opts.model.lower() == 'faster_rcnn':
    ### Load the pretrained faster-rcnn model ###
    net = gcv.model_zoo.get_model('faster_rcnn_resnet50_v1b_coco', pretrained=True)
    ### Finetune the model ###
    net.reset_class(opts.classes)
    ### Get the dataset ###
    train_dataset, val_dataset, eval_metric = get_dataset(opts.dataset, opts)
    train_data, val_data = faster_rcnn_get_dataloader(
        net, train_dataset, val_dataset, opts.batch_size, opts.num_workers)
    faster_rcnn_train(net, train_data, val_data, eval_metric, ctx, opts)
if opts.model.lower() == 'yolo':
    ### Load the pretrained yolo model ###
    net = gcv.model_zoo.get_model('yolo3_darknet53_coco', pretrained=True)
    ### Finetune the model ###
    net.reset_class(opts.classes)
    ### Get the dataset ###
    train_dataset, val_dataset, eval_metric = get_dataset(opts.dataset, opts)
    train_data, val_data = yolo3_get_dataloader(
        net, train_dataset, val_dataset, opts.data_shape, opts.batch_size, opts.num_workers)
    yolotrain(net, train_data, val_data, eval_metric, ctx, opts)








































