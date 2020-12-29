# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os

# disable autotune
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '1'
#os.environ['MXNET_GPU_MEM_POOL_TYPE'] = 'Round'
os.environ['MXNET_GPU_MEM_POOL_ROUND_LINEAR_CUTOFF'] = '26'
os.environ['MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD'] = '999'
os.environ['MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD'] = '25'
os.environ['MXNET_GPU_COPY_NTHREADS'] = '1'
os.environ['MXNET_OPTIMIZER_AGGREGATION_SIZE'] = '54'

import argparse
import logging
import math
import time
import random
import gluoncv as gcv
gcv.utils.check_version('0.6.0')

from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs#, LRSequential, LRScheduler
import horovod.mxnet as hvd
import mxnet as mx
import numpy as np
from mxnet import autograd, gluon, lr_scheduler
from mxnet.io import DataBatch, DataIter

from mxnet.gluon.data.vision import transforms

from PIL import Image

try:
    from mpi4py import MPI
except ImportError:
    logging.info('mpi4py is not installed. Use "pip install --no-cache mpi4py" to install')
    MPI = None

# Training settings
parser = argparse.ArgumentParser(description='MXNet ImageNet Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--use-rec', action='store_true', default=False,
                    help='use image record iter for data input (default: False)')
parser.add_argument('--data-nthreads', type=int, default=8,
                    help='number of threads for data decoding (default: 2)')
parser.add_argument('--rec-train', type=str, default='',
                    help='the training data')
parser.add_argument('--rec-val', type=str, default='',
                    help='the validation data')
parser.add_argument('--batch-size', type=int, default=128,
                    help='training batch size per device (default: 128)')
parser.add_argument('--dtype', type=str, default='float32',
                    help='data type for training (default: float32)')
parser.add_argument('--num-epochs', type=int, default=90,
                    help='number of training epochs (default: 90)')
parser.add_argument('--lr', type=float, default=0.05,
                    help='learning rate for a single GPU (default: 0.05)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum value for optimizer (default: 0.9)')
parser.add_argument('--wd', type=float, default=0.0001,
                    help='weight decay rate (default: 0.0001)')
parser.add_argument('--warmup-lr', type=float, default=0.0,
                    help='starting warmup learning rate (default: 0.0)')
parser.add_argument('--warmup-epochs', type=int, default=10,
                    help='number of warmup epochs (default: 10)')
parser.add_argument('--last-gamma', action='store_true', default=False,
                    help='whether to init gamma of the last BN layer in \
                    each bottleneck to 0 (default: False)')
parser.add_argument('--mixup', action='store_true',
                    help='whether train the model with mix-up. default is false.')
parser.add_argument('--mixup-alpha', type=float, default=0.2,
                    help='beta distribution parameter for mixup sampling, default is 0.2.')
parser.add_argument('--mixup-off-epoch', type=int, default=0,
                    help='how many last epochs to train without mixup, default is 0.')
parser.add_argument('--label-smoothing', action='store_true',
                    help='use label smoothing or not in training. default is false.')
parser.add_argument('--no-wd', action='store_true',
                    help='whether to remove weight decay on bias, and beta/gamma for batchnorm layers.')
parser.add_argument('--teacher', type=str, default=None,
                    help='teacher model for distillation training')
parser.add_argument('--temperature', type=float, default=20,
                    help='temperature parameter for distillation teacher model')
parser.add_argument('--hard-weight', type=float, default=0.5,
                    help='weight for the loss of one-hot label for distillation training')

parser.add_argument('--model', type=str, default='resnet50_v1',
                    help='type of model to use. see vision_model for options.')
parser.add_argument('--use-pretrained', action='store_true', default=False,
                    help='load pretrained model weights (default: False)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training (default: False)')
parser.add_argument('--eval-frequency', type=int, default=0,
                    help='frequency of evaluating validation accuracy \
                    when training with gluon mode (default: 0)')
parser.add_argument('--log-interval', type=int, default=40,
                    help='number of batches to wait before logging (default: 40)')
parser.add_argument('--save-frequency', type=int, default=20,
                    help='frequency of model saving (default: 0)')
parser.add_argument('--save-dir', type=str, default='params',
                    help='directory of saved models')
# data
parser.add_argument('--input-size', type=int, default=224,
                    help='size of the input image size. default is 224')
parser.add_argument('--crop-ratio', type=float, default=0.875,
                    help='Crop ratio during validation. default is 0.875')
# resume
parser.add_argument('--resume-epoch', type=int, default=0,
                    help='epoch to resume training from.')
parser.add_argument('--resume-params', type=str, default='',
                    help='path of parameters to load from.')
parser.add_argument('--resume-states', type=str, default='',
                    help='path of trainer state to load from.')
# new tricks
parser.add_argument('--dropblock-prob', type=float, default=0,
                    help='DropBlock prob. default is 0.')
parser.add_argument('--use_sk', action='store_true',
                    help='use SK layers or not in resnext. default is false.')
parser.add_argument('--auto_aug', action='store_true',
                    help='use auto_aug. default is false.')
parser.add_argument('--use_avd', action='store_true',
                    help='use avd. default is false.')

args = parser.parse_args()

# Horovod: initialize Horovod
hvd.init()
num_workers = hvd.size()
rank = hvd.rank()
local_rank = hvd.local_rank()

if rank==0:
    logging.basicConfig(level=logging.INFO)
    logging.info(args)

num_classes = 1000
num_training_samples = 1281167
batch_size = args.batch_size
epoch_size = \
    int(math.ceil(int(num_training_samples // num_workers) / batch_size))


lr_sched = lr_scheduler.CosineScheduler(
    args.num_epochs * epoch_size,
    base_lr=(args.lr * num_workers),
    warmup_steps=(args.warmup_epochs * epoch_size),
    warmup_begin_lr=args.warmup_lr
)


class SplitSampler(mx.gluon.data.sampler.Sampler):
    """ Split the dataset into `num_parts` parts and sample from the part with
    index `part_index`
 
    Parameters
    ----------
    length: int
      Number of examples in the dataset
    num_parts: int
      Partition the data into multiple parts
    part_index: int
      The index of the part to read from
    """
    def __init__(self, length, num_parts=1, part_index=0, random=True):
        # Compute the length of each partition
        self.part_len = length // num_parts
        # Compute the start index for this partition
        self.start = self.part_len * part_index
        # Compute the end index for this partition
        self.end = self.start + self.part_len
        self.random = random
 
    def __iter__(self):
        # Extract examples between `start` and `end`, shuffle and return them.
        indices = list(range(self.start, self.end))
        if self.random:
            random.shuffle(indices)
        return iter(indices)
 
    def __len__(self):
        return self.part_len

def get_train_data(rec_train, batch_size, data_nthreads, input_size, crop_ratio, args):
    def train_batch_fn(batch, ctx):
        data = batch[0].as_in_context(ctx)
        label = batch[1].as_in_context(ctx)
        return data, label

    jitter_param = 0.4
    lighting_param = 0.1
    resize = int(math.ceil(input_size / crop_ratio))

    train_transforms = []
    if args.auto_aug:
        print('Using AutoAugment')
        from autogluon.utils.augment import AugmentationBlock, autoaug_imagenet_policies
        train_transforms.append(AugmentationBlock(autoaug_imagenet_policies()))

    from gluoncv.utils.transforms import EfficientNetRandomCrop
    from autogluon.utils import pil_transforms

    if input_size >= 320:
        train_transforms.extend([
            EfficientNetRandomCrop(input_size),
            pil_transforms.Resize((input_size, input_size), interpolation=Image.BICUBIC),
            pil_transforms.RandomHorizontalFlip(),
            pil_transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomLighting(lighting_param),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        train_transforms.extend([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomFlipLeftRight(),
            transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param,
                                         saturation=jitter_param),
            transforms.RandomLighting(lighting_param),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    transform_train = transforms.Compose(train_transforms)

    train_set = mx.gluon.data.vision.ImageRecordDataset(rec_train).transform_first(transform_train)
    train_sampler = SplitSampler(len(train_set), num_parts=num_workers, part_index=rank)

    train_data = gluon.data.DataLoader(train_set, batch_size=batch_size,# shuffle=True,
                                       last_batch='discard', num_workers=data_nthreads,
                                       sampler=train_sampler)
    return train_data, train_batch_fn


def get_val_data(rec_val, batch_size, data_nthreads, input_size, crop_ratio):
    def val_batch_fn(batch, ctx):
        data = batch[0].as_in_context(ctx)
        label = batch[1].as_in_context(ctx)
        return data, label

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    crop_ratio = crop_ratio if crop_ratio > 0 else 0.875
    resize = int(math.ceil(input_size/crop_ratio))

    from gluoncv.utils.transforms import EfficientNetCenterCrop
    from autogluon.utils import pil_transforms

    if input_size >= 320:
        transform_test = transforms.Compose([
            pil_transforms.ToPIL(),
            EfficientNetCenterCrop(input_size),
            pil_transforms.Resize((input_size, input_size), interpolation=Image.BICUBIC),
            pil_transforms.ToNDArray(),
            transforms.ToTensor(),
            normalize
        ])
    else:
        transform_test = transforms.Compose([
            transforms.Resize(resize, keep_ratio=True),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize
        ])

    val_set = mx.gluon.data.vision.ImageRecordDataset(rec_val).transform_first(transform_test)

    val_sampler = SplitSampler(len(val_set), num_parts=num_workers, part_index=rank)
    val_data = gluon.data.DataLoader(val_set, batch_size=batch_size,
                                     num_workers=data_nthreads,
                                     sampler=val_sampler)

    return val_data, val_batch_fn

# Horovod: pin GPU to local rank
context = mx.cpu(local_rank) if args.no_cuda else mx.gpu(local_rank)

#def get_train_data(rec_train, batch_size, data_nthreads, input_size, crop_ratio, args):
train_data, train_batch_fn = get_train_data(args.rec_train, batch_size, args.data_nthreads,
                                            args.input_size, args.crop_ratio, args)
val_data, val_batch_fn = get_val_data(args.rec_val, batch_size, args.data_nthreads, args.input_size,
                                      args.crop_ratio)

# Get model from GluonCV model zoo
# https://gluon-cv.mxnet.io/model_zoo/index.html
kwargs = {'ctx': context,
          'pretrained': args.use_pretrained,
          'classes': num_classes,
          'input_size': args.input_size}

if args.last_gamma:
    kwargs['last_gamma'] = True

if args.dropblock_prob > 0:
        kwargs['dropblock_prob'] = args.dropblock_prob

if args.use_sk:
    kwargs['use_sk'] = True

if args.use_avd:
    kwargs['avd'] = True

net = get_model(args.model, **kwargs)
net.cast(args.dtype)

from gluoncv.nn.dropblock import DropBlockScheduler
# does not impact normal model
drop_scheduler = DropBlockScheduler(net, 0, 0.1, args.num_epochs)

if rank==0:
    logging.info(net)

# Create initializer
initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2)

def train_gluon():
    if args.save_dir:
        save_dir = args.save_dir
        save_dir = os.path.expanduser(save_dir)
        makedirs(save_dir)
    else:
        save_dir = './'
        save_frequency = 0

    def evaluate(epoch):
        acc_top1 = mx.metric.Accuracy()
        acc_top5 = mx.metric.TopKAccuracy(5)
        for _, batch in enumerate(val_data):
            data, label = val_batch_fn(batch, context)
            output = net(data.astype(args.dtype, copy=False))
            acc_top1.update([label], [output])
            acc_top5.update([label], [output])

        top1_name, top1_acc = acc_top1.get()
        top5_name, top5_acc = acc_top5.get()
        if MPI is not None:
            comm = MPI.COMM_WORLD
            res1 = comm.gather(top1_acc, root=0)
            res2 = comm.gather(top5_acc, root=0)
        if rank==0:
            if MPI is not None:
                #logging.info('MPI gather res1: {}'.format(res1))
                top1_acc = sum(res1) / len(res1)
                top5_acc = sum(res2) / len(res2)
            logging.info('Epoch[%d] Rank[%d]\tValidation-%s=%f\tValidation-%s=%f',
                         epoch, rank, top1_name, top1_acc, top5_name, top5_acc)

    # Hybridize and initialize model
    net.hybridize()
    #net.initialize(initializer, ctx=context)
    if args.resume_params != '':
        net.load_parameters(args.resume_params, ctx = context)

    else:
        net.initialize(initializer, ctx=context)

    if args.no_wd:
        for k, v in net.collect_params('.*beta|.*gamma|.*bias').items():
            v.wd_mult = 0.0

    # Horovod: fetch and broadcast parameters
    params = net.collect_params()
    if params is not None:
        hvd.broadcast_parameters(params, root_rank=0)

    # Create optimizer
    optimizer = 'nag'
    optimizer_params = {'wd': args.wd,
                        'momentum': args.momentum,
                        'lr_scheduler': lr_sched}
    if args.dtype == 'float16':
        optimizer_params['multi_precision'] = True
    opt = mx.optimizer.create(optimizer, **optimizer_params)

    # Horovod: create DistributedTrainer, a subclass of gluon.Trainer
    trainer = hvd.DistributedTrainer(params, opt)
    if args.resume_states != '':
        trainer.load_states(args.resume_states)

    # Create loss function and train metric
    if args.label_smoothing or args.mixup:
        sparse_label_loss = False
    else:
        sparse_label_loss = True

    distillation = args.teacher is not None and args.hard_weight < 1.0
    if distillation:
        teacher = get_model(args.teacher, pretrained=True, classes=num_classes, ctx=context)
        teacher.hybridize()
        teacher.cast(args.dtype)
        loss_fn = gcv.loss.DistillationSoftmaxCrossEntropyLoss(temperature=args.temperature,
                                                               hard_weight=args.hard_weight,
                                                               sparse_label=sparse_label_loss)
        if rank == 0:
            logging.info('Using Distillation')
    else:
        loss_fn = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=sparse_label_loss)
    if args.mixup:
        train_metric = mx.metric.RMSE()
    else:
        train_metric = mx.metric.Accuracy()

    def mixup_transform(label, classes, lam=1, eta=0.0):
        if isinstance(label, mx.nd.NDArray):
            label = [label]
        res = []
        for l in label:
            y1 = l.one_hot(classes, on_value = 1 - eta + eta/classes, off_value = eta/classes)
            y2 = l[::-1].one_hot(classes, on_value = 1 - eta + eta/classes, off_value = eta/classes)
            res.append(lam*y1 + (1-lam)*y2)
        return res

    def smooth(label, classes, eta=0.1):
        if isinstance(label, mx.NDArray):
            label = [label]
        smoothed = []
        for l in label:
            res = l.one_hot(classes, on_value = 1 - eta + eta/classes, off_value = eta/classes)
            smoothed.append(res)
        return smoothed

    # Train model
    for epoch in range(args.resume_epoch, args.num_epochs):
        drop_scheduler(epoch)
        tic = time.time()
        train_metric.reset()

        btic = time.time()
        for nbatch, batch in enumerate(train_data, start=1):
            data, label = train_batch_fn(batch, context)
            data, label = [data], [label]
            if args.mixup:
                lam = np.random.beta(args.mixup_alpha, args.mixup_alpha)
                if epoch >= args.num_epochs - args.mixup_off_epoch:
                    lam = 1
                data = [lam*X + (1-lam)*X[::-1] for X in data]

                if args.label_smoothing:
                    eta = 0.1
                else:
                    eta = 0.0
                label = mixup_transform(label, num_classes, lam, eta)

            elif args.label_smoothing:
                hard_label = label
                label = smooth(label, num_classes)

            if distillation:
                teacher_prob = [mx.nd.softmax(teacher(X.astype(args.dtype, copy=False)) / args.temperature) \
                                for X in data]

            with autograd.record():
                outputs = [net(X.astype(args.dtype, copy=False)) for X in data]
                if distillation:
                    loss = [loss_fn(yhat.astype('float32', copy=False),
                            y.astype('float32', copy=False),
                            p.astype('float32', copy=False)) for yhat, y, p in zip(outputs, label, teacher_prob)]
                else:
                    loss = [loss_fn(yhat, y.astype(args.dtype, copy=False)) for yhat, y in zip(outputs, label)]
            for l in loss:
                l.backward()
            trainer.step(batch_size)

            if args.mixup:
                output_softmax = [mx.nd.SoftmaxActivation(out.astype('float32', copy=False)) \
                                  for out in outputs]
                train_metric.update(label, output_softmax)
            else:
                if args.label_smoothing:
                    train_metric.update(hard_label, outputs)
                else:
                    train_metric.update(label, outputs)

            if args.log_interval and nbatch % args.log_interval == 0:
                if rank == 0:
                    logging.info('Epoch[%d] Batch[%d] Loss[%.3f]', epoch, nbatch,
                                 loss[0].mean().asnumpy()[0])

                    train_metric_name, train_metric_score = train_metric.get()
                    logging.info('Epoch[%d] Rank[%d] Batch[%d]\t%s=%f\tlr=%f',
                                 epoch, rank, nbatch, train_metric_name, train_metric_score, trainer.learning_rate)
                    #batch_speed = num_workers * batch_size * args.log_interval / (time.time() - btic)
                    #logging.info('Epoch[%d] Batch[%d]\tSpeed: %.2f samples/sec',
                    #             epoch, nbatch, batch_speed)
                btic = time.time()

        # Report metrics
        elapsed = time.time() - tic
        _, acc = train_metric.get()
        if rank == 0:
            logging.info('Epoch[%d] Rank[%d] Batch[%d]\tTime cost=%.2f\tTrain-metric=%f',
                         epoch, rank, nbatch, elapsed, acc)
            epoch_speed = num_workers * batch_size * nbatch / elapsed
            logging.info('Epoch[%d]\tSpeed: %.2f samples/sec', epoch, epoch_speed)

        # Evaluate performance
        if args.eval_frequency and (epoch + 1) % args.eval_frequency == 0:
            evaluate(epoch)

        # Save model
        if args.save_frequency and (epoch + 1) % args.save_frequency == 0:
            net.save_parameters('%s/imagenet-%s-%d.params'%(save_dir, args.model, epoch))
            trainer.save_states('%s/imagenet-%s-%d.states'%(save_dir, args.model, epoch))

    # Evaluate performance at the end of training
    evaluate(epoch)

    net.save_parameters('%s/imagenet-%s-%d.params'%(save_dir, args.model, args.num_epochs-1))
    trainer.save_states('%s/imagenet-%s-%d.states'%(save_dir, args.model, args.num_epochs-1))

if __name__ == '__main__':
    train_gluon()
