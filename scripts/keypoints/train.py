import argparse, time, logging, os, math

import numpy as np
import mxnet as mx
from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

from gluoncv.data import mscoco
from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, LRScheduler
from gluoncv.data.transforms.presets.simple_pose import SimplePoseDefaultTrainTransform

# CLI
parser = argparse.ArgumentParser(description='Train a model for image classification.')
parser.add_argument('--data-dir', type=str, default='~/.mxnet/datasets/coco',
                    help='training and validation pictures to use.')
parser.add_argument('--batch-size', type=int, default=32,
                    help='training batch size per device (CPU/GPU).')
parser.add_argument('--dtype', type=str, default='float32',
                    help='data type for training. default is float32')
parser.add_argument('--num-gpus', type=int, default=0,
                    help='number of gpus to use.')
parser.add_argument('-j', '--num-data-workers', dest='num_workers', default=4, type=int,
                    help='number of preprocessing workers')
parser.add_argument('--num-epochs', type=int, default=3,
                    help='number of training epochs.')
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate. default is 0.1.')
parser.add_argument('--wd', type=float, default=0.0001,
                    help='weight decay rate. default is 0.0001.')
parser.add_argument('--lr-mode', type=str, default='step',
                    help='learning rate scheduler mode. options are step, poly and cosine.')
parser.add_argument('--lr-decay', type=float, default=0.1,
                    help='decay rate of learning rate. default is 0.1.')
parser.add_argument('--lr-decay-period', type=int, default=0,
                    help='interval for periodic learning rate decays. default is 0 to disable.')
parser.add_argument('--lr-decay-epoch', type=str, default='40,60',
                    help='epoches at which learning rate decays. default is 40,60.')
parser.add_argument('--warmup-lr', type=float, default=0.0,
                    help='starting warmup learning rate. default is 0.0.')
parser.add_argument('--warmup-epochs', type=int, default=0,
                    help='number of warmup epochs.')
parser.add_argument('--last-gamma', action='store_true',
                    help='whether to init gamma of the last BN layer in each bottleneck to 0.')
parser.add_argument('--mode', type=str,
                    help='mode in which to train the model. options are symbolic, imperative, hybrid')
parser.add_argument('--model', type=str, required=True,
                    help='type of model to use. see vision_model for options.')
parser.add_argument('--input-size', type=str, default='256,192',
                    help='size of the input image size. default is 256,192')
parser.add_argument('--use-pretrained', action='store_true',
                    help='enable using pretrained model from gluon.')
parser.add_argument('--use-pretrained-base', action='store_true',
                    help='enable using pretrained base model from gluon.')
parser.add_argument('--no-wd', action='store_true',
                    help='whether to remove weight decay on bias, and beta/gamma for batchnorm layers.')
parser.add_argument('--save-frequency', type=int, default=10,
                    help='frequency of model saving.')
parser.add_argument('--save-dir', type=str, default='params',
                    help='directory of saved models')
parser.add_argument('--log-interval', type=int, default=50,
                    help='Number of batches to wait before logging.')
parser.add_argument('--logging-file', type=str, default='train_imagenet.log',
                    help='name of training log file')
opt = parser.parse_args()

filehandler = logging.FileHandler(opt.logging_file)
streamhandler = logging.StreamHandler()

logger = logging.getLogger('')
logger.setLevel(logging.INFO)
logger.addHandler(filehandler)
logger.addHandler(streamhandler)

logger.info(opt)

batch_size = opt.batch_size
num_joints = 17
num_training_samples = 1281167

num_gpus = opt.num_gpus
batch_size *= max(1, num_gpus)
context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
num_workers = opt.num_workers


lr_decay = opt.lr_decay
lr_decay_period = opt.lr_decay_period
if opt.lr_decay_period > 0:
    lr_decay_epoch = list(range(lr_decay_period, opt.num_epochs, lr_decay_period))
else:
    lr_decay_epoch = [int(i) for i in opt.lr_decay_epoch.split(',')]
num_batches = num_training_samples // batch_size
lr_scheduler = LRScheduler(mode=opt.lr_mode, baselr=opt.lr,
                           niters=num_batches, nepochs=opt.num_epochs,
                           step=lr_decay_epoch, step_factor=opt.lr_decay, power=2,
                           warmup_epochs=opt.warmup_epochs)

model_name = opt.model

kwargs = {'ctx': context, 'pretrained': opt.use_pretrained, 'num_joints': num_joints}
if model_name.startswith('vgg'):
    kwargs['batch_norm'] = opt.batch_norm
elif model_name.startswith('resnext'):
    kwargs['use_se'] = opt.use_se

if opt.last_gamma:
    kwargs['last_gamma'] = True

optimizer = 'adam'
optimizer_params = {'wd': opt.wd, 'lr_scheduler': lr_scheduler}
if opt.dtype != 'float32':
    optimizer_params['multi_precision'] = True

net = get_model(model_name, **kwargs)
net.cast(opt.dtype)

def get_data_loader(data_dir, batch_size, num_workers, input_size):

    def batch_fn(batch, ctx):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        weight = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)
        return data, label, weight

    dataset = mscoco.keypoints.COCOKeyPoints(data_dir, aspect_ratio=4./3.)
    heatmap_size = [int(i/4) for i in input_size]
    transform_train = SimplePoseDefaultTrainTransform(num_joints=dataset.num_joints,
                                                      joint_pairs=dataset.joint_pairs,
                                                      image_size=input_size, heatmap_size=heatmap_size,
                                                      scale_factor=0.30, rotation_factor=40)

    train_data = gluon.data.DataLoader(
        dataset.transform(transform_train),
        batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=num_workers)

    return train_data, batch_fn

input_size = [int(i) for i in opt.input_size.split(',')]
train_data, batch_fn = get_data_loader(opt.data_dir, batch_size, num_workers, input_size)
train_metric = mx.metric.Accuracy()

save_frequency = opt.save_frequency
if opt.save_dir and save_frequency:
    save_dir = opt.save_dir
    makedirs(save_dir)
else:
    save_dir = ''
    save_frequency = 0

def train(ctx):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    if opt.use_pretrained_base:
        net.deconv_layers.initialize(mx.init.MSRAPrelu(), ctx=ctx)
        net.final_layer.initialize(mx.init.MSRAPrelu(), ctx=ctx)
    else:
        net.initialize(mx.init.MSRAPrelu(), ctx=ctx)

    if opt.no_wd:
        for k, v in net.collect_params('.*beta|.*gamma|.*bias').items():
            v.wd_mult = 0.0

    trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)

    L = gluon.loss.L2Loss()

    best_val_score = 1

    for epoch in range(opt.num_epochs):
        tic = time.time()
        train_metric.reset()
        btic = time.time()

        for i, batch in enumerate(train_data):
            data, label, weight = batch_fn(batch, ctx)

            import pdb; pdb.set_trace()
            with ag.record():
                outputs = [net(X.astype(opt.dtype, copy=False)) for X in data]
                loss = [L(yhat, y.astype(opt.dtype, copy=False), w)
                        for yhat, y, w in zip(outputs, label, weight)]
            for l in loss:
                l.backward()
            lr_scheduler.update(i, epoch)
            trainer.step(batch_size)

            train_metric.update(label, outputs)

            if opt.log_interval and not (i+1)%opt.log_interval:
                train_metric_name, train_metric_score = train_metric.get()
                logger.info('Epoch[%d] Batch [%d]\tSpeed: %f samples/sec\t%s=%f\tlr=%f'%(
                             epoch, i, batch_size*opt.log_interval/(time.time()-btic),
                             train_metric_name, train_metric_score, trainer.learning_rate))
                btic = time.time()

        train_metric_name, train_metric_score = train_metric.get()
        throughput = int(batch_size * i /(time.time() - tic))

        logger.info('[Epoch %d] training: %s=%f'%(epoch, train_metric_name, train_metric_score))

        if save_frequency and save_dir and (epoch + 1) % save_frequency == 0:
            net.save_parameters('%s/imagenet-%s-%d.params'%(save_dir, model_name, epoch))
            trainer.save_states('%s/imagenet-%s-%d.states'%(save_dir, model_name, epoch))

    if save_frequency and save_dir:
        net.save_parameters('%s/imagenet-%s-%d.params'%(save_dir, model_name, opt.num_epochs-1))
        trainer.save_states('%s/imagenet-%s-%d.states'%(save_dir, model_name, opt.num_epochs-1))

def main():
    if opt.mode == 'hybrid':
        net.hybridize(static_alloc=True, static_shape=True)
    train(context)

if __name__ == '__main__':
    main()
