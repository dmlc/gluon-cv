import argparse, time, logging, os

import numpy as np
import mxnet as mx
from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

from gluoncv.data import imagenet
from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, LRScheduler

# CLI
parser = argparse.ArgumentParser(description='Train a model for image classification.')
parser.add_argument('--data-dir', type=str, default='~/.mxnet/datasets/imagenet',
                    help='training and validation pictures to use.')
parser.add_argument('--rec-train', type=str, default='~/.mxnet/datasets/imagenet/rec/train.rec',
                    help='the training data')
parser.add_argument('--rec-train-idx', type=str, default='~/.mxnet/datasets/imagenet/rec/train.idx',
                    help='the index of training data')
parser.add_argument('--rec-val', type=str, default='~/.mxnet/datasets/imagenet/rec/val.rec',
                    help='the validation data')
parser.add_argument('--rec-val-idx', type=str, default='~/.mxnet/datasets/imagenet/rec/val.idx',
                    help='the index of validation data')
parser.add_argument('--use-rec', action='store_true',
                    help='use image record iter for data input. default is false.')
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
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum value for optimizer, default is 0.9.')
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
parser.add_argument('--input-size', type=int, default=224,
                    help='size of the input image size. default is 224')
parser.add_argument('--use-pretrained', action='store_true',
                    help='enable using pretrained model from gluon.')
parser.add_argument('--use_se', action='store_true',
                    help='use SE layers or not in resnext. default is false.')
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
parser.add_argument('--batch-norm', action='store_true',
                    help='enable batch normalization or not in vgg. default is false.')
parser.add_argument('--log-interval', type=int, default=50,
                    help='Number of batches to wait before logging.')
parser.add_argument('--save-frequency', type=int, default=10,
                    help='frequency of model saving.')
parser.add_argument('--save-dir', type=str, default='params',
                    help='directory of saved models')
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
classes = 1000
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

kwargs = {'ctx': context, 'pretrained': opt.use_pretrained, 'classes': classes}
if model_name.startswith('vgg'):
    kwargs['batch_norm'] = opt.batch_norm
elif model_name.startswith('resnext'):
    kwargs['use_se'] = opt.use_se

if opt.last_gamma:
    kwargs['last_gamma'] = True

optimizer = 'nag'
optimizer_params = {'wd': opt.wd, 'momentum': opt.momentum, 'lr_scheduler': lr_scheduler}
if opt.dtype != 'float32':
    optimizer_params['multi_precision'] = True

net = get_model(model_name, **kwargs)
net.cast(opt.dtype)

# Two functions for reading data from record file or raw images
def get_data_rec(rec_train, rec_train_idx, rec_val, rec_val_idx, batch_size, num_workers):
    rec_train = os.path.expanduser(rec_train)
    rec_train_idx = os.path.expanduser(rec_train_idx)
    rec_val = os.path.expanduser(rec_val)
    rec_val_idx = os.path.expanduser(rec_val_idx)
    jitter_param = 0.4
    lighting_param = 0.1
    input_size = opt.input_size
    mean_rgb = [123.68, 116.779, 103.939]
    std_rgb = [58.393, 57.12, 57.375]

    def batch_fn(batch, ctx):
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        return data, label

    train_data = mx.io.ImageRecordIter(
        path_imgrec         = rec_train,
        path_imgidx         = rec_train_idx,
        preprocess_threads  = num_workers,
        shuffle             = True,
        batch_size          = batch_size,

        data_shape          = (3, input_size, input_size),
        mean_r              = mean_rgb[0],
        mean_g              = mean_rgb[1],
        mean_b              = mean_rgb[2],
        std_r               = std_rgb[0],
        std_g               = std_rgb[1],
        std_b               = std_rgb[2],
        rand_mirror         = True,
        random_resized_crop = True,
        max_aspect_ratio    = 4. / 3.,
        min_aspect_ratio    = 3. / 4.,
        max_random_area     = 1,
        min_random_area     = 0.08,
        brightness          = jitter_param,
        saturation          = jitter_param,
        contrast            = jitter_param,
        pca_noise           = lighting_param,
    )
    val_data = mx.io.ImageRecordIter(
        path_imgrec         = rec_val,
        path_imgidx         = rec_val_idx,
        preprocess_threads  = num_workers,
        shuffle             = False,
        batch_size          = batch_size,

        resize              = 256,
        data_shape          = (3, input_size, input_size),
        mean_r              = mean_rgb[0],
        mean_g              = mean_rgb[1],
        mean_b              = mean_rgb[2],
        std_r               = std_rgb[0],
        std_g               = std_rgb[1],
        std_b               = std_rgb[2],
    )
    return train_data, val_data, batch_fn

def get_data_loader(data_dir, batch_size, num_workers):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    jitter_param = 0.4
    lighting_param = 0.1
    input_size = opt.input_size

    def batch_fn(batch, ctx):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        return data, label

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomFlipLeftRight(),
        transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param,
                                     saturation=jitter_param),
        transforms.RandomLighting(lighting_param),
        transforms.ToTensor(),
        normalize
    ])
    transform_test = transforms.Compose([
        transforms.Resize(256, keep_ratio=True),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize
    ])

    train_data = gluon.data.DataLoader(
        imagenet.classification.ImageNet(data_dir, train=True).transform_first(transform_train),
        batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=num_workers)
    val_data = gluon.data.DataLoader(
        imagenet.classification.ImageNet(data_dir, train=False).transform_first(transform_test),
        batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_data, val_data, batch_fn

if opt.use_rec:
    train_data, val_data, batch_fn = get_data_rec(opt.rec_train, opt.rec_train_idx,
                                                  opt.rec_val, opt.rec_val_idx,
                                                  batch_size, num_workers)
else:
    train_data, val_data, batch_fn = get_data_loader(opt.data_dir, batch_size, num_workers)

if opt.mixup:
    train_metric = mx.metric.RMSE()
else:
    train_metric = mx.metric.Accuracy()
acc_top1 = mx.metric.Accuracy()
acc_top5 = mx.metric.TopKAccuracy(5)

save_frequency = opt.save_frequency
if opt.save_dir and save_frequency:
    save_dir = opt.save_dir
    makedirs(save_dir)
else:
    save_dir = ''
    save_frequency = 0

def label_transform(label, classes, eta=0.0):
    ind = label.astype('int')
    res = nd.zeros((ind.shape[0], classes), ctx = label.context)
    res += eta/classes
    res[nd.arange(ind.shape[0], ctx = label.context), ind] = 1 - eta + eta/classes
    return res

def smooth(label, classes, eta=0.1):
    if isinstance(label, nd.NDArray):
        label = [label]
    smoothed = []
    for l in label:
        ind = l.astype('int')
        res = nd.zeros((ind.shape[0], classes), ctx = l.context)
        res += eta/classes
        res[nd.arange(ind.shape[0], ctx = l.context), ind] = 1 - eta + eta/classes
        smoothed.append(res)
    return smoothed

def test(ctx, val_data):
    if opt.use_rec:
        val_data.reset()
    acc_top1.reset()
    acc_top5.reset()
    for i, batch in enumerate(val_data):
        data, label = batch_fn(batch, ctx)
        outputs = [net(X.astype(opt.dtype, copy=False)) for X in data]
        acc_top1.update(label, outputs)
        acc_top5.update(label, outputs)

    _, top1 = acc_top1.get()
    _, top5 = acc_top5.get()
    return (1-top1, 1-top5)

def train(ctx):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    net.initialize(mx.init.MSRAPrelu(), ctx=ctx)

    if opt.no_wd:
        for k, v in net.collect_params('.*beta|.*gamma|.*bias').items():
            v.wd_mult = 0.0

    trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)

    if opt.label_smoothing or opt.mixup:
        L = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False)
    else:
        L = gluon.loss.SoftmaxCrossEntropyLoss()

    best_val_score = 1

    for epoch in range(opt.num_epochs):
        tic = time.time()
        if opt.use_rec:
            train_data.reset()
        train_metric.reset()
        btic = time.time()

        for i, batch in enumerate(train_data):
            data, label = batch_fn(batch, ctx)

            if opt.mixup:
                lam = np.random.beta(opt.mixup_alpha, opt.mixup_alpha)
                if epoch >= opt.num_epochs - opt.mixup_off_epoch:
                    lam = 1
                data_mixup = [lam*X + (1-lam)*X[::-1] for X in data]

                label_mixup = []
                if opt.label_smoothing:
                    eta = 0.1
                else:
                    eta = 0.0
                for Y in label:
                    y1 = label_transform(Y, classes, eta)
                    y2 = label_transform(Y[::-1], classes, eta)
                    label_mixup.append(lam*y1 + (1-lam)*y2)

                data = data_mixup
                label = label_mixup
            elif opt.label_smoothing:
                label = smooth(label, classes)

            with ag.record():
                outputs = [net(X.astype(opt.dtype, copy=False)) for X in data]
                loss = [L(yhat, y.astype(opt.dtype, copy=False)) for yhat, y in zip(outputs, label)]
            for l in loss:
                l.backward()
            lr_scheduler.update(i, epoch)
            trainer.step(batch_size)

            if opt.mixup:
                output_softmax = [nd.SoftmaxActivation(out.astype('float32', copy=False)) \
                                  for out in outputs]
                train_metric.update(label, output_softmax)
            else:
                train_metric.update(label, outputs)

            if opt.log_interval and not (i+1)%opt.log_interval:
                train_metric_name, train_metric_score = train_metric.get()
                logger.info('Epoch[%d] Batch [%d]\tSpeed: %f samples/sec\t%s=%f\tlr=%f'%(
                             epoch, i, batch_size*opt.log_interval/(time.time()-btic),
                             train_metric_name, train_metric_score, trainer.learning_rate))
                btic = time.time()

        train_metric_name, train_metric_score = train_metric.get()
        throughput = int(batch_size * i /(time.time() - tic))

        err_top1_val, err_top5_val = test(ctx, val_data)

        logger.info('[Epoch %d] training: %s=%f'%(epoch, train_metric_name, train_metric_score))
        logger.info('[Epoch %d] speed: %d samples/sec\ttime cost: %f'%(epoch, throughput, time.time()-tic))
        logger.info('[Epoch %d] validation: err-top1=%f err-top5=%f'%(epoch, err_top1_val, err_top5_val))

        if err_top1_val < best_val_score and epoch > 50:
            best_val_score = err_top1_val
            net.save_parameters('%s/%.4f-imagenet-%s-%d-best.params'%(save_dir, best_val_score, model_name, epoch))

        if save_frequency and save_dir and (epoch + 1) % save_frequency == 0:
            net.save_parameters('%s/imagenet-%s-%d.params'%(save_dir, model_name, epoch))

    if save_frequency and save_dir:
        net.save_parameters('%s/imagenet-%s-%d.params'%(save_dir, model_name, opt.num_epochs-1))

def main():
    if opt.mode == 'hybrid':
        net.hybridize(static_alloc=True, static_shape=True)
    train(context)

if __name__ == '__main__':
    main()
