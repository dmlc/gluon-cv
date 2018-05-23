import matplotlib
matplotlib.use('Agg')

import argparse, time, logging

import mxnet as mx
import numpy as np
from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

from gluoncv.data import imagenet
from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, TrainingHistory, freeze_bn

# CLI
parser = argparse.ArgumentParser(description='Train a model for image classification.')
parser.add_argument('--data-dir', type=str, default='~/.mxnet/datasets/imagenet',
                    help='training and validation pictures to use.')
parser.add_argument('--dummy', action='store_true',
                    help='use dummy data to test training speed. default is false.')
parser.add_argument('--batch-size', type=int, default=32,
                    help='training batch size per device (CPU/GPU).')
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
parser.add_argument('--use_se', action='store_true',
                    help='use SE layers or not in resnext. default is false.')
parser.add_argument('--label-smoothing', action='store_true',
                    help='use label smoothing or not in training. default is false.')
parser.add_argument('--freeze-bn', action='store_true',
                    help='freeze batchnorm layers in the last frew training epochs or not. default is false.')
parser.add_argument('--batch-norm', action='store_true',
                    help='enable batch normalization or not in vgg. default is false.')
parser.add_argument('--use-pretrained', action='store_true',
                    help='enable using pretrained model from gluon.')
parser.add_argument('--log-interval', type=int, default=50,
                    help='Number of batches to wait before logging.')
parser.add_argument('--save-frequency', type=int, default=10,
                    help='frequency of model saving.')
parser.add_argument('--save-dir', type=str, default='params',
                    help='directory of saved models')
parser.add_argument('--logging-dir', type=str, default='logs',
                    help='directory of training logs')
parser.add_argument('--save-plot-dir', type=str, default='.',
                    help='the path to save the history plot')
opt = parser.parse_args()

logging.basicConfig(level=logging.INFO)
logging.info(opt)

batch_size = opt.batch_size
classes = 1000

num_gpus = opt.num_gpus
batch_size *= max(1, num_gpus)
context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus > 0 else [mx.cpu()]
num_workers = opt.num_workers

lr_decay = opt.lr_decay
lr_decay_period = opt.lr_decay_period
lr_decay_epoch = [int(i) for i in opt.lr_decay_epoch.split(',')] + [np.inf]

model_name = opt.model

kwargs = {'ctx': context, 'pretrained': opt.use_pretrained, 'classes': classes}
if model_name.startswith('vgg'):
    kwargs['batch_norm'] = opt.batch_norm
elif model_name.startswith('resnext'):
    kwargs['use_se'] = opt.use_se

optimizer = 'nag'
optimizer_params = {'learning_rate': opt.lr, 'wd': opt.wd, 'momentum': opt.momentum}

net = get_model(model_name, **kwargs)
if opt.use_se:
    model_name = 'se_' + model_name

acc_top1 = mx.metric.Accuracy()
acc_top5 = mx.metric.TopKAccuracy(5)
train_history = TrainingHistory(['training-top1-err', 'training-top5-err',
                                 'validation-top1-err', 'validation-top5-err'])

save_frequency = opt.save_frequency
if opt.save_dir and save_frequency:
    save_dir = opt.save_dir
    makedirs(save_dir)
else:
    save_dir = ''
    save_frequency = 0

plot_path = opt.save_plot_dir

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
    net.initialize(mx.init.MSRAPrelu(), ctx=ctx)

    train_data = gluon.data.DataLoader(
        imagenet.classification.ImageNet(opt.data_dir, train=True).transform_first(transform_train),
        batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=num_workers)
    val_data = gluon.data.DataLoader(
        imagenet.classification.ImageNet(opt.data_dir, train=False).transform_first(transform_test),
        batch_size=batch_size, shuffle=False, num_workers=num_workers)

    trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)
    if opt.label_smoothing:
        L = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False)
    else:
        L = gluon.loss.SoftmaxCrossEntropyLoss()

    lr_decay_count = 0

    best_val_score = 1

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

        if opt.freeze_bn and epoch == epochs - 10:
            freeze_bn(net, True)

        for i, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            if opt.label_smoothing:
                label_smooth = smooth(label, classes)
            else:
                label_smooth = label
            with ag.record():
                outputs = [net(X) for X in data]
                loss = [L(yhat, y) for yhat, y in zip(outputs, label_smooth)]
            ag.backward(loss)
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
        train_loss /= num_batch * batch_size

        err_top1_val, err_top5_val = test(ctx, val_data)
        train_history.update([err_top1, err_top5, err_top1_val, err_top5_val])
        train_history.plot(['training-top1-err', 'validation-top1-err'],
                           save_path='%s/%s_top1.png'%(plot_path, model_name))
        train_history.plot(['training-top5-err', 'validation-top5-err'],
                           save_path='%s/%s_top5.png'%(plot_path, model_name))

        logging.info('[Epoch %d] training: err-top1=%f err-top5=%f loss=%f'%(epoch, err_top1, err_top5, train_loss))
        logging.info('[Epoch %d] time cost: %f'%(epoch, time.time()-tic))
        logging.info('[Epoch %d] validation: err-top1=%f err-top5=%f'%(epoch, err_top1_val, err_top5_val))

        if err_top1_val < best_val_score and epoch > 50:
            best_val_score = err_top1_val
            net.save_params('%s/%.4f-imagenet-%s-%d-best.params'%(save_dir, best_val_score, model_name, epoch))

        if save_frequency and save_dir and (epoch + 1) % save_frequency == 0:
            net.save_params('%s/imagenet-%s-%d.params'%(save_dir, model_name, epoch))

    if save_frequency and save_dir:
        net.save_params('%s/imagenet-%s-%d.params'%(save_dir, model_name, epochs-1))

def train_dummy(ctx):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    net.initialize(mx.init.MSRAPrelu(), ctx=ctx)

    data = []
    label = []
    bs = batch_size // len(ctx)
    for c in ctx:
        data.append(mx.nd.random.uniform(shape=(bs,3,224,224), ctx = c))
        label.append(mx.nd.ones(shape=(bs), ctx = c))

    trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)
    if opt.label_smoothing:
        L = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False)
    else:
        L = gluon.loss.SoftmaxCrossEntropyLoss()

    acc_top1.reset()
    acc_top5.reset()
    btic = time.time()
    train_loss = 0
    num_batch = 1000
    warm_up = 100

    for i in range(num_batch):
        if i == warm_up:
            tic = time.time()
        if opt.label_smoothing:
            label_smooth = smooth(label, classes)
        else:
            label_smooth = label
        with ag.record():
            outputs = [net(X) for X in data]
            loss = [L(yhat, y) for yhat, y in zip(outputs, label_smooth)]
        ag.backward(loss)
        trainer.step(batch_size)
        acc_top1.update(label, outputs)
        acc_top5.update(label, outputs)
        train_loss += sum([l.sum().asscalar() for l in loss])

        if opt.log_interval and not (i+1)%opt.log_interval:
            logging.info('Batch [%d]\tSpeed: %f samples/sec'%(
                         i, batch_size*opt.log_interval/(time.time()-btic)))
            btic = time.time()

    total_time_cost = time.time()-tic
    logging.info('Test finished. Average Speed: %f samples/sec. Total time cost: %f'%(
                 batch_size*(num_batch-warm_up)/total_time_cost, total_time_cost))

def main():
    if opt.mode == 'hybrid':
        net.hybridize()
    if opt.dummy:
        train_dummy(context)
    else:
        train(opt.num_epochs, context)

if __name__ == '__main__':
    main()
