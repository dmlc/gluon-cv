"""Train SSD on Pascal VOC dataset"""
import argparse
import os
import logging
import time
import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
from gluonvision import data as gdata
from gluonvision import utils as gutils
from gluonvision.model_zoo import get_model
from gluonvision.data.transforms.presets.ssd import SSDDefaultTrainTransform
from gluonvision.data.transforms.presets.ssd import SSDDefaultValTransform
from gluonvision.utils.metrics.voc_detection import VOC07MApMetric
from gluonvision.utils.metrics.accuracy import Accuracy

def parse_args():
    parser = argparse.ArgumentParser(description='Train SSD networks.')
    parser.add_argument('--network', type=str, default='vgg16_atrous',
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
    parser.add_argument('--gpus', type=str, default='0',
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
    parser.add_argument('--seed', type=int, default=233,
                        help='Random seed to be fixed.')
    args = parser.parse_args()
    return args

def get_dataset(dataset):
    if dataset.lower() == 'voc':
        train_dataset = gdata.VOCDetection(
            splits=[(2007, 'trainval'), (2012, 'trainval')])
        val_dataset = gdata.VOCDetection(
            splits=[(2007, 'test')])
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    return train_dataset, val_dataset

def get_dataloader(train_dataset, val_dataset, data_shape, batch_size, num_workers):
    """Get dataloader."""
    width, height = data_shape, data_shape
    train_loader = gdata.DetectionDataLoader(
        train_dataset.transform(SSDDefaultTrainTransform(width, height)),
        batch_size, True, last_batch='rollover', num_workers=num_workers)
    val_loader = gdata.DetectionDataLoader(
        val_dataset.transform(SSDDefaultValTransform(width, height)),
        batch_size, False, last_batch='keep', num_workers=num_workers)
    return train_loader, val_loader

def save_params(net, best_map, current_map, epoch, save_interval, prefix):
    if current_map > best_map[0]:
        best_map[0] = current_map
        net.save_params('{:s}_best.params'.format(prefix, epoch, current_map))
        with open(prefix+'_best_map.log', 'a') as f:
            f.write('\n{:04d}:\t{:.4f}'.format(epoch, current_map))
    if save_interval and epoch % save_interval == 0:
        net.save_params('{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map))

def validate(net, val_data, ctx, classes):
    """Test on validation dataset."""
    metric = VOC07MApMetric(iou_thresh=0.5, class_names=classes)
    # set nms threshold and topk constraint
    net.set_nms(nms_thresh=0.45, nms_topk=400)
    net.hybridize()
    for batch in val_data:
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        for x, y in zip(data, label):
            # get prediction results
            ids, scores, bboxes = net(x)
            # clip to image size
            bboxes = bboxes.clip(0, batch[0].shape[2])
            # split ground truths
            gt_ids = y.slice_axis(axis=-1, begin=4, end=5)
            gt_bboxes = y.slice_axis(axis=-1, begin=0, end=4)
            gt_difficults = y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None
            # update metric
            metric.update(bboxes, ids, scores, gt_bboxes, gt_ids, gt_difficults)

    return metric.get()

def train(net, train_data, val_data, classes, args):
    """Training pipeline"""
    for param in net.collect_params().values():
        if param._data is not None:
            continue
        param.initialize()
    net.collect_params().reset_ctx(ctx)
    trainer = gluon.Trainer(
        net.collect_params(), 'sgd',
        {'learning_rate': args.lr, 'wd': args.wd, 'momentum': args.momentum})

    # lr decay policy
    lr_decay = args.lr_decay
    lr_steps = [float(ls) for ls in args.lr_decay_epoch.split(',') if ls.strip()]
    if not isinstance(lr_decay, list):
        lr_decay = [lr_decay]
    if len(lr_decay) == 1 and len(lr_steps) > 1:
        lr_decay *= len(lr_steps)

    cls_loss = gluon.loss.SoftmaxCrossEntropyLoss()
    box_loss = gluon.loss.HuberLoss()
    acc_metric = Accuracy(axis=-1, ignore_labels=[-1])
    ce_metric = mx.metric.Loss('CrossEntropy')
    smoothl1_metric = mx.metric.Loss('SmoothL1')

    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = args.save_prefix + '_train.log'
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    logger.info(args)
    logger.info('Start training from [Epoch %d]' % args.start_epoch)
    best_map = [0]
    for epoch in range(args.start_epoch, args.epochs):
        if epoch in lr_steps:
            new_lr = trainer.learning_rate * np.prod(lr_decay[:lr_steps.index(epoch)+1])
            trainer.set_learning_rate(new_lr)
            logger.info("[Epoch {}] Set learning rate to {}".format(epoch, new_lr))
        acc_metric.reset()
        ce_metric.reset()
        smoothl1_metric.reset()
        tic = time.time()
        btic = time.time()
        net.hybridize()
        for i, batch in enumerate(train_data):
            batch_size = batch[0].shape[0]
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            outputs = []
            labels = []
            losses1 = []
            losses2 = []
            losses3 = []  # temporary cls loss holder
            losses4 = []  # temporary box loss holder
            Ls = []
            num_positive = []
            with autograd.record():
                for x, y in zip(data, label):
                    cls_preds, box_preds, anchors = net(x)
                    with autograd.pause():
                        # we generate training targets here in autograd.pause scope
                        # because we don't need to bp to labels. This can reduce the
                        # overhead of auto differentiation.
                        gt_boxes = nd.slice_axis(y, axis=-1, begin=0, end=4)
                        gt_ids = nd.slice_axis(y, axis=-1, begin=4, end=5)
                        cls_targets, box_targets, box_masks = net.target_generator(
                            anchors, cls_preds, gt_boxes, gt_ids)
                        # save how many positive samples are used, it will be used to
                        # normalize the loss
                        num_positive.append(nd.sum(cls_targets > 0).asscalar())

                    # cls loss, multi class cross entropy loss, we mask out ignored
                    # labels here by broadcast_mul the positive labels
                    l1 = cls_loss(cls_preds, cls_targets, (cls_targets >= 0).expand_dims(axis=-1))
                    losses3.append(l1 * cls_targets.size / cls_targets.shape[0])
                    # box loss, it's a huber loss(or namely smoothl1 loss in paper)
                    l2 = box_loss(box_preds * box_masks, box_targets)
                    losses4.append(l2 * box_targets.size / box_targets.shape[0])
                    # some records for metrics
                    outputs.append(cls_preds)
                    labels.append(cls_targets)
                # n_pos is the overall positive samples in the entire batch
                n_pos = max(1, sum(num_positive))
                for l3, l4 in zip(losses3, losses4):
                    # normalize the losses by n_pos
                    L = l3 / n_pos + l4 / n_pos
                    Ls.append(L)
                    # losses1 and losses2 are used for loss metrics
                    losses1.append(l3 / n_pos * batch_size)  # rescale for batch
                    losses2.append(l4 / n_pos * batch_size)  # rescale for batch
                autograd.backward(Ls)
            # since we have already normalized the loss, we don't want to normalize
            # by batch-size anymore
            trainer.step(1)
            ce_metric.update(0, losses1)
            smoothl1_metric.update(0, losses2)
            acc_metric.update(labels, outputs)
            if args.log_interval and not (i + 1) % args.log_interval:
                name1, loss1 = ce_metric.get()
                name2, loss2 = smoothl1_metric.get()
                name3, loss3 = acc_metric.get()
                logger.info('[Epoch %d][Batch %d], Speed: %f samples/sec, %s=%f, %s=%f, %s=%f'%(
                    epoch, i, batch_size/(time.time()-btic), name1, loss1, name2, loss2, name3, loss3))
            btic = time.time()

        name1, loss1 = ce_metric.get()
        name2, loss2 = smoothl1_metric.get()
        name3, loss3 = acc_metric.get()
        logger.info('[Epoch %d] Training cost: %f, %s=%f, %s=%f, %s=%f'%(
            epoch, (time.time()-tic), name1, loss1, name2, loss2, name3, loss3))
        map_name, mean_ap = validate(net, val_data, ctx, classes)
        val_msg = '\n'.join(['%s=%f'%(k, v) for k, v in zip(map_name, mean_ap)])
        logger.info('[Epoch %d] Validation: \n%s'%(epoch, val_msg))
        save_params(net, best_map, mean_ap[-1], epoch, args.save_interval, args.save_prefix)

if __name__ == '__main__':
    args = parse_args()
    # fix seed for mxnet, numpy and python builtin random generator.
    gutils.random.seed(args.seed)

    # training contexts
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]

    # training data
    train_dataset, val_dataset = get_dataset(args.dataset)
    train_data, val_data = get_dataloader(
        train_dataset, val_dataset, args.data_shape, args.batch_size, args.num_workers)
    classes = train_dataset.classes  # class names

    # network
    net_name = '_'.join(('ssd', str(args.data_shape), args.network, args.dataset))
    net = get_model(net_name, pretrained_base=True)
    if args.resume.strip():
        net.load_params(args.resume.strip())

    # training
    args.save_prefix += net_name
    train(net, train_data, val_data, classes, args)
