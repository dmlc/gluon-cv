import argparse
import logging
logging.basicConfig(level=logging.INFO)
import time
import random
import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
from gluonvision import data as gdata
from gluonvision.model_zoo import get_model
from gluonvision.model_zoo.losses import *
from gluonvision.model_zoo.ssd.transforms import SSDDefaultTrainTransform
from gluonvision.model_zoo.ssd.transforms import SSDDefaultValTransform
from gluonvision.model_zoo.ssd.target import SSDTargetGenerator
from gluonvision.utils.metrics.voc_detection import VOC07MApMetric
from gluonvision.utils.metrics.accuracy import Accuracy

def parse_args():
    parser = argparse.ArgumentParser(description='Train SSD networks.')
    parser.add_argument('--network', type=str, default='resnet50_v1',
                        help="Base network name")
    parser.add_argument('--data-shape', type=int, default=512,
                        help="Input data shape")
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training mini-batch size')
    parser.add_argument('--dataset', type=str, default='voc',
                        help='Training dataset.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=0, help='Number of data workers')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--epochs', type=int, default=240,
                        help='Training epochs.')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from previously saved parameters.')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Starting epoch for resuming, default is 0 for new training.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate, default is 0.001')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-epoch', type=str, default='160,200',
                        help='epoches at which learning rate decays. default is 160,200.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum, default is 0.9')
    parser.add_argument('--wd', type=float, default=0.0005,
                        help='Weight decay, default is 1e-4')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Logging mini-batch interval.')
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

def validate(net, val_data, ctx, classes):
    """Test on validation dataset."""
    metric = VOC07MApMetric(iou_thresh=0.5, class_names=classes)
    net.set_nms(nms_thresh=0.5, nms_topk=400, force_nms=False)
    net.hybridize()
    for batch in val_data:
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        for x, y in zip(data, label):
            ids, scores, bboxes = net(x)
            gt_ids = y.slice_axis(axis=-1, begin=4, end=5)
            gt_bboxes = y.slice_axis(axis=-1, begin=0, end=4)
            gt_difficults = y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None
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

    logging.info('Start training from [Epoch %d]' % args.start_epoch)
    for epoch in range(args.start_epoch, args.epochs):
        if epoch in lr_steps:
            new_lr = trainer.learning_rate * lr_decay[lr_steps.index(epoch)]
            trainer.set_learning_rate(new_lr)
            logging.info("[Epoch {}] Set learning rate to {}".format(epoch, new_lr))
        tic = time.time()
        btic = time.time()
        net.hybridize()
        for i, batch in enumerate(train_data):
            data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
            label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
            outputs = []
            labels = []
            box_outputs = []
            box_labels = []
            losses1 = []
            losses2 = []
            losses3 = []
            losses4 = []
            Ls = []
            num_positive = []
            with autograd.record():
                for x, y in zip(data, label):
                    cls_preds, box_preds, anchors = net(x)
                    with autograd.pause():
                        gt_boxes = nd.slice_axis(y, axis=-1, begin=0, end=4)
                        gt_ids = nd.slice_axis(y, axis=-1, begin=4, end=5)
                        cls_targets, box_targets, box_masks = net.target_generator(
                            anchors, cls_preds, gt_boxes, gt_ids)
                        num_positive.append(nd.sum(box_masks >= 0).asscalar())
                        valid_cls = nd.sum(cls_targets >= 0, axis=0, exclude=True)
                        valid_cls = nd.maximum(valid_cls, nd.ones_like(valid_cls))
                        valid_box = nd.sum(box_masks > 0, axis=0, exclude=True)

                    l1 = cls_loss(cls_preds, cls_targets, (cls_targets >= 0).expand_dims(axis=-1))
                    # losses3.append(l1 * cls_targets.size / cls_targets.shape[0])
                    l1 = l1 / valid_cls * cls_targets.shape[-1]
                    l2 = box_loss(box_preds * box_masks, box_targets)
                    # losses4.append(l2 * box_targets.size / box_targets.shape[0])
                    l2 = l2 / valid_cls * box_targets.size / box_targets.shape[0]
                    L = l1 + l2
                    Ls.append(L)
                    outputs.append(cls_preds)
                    labels.append(cls_targets)
                    box_outputs.append(box_preds * box_masks)
                    box_labels.append(box_targets)
                    losses1.append(l1)
                    losses2.append(l2)
                # n_pos = max(1, sum(num_positive)) / batch[0].shape[0]
                # for l3, l4 in zip(losses3, losses4):
                #     L = l3 / n_pos + l4 / n_pos
                #     Ls.append(L)
                #     losses1.append(l3 / n_pos)
                #     losses2.append(l4 / n_pos)
                autograd.backward(Ls)
            batch_size = batch[0].shape[0]
            trainer.step(batch_size)
            ce_metric.update(0, losses1)
            smoothl1_metric.update(0, losses2)
            acc_metric.update(labels, outputs)
            if args.log_interval and not (i + 1) % args.log_interval:
                name1, loss1 = ce_metric.get()
                name2, loss2 = smoothl1_metric.get()
                name3, loss3 = acc_metric.get()
                logging.info('[Epoch %d][Batch %d], Speed: %f samples/sec, %s=%f, %s=%f, %s=%f'%(
                    epoch, i, batch_size/(time.time()-btic), name1, loss1, name2, loss2, name3, loss3))
            btic = time.time()

        name1, loss1 = ce_metric.get()
        name2, loss2 = smoothl1_metric.get()
        name3, loss3 = acc_metric.get()
        logging.info('[Epoch %d] Training cost: %f, %s=%f, %s=%f, %s=%f'%(
            epoch, (time.time()-tic), name1, loss1, name2, loss2, name3, loss3))
        map_name, mean_ap = validate(net, val_data, ctx, classes)
        val_msg = '\n'.join(['%s=%f'%(k, v) for k, v in zip(map_name, mean_ap)])
        logging.info('[Epoch %d] Validation: \n%s'%(epoch, val_msg))
        net.save_params('ssd_%d_%f' % (epoch, mean_ap[-1]))

if __name__ == '__main__':
    args = parse_args()
    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # training contexts
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]

    # training data
    train_dataset, val_dataset = get_dataset(args.dataset)
    train_data, val_data = get_dataloader(
        train_dataset, val_dataset, args.data_shape, args.batch_size, args.num_workers)
    classes = train_dataset.classes  # class names

    # network
    net_name = '_'.join(('ssd', str(args.data_shape), args.network))
    net = get_model(net_name, classes=len(classes), pretrained=1)  # load pretrained base network

    # training
    train(net, train_data, val_data, classes, args)
