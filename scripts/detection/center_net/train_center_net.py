"""Train CenterNet"""
import argparse
import os
import logging
import warnings
import time
import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
import gluoncv as gcv
gcv.utils.check_version('0.6.0')
from gluoncv import data as gdata
from gluoncv import utils as gutils
from gluoncv.model_zoo import get_model
from gluoncv.data.batchify import Tuple, Stack, Pad
from gluoncv.data.transforms.presets.center_net import CenterNetDefaultTrainTransform
from gluoncv.data.transforms.presets.center_net import CenterNetDefaultValTransform, get_post_transform

from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric
from gluoncv.utils.metrics.accuracy import Accuracy
from gluoncv.utils import LRScheduler, LRSequential


def parse_args():
    parser = argparse.ArgumentParser(description='Train CenterNet networks.')
    parser.add_argument('--network', type=str, default='resnet18_v1b',
                        help="Base network name which serves as feature extraction base.")
    parser.add_argument('--data-shape', type=int, default=512,
                        help="Input data shape, use 300, 512.")
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training mini-batch size')
    parser.add_argument('--dataset', type=str, default='voc',
                        help='Training dataset. Now support voc.')
    parser.add_argument('--dataset-root', type=str, default='~/.mxnet/datasets/',
                        help='Path of the directory where the dataset is located.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=4, help='Number of data workers, you can use larger '
                        'number to accelerate data loading, if you CPU and GPUs are powerful.')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--epochs', type=int, default=140,
                        help='Training epochs.')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from previously saved parameters if not None. '
                        'For example, you can resume from ./ssd_xxx_0123.params')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Starting epoch for resuming, default is 0 for new training.'
                        'You can specify it to 100 for example to start from 100 epoch.')
    parser.add_argument('--lr', type=float, default=1.25e-4,
                        help='Learning rate, default is 0.000125')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-epoch', type=str, default='90,120',
                        help='epochs at which learning rate decays. default is 90,120.')
    parser.add_argument('--lr-mode', type=str, default='step',
                        help='learning rate scheduler mode. options are step, poly and cosine.')
    parser.add_argument('--warmup-lr', type=float, default=0.0,
                        help='starting warmup learning rate. default is 0.0.')
    parser.add_argument('--warmup-epochs', type=int, default=0,
                        help='number of warmup epochs.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum, default is 0.9')
    parser.add_argument('--wd', type=float, default=0.0001,
                        help='Weight decay, default is 1e-4')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Logging mini-batch interval. Default is 100.')
    parser.add_argument('--num-samples', type=int, default=-1,
                        help='Training images. Use -1 to automatically get the number.')
    parser.add_argument('--save-prefix', type=str, default='',
                        help='Saving parameter prefix')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Saving parameters epoch interval, best model will always be saved.')
    parser.add_argument('--val-interval', type=int, default=1,
                        help='Epoch interval for validation, increase the number will reduce the '
                             'training time if validation is slow.')
    parser.add_argument('--seed', type=int, default=233,
                        help='Random seed to be fixed.')
    parser.add_argument('--wh-weight', type=float, default=0.1,
                        help='Loss weight for width/height')
    parser.add_argument('--center-reg-weight', type=float, default=1.0,
                        help='Center regression loss weight')
    parser.add_argument('--flip-validation', action='store_true',
                        help='flip data augmentation in validation.')

    args = parser.parse_args()
    return args

def get_dataset(dataset, args):
    if dataset.lower() == 'voc':
        train_dataset = gdata.VOCDetection(
            splits=[(2007, 'trainval'), (2012, 'trainval')])
        val_dataset = gdata.VOCDetection(
            splits=[(2007, 'test')])
        val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    elif dataset.lower() == 'coco':
        train_dataset = gdata.COCODetection(root=args.dataset_root + "/coco", splits='instances_train2017')
        val_dataset = gdata.COCODetection(root=args.dataset_root + "/coco", splits='instances_val2017', skip_empty=False)
        val_metric = COCODetectionMetric(
            val_dataset, args.save_prefix + '_eval', cleanup=True,
            data_shape=(args.data_shape, args.data_shape), post_affine=get_post_transform)
        # coco validation is slow, consider increase the validation interval
        if args.val_interval == 1:
            args.val_interval = 10
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    if args.num_samples < 0:
        args.num_samples = len(train_dataset)
    return train_dataset, val_dataset, val_metric

def get_dataloader(net, train_dataset, val_dataset, data_shape, batch_size, num_workers, ctx):
    """Get dataloader."""
    width, height = data_shape, data_shape
    num_class = len(train_dataset.classes)
    batchify_fn = Tuple([Stack() for _ in range(6)])  # stack image, cls_targets, box_targets
    train_loader = gluon.data.DataLoader(
        train_dataset.transform(CenterNetDefaultTrainTransform(
            width, height, num_class=num_class, scale_factor=net.scale)),
        batch_size, True, batchify_fn=batchify_fn, last_batch='rollover', num_workers=num_workers)
    val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
    val_loader = gluon.data.DataLoader(
        val_dataset.transform(CenterNetDefaultValTransform(width, height)),
        batch_size, False, batchify_fn=val_batchify_fn, last_batch='keep', num_workers=num_workers)
    return train_loader, val_loader

def save_params(net, best_map, current_map, epoch, save_interval, prefix):
    current_map = float(current_map)
    if current_map > best_map[0]:
        best_map[0] = current_map
        net.save_parameters('{:s}_best.params'.format(prefix, epoch, current_map))
        with open(prefix+'_best_map.log', 'a') as f:
            f.write('{:04d}:\t{:.4f}\n'.format(epoch, current_map))
    if save_interval and epoch % save_interval == 0:
        net.save_parameters('{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map))

def validate(net, val_data, ctx, eval_metric, flip_test=False):
    """Test on validation dataset."""
    eval_metric.reset()
    net.flip_test = flip_test
    mx.nd.waitall()
    net.hybridize()
    for batch in val_data:
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0, even_split=False)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0, even_split=False)
        det_bboxes = []
        det_ids = []
        det_scores = []
        gt_bboxes = []
        gt_ids = []
        gt_difficults = []
        for x, y in zip(data, label):
            # get prediction results
            ids, scores, bboxes = net(x)
            det_ids.append(ids)
            det_scores.append(scores)
            # clip to image size
            det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
            # split ground truths
            gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
            gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
            gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

        # update metric
        eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
    return eval_metric.get()

def train(net, train_data, val_data, eval_metric, ctx, args):
    """Training pipeline"""
    net.collect_params().reset_ctx(ctx)
    # lr decay policy
    lr_decay = float(args.lr_decay)
    lr_steps = sorted([int(ls) for ls in args.lr_decay_epoch.split(',') if ls.strip()])
    lr_decay_epoch = [e - args.warmup_epochs for e in lr_steps]
    num_batches = args.num_samples // args.batch_size
    lr_scheduler = LRSequential([
        LRScheduler('linear', base_lr=0, target_lr=args.lr,
                    nepochs=args.warmup_epochs, iters_per_epoch=num_batches),
        LRScheduler(args.lr_mode, base_lr=args.lr,
                    nepochs=args.epochs - args.warmup_epochs,
                    iters_per_epoch=num_batches,
                    step_epoch=lr_decay_epoch,
                    step_factor=args.lr_decay, power=2),
    ])

    for k, v in net.collect_params('.*bias').items():
        v.wd_mult = 0.0
    trainer = gluon.Trainer(
                net.collect_params(), 'adam',
                {'learning_rate': args.lr, 'wd': args.wd,
                 'lr_scheduler': lr_scheduler})

    heatmap_loss = gcv.loss.HeatmapFocalLoss(from_logits=True)
    wh_loss = gcv.loss.MaskedL1Loss(weight=args.wh_weight)
    center_reg_loss = gcv.loss.MaskedL1Loss(weight=args.center_reg_weight)
    heatmap_loss_metric = mx.metric.Loss('HeatmapFocal')
    wh_metric = mx.metric.Loss('WHL1')
    center_reg_metric = mx.metric.Loss('CenterRegL1')

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
    logger.info('Start training from [Epoch {}]'.format(args.start_epoch))
    best_map = [0]

    for epoch in range(args.start_epoch, args.epochs):
        wh_metric.reset()
        center_reg_metric.reset()
        tic = time.time()
        btic = time.time()
        net.hybridize()

        for i, batch in enumerate(train_data):
            split_data = [gluon.utils.split_and_load(batch[ind], ctx_list=ctx, batch_axis=0) for ind in range(6)]
            data, heatmap_targets, wh_targets, wh_masks, center_reg_targets, center_reg_masks = split_data
            batch_size = args.batch_size
            with autograd.record():
                sum_losses = []
                heatmap_losses = []
                wh_losses = []
                center_reg_losses = []
                wh_preds = []
                center_reg_preds = []
                for x, heatmap_target, wh_target, wh_mask, center_reg_target, center_reg_mask in zip(*split_data):
                    heatmap_pred, wh_pred, center_reg_pred = net(x)
                    wh_preds.append(wh_pred)
                    center_reg_preds.append(center_reg_pred)
                    wh_losses.append(wh_loss(wh_pred, wh_target, wh_mask))
                    center_reg_losses.append(center_reg_loss(center_reg_pred, center_reg_target, center_reg_mask))
                    heatmap_losses.append(heatmap_loss(heatmap_pred, heatmap_target))
                    curr_loss = heatmap_losses[-1]+ wh_losses[-1] + center_reg_losses[-1]
                    sum_losses.append(curr_loss)
                autograd.backward(sum_losses)
            trainer.step(len(sum_losses))  # step with # gpus

            heatmap_loss_metric.update(0, heatmap_losses)
            wh_metric.update(0, wh_losses)
            center_reg_metric.update(0, center_reg_losses)
            if args.log_interval and not (i + 1) % args.log_interval:
                name2, loss2 = wh_metric.get()
                name3, loss3 = center_reg_metric.get()
                name4, loss4 = heatmap_loss_metric.get()
                logger.info('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, LR={}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
                    epoch, i, batch_size/(time.time()-btic), trainer.learning_rate, name2, loss2, name3, loss3, name4, loss4))
            btic = time.time()

        name2, loss2 = wh_metric.get()
        name3, loss3 = center_reg_metric.get()
        name4, loss4 = heatmap_loss_metric.get()
        logger.info('[Epoch {}] Training cost: {:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
            epoch, (time.time()-tic), name2, loss2, name3, loss3, name4, loss4))
        if (epoch % args.val_interval == 0) or (args.save_interval and epoch % args.save_interval == 0) or (epoch == args.epochs - 1):
            # consider reduce the frequency of validation to save time
            map_name, mean_ap = validate(net, val_data, ctx, eval_metric, flip_test=args.flip_validation)
            val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
            logger.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
            current_map = float(mean_ap[-1])
        else:
            current_map = 0.
        save_params(net, best_map, current_map, epoch, args.save_interval, args.save_prefix)

if __name__ == '__main__':
    args = parse_args()

    # fix seed for mxnet, numpy and python builtin random generator.
    gutils.random.seed(args.seed)

    # training contexts
    ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
    ctx = ctx if ctx else [mx.cpu()]

    # network
    net_name = '_'.join(('center_net', args.network, args.dataset))
    args.save_prefix += net_name
    net = get_model(net_name, pretrained_base=True, norm_layer=gluon.nn.BatchNorm)
    if args.resume.strip():
        net.load_parameters(args.resume.strip())
    else:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            net.initialize()
            # needed for net to be first gpu when using AMP
            net.collect_params().reset_ctx(ctx[0])

    # training data
    train_dataset, val_dataset, eval_metric = get_dataset(args.dataset, args)
    batch_size = args.batch_size
    train_data, val_data = get_dataloader(
        net, train_dataset, val_dataset, args.data_shape, batch_size, args.num_workers, ctx[0])


    # training
    train(net, train_data, val_data, eval_metric, ctx, args)
