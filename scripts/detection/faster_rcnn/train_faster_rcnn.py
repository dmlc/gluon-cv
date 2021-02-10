"""Train Faster-RCNN end to end."""
import argparse
import os

# disable autotune
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_GPU_MEM_POOL_TYPE'] = 'Round'
os.environ['MXNET_GPU_MEM_POOL_ROUND_LINEAR_CUTOFF'] = '26'
os.environ['MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD'] = '999'
os.environ['MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD'] = '25'
os.environ['MXNET_GPU_COPY_NTHREADS'] = '1'
os.environ['MXNET_OPTIMIZER_AGGREGATION_SIZE'] = '54'

import logging
import time
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.contrib import amp
import gluoncv as gcv

gcv.utils.check_version('0.7.0')
from gluoncv import data as gdata
from gluoncv import utils as gutils
from gluoncv.model_zoo import get_model
from gluoncv.data.batchify import FasterRCNNTrainBatchify, Tuple, Append
from gluoncv.data.transforms.presets.rcnn import FasterRCNNDefaultTrainTransform, \
    FasterRCNNDefaultValTransform
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric
from gluoncv.utils.metrics.coco_detection import COCODetectionMetric
from gluoncv.utils.parallel import Parallel
from gluoncv.utils.metrics.rcnn import RPNAccMetric, RPNL1LossMetric, RCNNAccMetric, \
    RCNNL1LossMetric
from gluoncv.model_zoo.rcnn.faster_rcnn.data_parallel import ForwardBackwardTask

try:
    import horovod.mxnet as hvd
except ImportError:
    hvd = None


def parse_args():
    parser = argparse.ArgumentParser(description='Train Faster-RCNN networks e2e.')
    parser.add_argument('--network', type=str, default='resnet50_v1b',
                        choices=['resnet18_v1b', 'resnet50_v1b', 'resnet101_v1d',
                                 'resnest50', 'resnest101', 'resnest269'],
                        help="Base network name which serves as feature extraction base.")
    parser.add_argument('--dataset', type=str, default='voc',
                        help='Training dataset. Now support voc and coco.')
    parser.add_argument('--num-workers', '-j', dest='num_workers', type=int,
                        default=4, help='Number of data workers, you can use larger '
                                        'number to accelerate data loading, '
                                        'if your CPU and GPUs are powerful.')
    parser.add_argument('--batch-size', type=int, default=1, help='Training mini-batch size.')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')
    parser.add_argument('--epochs', type=str, default='',
                        help='Training epochs.')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from previously saved parameters if not None. '
                             'For example, you can resume from ./faster_rcnn_xxx_0123.params')
    parser.add_argument('--start-epoch', type=int, default=0,
                        help='Starting epoch for resuming, default is 0 for new training.'
                             'You can specify it to 100 for example to start from 100 epoch.')
    parser.add_argument('--lr', type=str, default='',
                        help='Learning rate, default is 0.001 for voc single gpu training.')
    parser.add_argument('--lr-decay', type=float, default=0.1,
                        help='decay rate of learning rate. default is 0.1.')
    parser.add_argument('--lr-decay-epoch', type=str, default='',
                        help='epochs at which learning rate decays. default is 14,20 for voc.')
    parser.add_argument('--lr-warmup', type=str, default='',
                        help='warmup iterations to adjust learning rate, default is 0 for voc.')
    parser.add_argument('--lr-warmup-factor', type=float, default=1. / 3.,
                        help='warmup factor of base lr.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum, default is 0.9')
    parser.add_argument('--wd', type=str, default='',
                        help='Weight decay, default is 5e-4 for voc')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Logging mini-batch interval. Default is 100.')
    parser.add_argument('--save-prefix', type=str, default='',
                        help='Saving parameter prefix')
    parser.add_argument('--save-interval', type=int, default=1,
                        help='Saving parameters epoch interval, best model will always be saved.')
    parser.add_argument('--val-interval', type=int, default=1,
                        help='Epoch interval for validation, increase the number will reduce the '
                             'training time if validation is slow.')
    parser.add_argument('--seed', type=int, default=233,
                        help='Random seed to be fixed.')
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        help='Print helpful debugging info once set.')
    parser.add_argument('--mixup', action='store_true', help='Use mixup training.')
    parser.add_argument('--no-mixup-epochs', type=int, default=20,
                        help='Disable mixup training if enabled in the last N epochs.')

    # Norm layer options
    parser.add_argument('--norm-layer', type=str, default=None, choices=[None, 'syncbn'],
                        help='Type of normalization layer to use. '
                             'If set to None, backbone normalization layer will be frozen,'
                             ' and no normalization layer will be used in R-CNN. '
                             'Currently supports \'syncbn\', and None, default is None.'
                             'Note that if horovod is enabled, sync bn will not work correctly.')

    # Loss options
    parser.add_argument('--rpn-smoothl1-rho', type=float, default=1. / 9.,
                        help='RPN box regression transition point from L1 to L2 loss.'
                             'Set to 0.0 to make the loss simply L1.')
    parser.add_argument('--rcnn-smoothl1-rho', type=float, default=1.,
                        help='RCNN box regression transition point from L1 to L2 loss.'
                             'Set to 0.0 to make the loss simply L1.')

    # FPN options
    parser.add_argument('--use-fpn', action='store_true',
                        help='Whether to use feature pyramid network.')

    # Performance options
    parser.add_argument('--disable-hybridization', action='store_true',
                        help='Whether to disable hybridize the model. '
                             'Memory usage and speed will decrese.')
    parser.add_argument('--static-alloc', action='store_true',
                        help='Whether to use static memory allocation. Memory usage will increase.')
    parser.add_argument('--amp', action='store_true',
                        help='Use MXNet AMP for mixed precision training.')
    parser.add_argument('--horovod', action='store_true',
                        help='Use MXNet Horovod for distributed training. Must be run with OpenMPI. '
                             '--gpus is ignored when using --horovod.')
    parser.add_argument('--executor-threads', type=int, default=1,
                        help='Number of threads for executor for scheduling ops. '
                             'More threads may incur higher GPU memory footprint, '
                             'but may speed up throughput. Note that when horovod is used, '
                             'it is set to 1.')
    parser.add_argument('--kv-store', type=str, default='nccl',
                        help='KV store options. local, device, nccl, dist_sync, dist_device_sync, '
                             'dist_async are available.')

    # Advanced options. Expert Only!! Currently non-FPN model is not supported!!
    # Default setting is for MS-COCO.
    # The following options are only used if --custom-model is enabled
    subparsers = parser.add_subparsers(dest='custom_model')
    custom_model_parser = subparsers.add_parser(
        'custom-model',
        help='Use custom Faster R-CNN w/ FPN model. This is for expert only!'
             ' You can modify model internal parameters here. Once enabled, '
             'custom model options become available.')
    custom_model_parser.add_argument(
        '--no-pretrained-base', action='store_true', help='Disable pretrained base network.')
    custom_model_parser.add_argument(
        '--num-fpn-filters', type=int, default=256, help='Number of filters in FPN output layers.')
    custom_model_parser.add_argument(
        '--num-box-head-conv', type=int, default=4,
        help='Number of convolution layers to use in box head if '
             'batch normalization is not frozen.')
    custom_model_parser.add_argument(
        '--num-box-head-conv-filters', type=int, default=256,
        help='Number of filters for convolution layers in box head.'
             ' Only applicable if batch normalization is not frozen.')
    custom_model_parser.add_argument(
        '--num_box_head_dense_filters', type=int, default=1024,
        help='Number of hidden units for the last fully connected layer in '
             'box head.')
    custom_model_parser.add_argument(
        '--image-short', type=str, default='800',
        help='Short side of the image. Pass a tuple to enable random scale augmentation.')
    custom_model_parser.add_argument(
        '--image-max-size', type=int, default=1333,
        help='Max size of the longer side of the image.')
    custom_model_parser.add_argument(
        '--nms-thresh', type=float, default=0.5,
        help='Non-maximum suppression threshold for R-CNN. '
             'You can specify < 0 or > 1 to disable NMS.')
    custom_model_parser.add_argument(
        '--nms-topk', type=int, default=-1,
        help='Apply NMS to top k detection results in R-CNN. '
             'Set to -1 to disable so that every Detection result is used in NMS.')
    custom_model_parser.add_argument(
        '--post-nms', type=int, default=-1,
        help='Only return top `post_nms` detection results, the rest is discarded.'
             ' Set to -1 to return all detections.')
    custom_model_parser.add_argument(
        '--roi-mode', type=str, default='align', choices=['align', 'pool'],
        help='ROI pooling mode. Currently support \'pool\' and \'align\'.')
    custom_model_parser.add_argument(
        '--roi-size', type=str, default='7,7',
        help='The output spatial size of ROI layer. eg. ROIAlign, ROIPooling')
    custom_model_parser.add_argument(
        '--strides', type=str, default='4,8,16,32,64',
        help='Feature map stride with respect to original image. '
             'This is usually the ratio between original image size and '
             'feature map size. Since the custom model uses FPN, it is a list of ints')
    custom_model_parser.add_argument(
        '--clip', type=float, default=4.14,
        help='Clip bounding box transformation predictions '
             'to prevent exponentiation from overflowing')
    custom_model_parser.add_argument(
        '--rpn-channel', type=int, default=256,
        help='Number of channels used in RPN convolution layers.')
    custom_model_parser.add_argument(
        '--anchor-base-size', type=int, default=16,
        help='The width(and height) of reference anchor box.')
    custom_model_parser.add_argument(
        '--anchor-aspect-ratio', type=str, default='0.5,1,2',
        help='The aspect ratios of anchor boxes.')
    custom_model_parser.add_argument(
        '--anchor-scales', type=str, default='2,4,8,16,32',
        help='The scales of anchor boxes with respect to base size. '
             'We use the following form to compute the shapes of anchors: '
             'anchor_width = base_size * scale * sqrt(1 / ratio)'
             'anchor_height = base_size * scale * sqrt(ratio)')
    custom_model_parser.add_argument(
        '--anchor-alloc-size', type=str, default='384,384',
        help='Allocate size for the anchor boxes as (H, W). '
             'We generate enough anchors for large feature map, e.g. 384x384. '
             'During inference we can have variable input sizes, '
             'at which time we can crop corresponding anchors from this large '
             'anchor map so we can skip re-generating anchors for each input. ')
    custom_model_parser.add_argument(
        '--rpn-nms-thresh', type=float, default='0.7',
        help='Non-maximum suppression threshold for RPN.')
    custom_model_parser.add_argument(
        '--rpn-train-pre-nms', type=int, default=12000,
        help='Filter top proposals before NMS in RPN training.')
    custom_model_parser.add_argument(
        '--rpn-train-post-nms', type=int, default=2000,
        help='Return top proposal results after NMS in RPN training. '
             'Will be set to rpn_train_pre_nms if it is larger than '
             'rpn_train_pre_nms.')
    custom_model_parser.add_argument(
        '--rpn-test-pre-nms', type=int, default=6000,
        help='Filter top proposals before NMS in RPN testing.')
    custom_model_parser.add_argument(
        '--rpn-test-post-nms', type=int, default=1000,
        help='Return top proposal results after NMS in RPN testing. '
             'Will be set to rpn_test_pre_nms if it is larger than rpn_test_pre_nms.')
    custom_model_parser.add_argument(
        '--rpn-min-size', type=int, default=1,
        help='Proposals whose size is smaller than ``min_size`` will be discarded.')
    custom_model_parser.add_argument(
        '--rcnn-num-samples', type=int, default=512, help='Number of samples for RCNN training.')
    custom_model_parser.add_argument(
        '--rcnn-pos-iou-thresh', type=float, default=0.5,
        help='Proposal whose IOU larger than ``pos_iou_thresh`` is '
             'regarded as positive samples for R-CNN.')
    custom_model_parser.add_argument(
        '--rcnn-pos-ratio', type=float, default=0.25,
        help='``pos_ratio`` defines how many positive samples '
             '(``pos_ratio * num_sample``) is to be sampled for R-CNN.')
    custom_model_parser.add_argument(
        '--max-num-gt', type=int, default=100,
        help='Maximum ground-truth number for each example. This is only an upper bound, not'
             'necessarily very precise. However, using a very big number may impact the '
             'training speed.')

    args = parser.parse_args()

    if args.horovod:
        if hvd is None:
            raise SystemExit("Horovod not found, please check if you installed it correctly.")
        hvd.init()

    if args.dataset == 'voc':
        args.epochs = int(args.epochs) if args.epochs else 20
        args.lr_decay_epoch = args.lr_decay_epoch if args.lr_decay_epoch else '14,20'
        args.lr = float(args.lr) if args.lr else 0.001
        args.lr_warmup = args.lr_warmup if args.lr_warmup else -1
        args.wd = float(args.wd) if args.wd else 5e-4
    elif args.dataset == 'coco':
        args.epochs = int(args.epochs) if args.epochs else 26
        args.lr_decay_epoch = args.lr_decay_epoch if args.lr_decay_epoch else '17,23'
        args.lr = float(args.lr) if args.lr else 0.00125
        args.lr_warmup = args.lr_warmup if args.lr_warmup else 1000
        args.wd = float(args.wd) if args.wd else 1e-4

    def str_args2num_args(arguments, args_name, num_type):
        try:
            ret = [num_type(x) for x in arguments.split(',')]
            if len(ret) == 1:
                return ret[0]
            return ret
        except ValueError:
            raise ValueError('invalid value for', args_name, arguments)

    if args.custom_model:
        args.image_short = str_args2num_args(args.image_short, '--image-short', int)
        args.roi_size = str_args2num_args(args.roi_size, '--roi-size', int)
        args.strides = str_args2num_args(args.strides, '--strides', int)
        args.anchor_aspect_ratio = str_args2num_args(args.anchor_aspect_ratio,
                                                     '--anchor-aspect-ratio', float)
        args.anchor_scales = str_args2num_args(args.anchor_scales, '--anchor-scales', float)
        args.anchor_alloc_size = str_args2num_args(args.anchor_alloc_size,
                                                   '--anchor-alloc-size', int)
    if args.amp and args.norm_layer == 'syncbn':
        raise NotImplementedError('SyncBatchNorm currently does not support AMP.')

    return args


def get_dataset(dataset, args):
    if dataset.lower() == 'voc':
        train_dataset = gdata.VOCDetection(
            splits=[(2007, 'trainval'), (2012, 'trainval')])
        val_dataset = gdata.VOCDetection(
            splits=[(2007, 'test')])
        val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    elif dataset.lower() in ['clipart', 'comic', 'watercolor']:
        root = os.path.join('~', '.mxnet', 'datasets', dataset.lower())
        train_dataset = gdata.CustomVOCDetection(root=root, splits=[('', 'train')],
                                                 generate_classes=True)
        val_dataset = gdata.CustomVOCDetection(root=root, splits=[('', 'test')],
                                               generate_classes=True)
        val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
    elif dataset.lower() == 'coco':
        train_dataset = gdata.COCODetection(splits='instances_train2017', use_crowd=False)
        val_dataset = gdata.COCODetection(splits='instances_val2017', skip_empty=False)
        val_metric = COCODetectionMetric(val_dataset, args.save_prefix + '_eval', cleanup=True)
    else:
        raise NotImplementedError('Dataset: {} not implemented.'.format(dataset))
    if args.mixup:
        from gluoncv.data.mixup import detection
        train_dataset = detection.MixupDetection(train_dataset)
    return train_dataset, val_dataset, val_metric


def get_dataloader(net, train_dataset, val_dataset, train_transform, val_transform, batch_size,
                   num_shards, args):
    """Get dataloader."""
    train_bfn = FasterRCNNTrainBatchify(net, num_shards)
    if hasattr(train_dataset, 'get_im_aspect_ratio'):
        im_aspect_ratio = train_dataset.get_im_aspect_ratio()
    else:
        im_aspect_ratio = [1.] * len(train_dataset)
    train_sampler = \
        gcv.data.sampler.SplitSortedBucketSampler(im_aspect_ratio, batch_size,
                                                num_parts=hvd.size() if args.horovod else 1,
                                                part_index=hvd.rank() if args.horovod else 0,
                                                shuffle=True)
    train_loader = mx.gluon.data.DataLoader(train_dataset.transform(
        train_transform(net.short, net.max_size, net, ashape=net.ashape, multi_stage=args.use_fpn)),
        batch_sampler=train_sampler, batchify_fn=train_bfn, num_workers=args.num_workers)
    val_bfn = Tuple(*[Append() for _ in range(3)])
    short = net.short[-1] if isinstance(net.short, (tuple, list)) else net.short
    # validation use 1 sample per device
    val_loader = mx.gluon.data.DataLoader(
        val_dataset.transform(val_transform(short, net.max_size)), num_shards, False,
        batchify_fn=val_bfn, last_batch='keep', num_workers=args.num_workers)
    return train_loader, val_loader


def save_params(net, logger, best_map, current_map, epoch, save_interval, prefix):
    current_map = float(current_map)
    if current_map > best_map[0]:
        logger.info('[Epoch {}] mAP {} higher than current best {} saving to {}'.format(
            epoch, current_map, best_map, '{:s}_best.params'.format(prefix)))
        best_map[0] = current_map
        net.save_parameters('{:s}_best.params'.format(prefix))
        with open(prefix + '_best_map.log', 'a') as f:
            f.write('{:04d}:\t{:.4f}\n'.format(epoch, current_map))
    if save_interval and (epoch + 1) % save_interval == 0:
        logger.info('[Epoch {}] Saving parameters to {}'.format(
            epoch, '{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map)))
        net.save_parameters('{:s}_{:04d}_{:.4f}.params'.format(prefix, epoch, current_map))


def split_and_load(batch, ctx_list):
    """Split data to 1 batch each device."""
    new_batch = []
    for i, data in enumerate(batch):
        if isinstance(data, (list, tuple)):
            new_data = [x.as_in_context(ctx) for x, ctx in zip(data, ctx_list)]
        else:
            new_data = [data.as_in_context(ctx_list[0])]
        new_batch.append(new_data)
    return new_batch


def validate(net, val_data, ctx, eval_metric, args):
    """Test on validation dataset."""
    clipper = gcv.nn.bbox.BBoxClipToImage()
    eval_metric.reset()
    if not args.disable_hybridization:
        # input format is differnet than training, thus rehybridization is needed.
        net.hybridize(static_alloc=args.static_alloc)
    for batch in val_data:
        batch = split_and_load(batch, ctx_list=ctx)
        det_bboxes = []
        det_ids = []
        det_scores = []
        gt_bboxes = []
        gt_ids = []
        gt_difficults = []
        for x, y, im_scale in zip(*batch):
            # get prediction results
            ids, scores, bboxes = net(x)
            det_ids.append(ids)
            det_scores.append(scores)
            # clip to image size
            det_bboxes.append(clipper(bboxes, x))
            # rescale to original resolution
            im_scale = im_scale.reshape((-1)).asscalar()
            det_bboxes[-1] *= im_scale
            # split ground truths
            gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
            gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
            gt_bboxes[-1] *= im_scale
            gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

        # update metric
        for det_bbox, det_id, det_score, gt_bbox, gt_id, gt_diff in zip(det_bboxes, det_ids,
                                                                        det_scores, gt_bboxes,
                                                                        gt_ids, gt_difficults):
            eval_metric.update(det_bbox, det_id, det_score, gt_bbox, gt_id, gt_diff)
    return eval_metric.get()


def get_lr_at_iter(alpha, lr_warmup_factor=1. / 3.):
    return lr_warmup_factor * (1 - alpha) + alpha


def train(net, train_data, val_data, eval_metric, batch_size, ctx, args):
    """Training pipeline"""
    args.kv_store = 'device' if (args.amp and 'nccl' in args.kv_store) else args.kv_store
    kv = mx.kvstore.create(args.kv_store)
    net.collect_params().setattr('grad_req', 'null')
    net.collect_train_params().setattr('grad_req', 'write')
    optimizer_params = {'learning_rate': args.lr, 'wd': args.wd, 'momentum': args.momentum}
    if args.amp:
        optimizer_params['multi_precision'] = True
    if args.horovod:
        hvd.broadcast_parameters(net.collect_params(), root_rank=0)
        trainer = hvd.DistributedTrainer(
            net.collect_train_params(),  # fix batchnorm, fix first stage, etc...
            'sgd',
            optimizer_params)
    else:
        trainer = gluon.Trainer(
            net.collect_train_params(),  # fix batchnorm, fix first stage, etc...
            'sgd',
            optimizer_params,
            update_on_kvstore=(False if args.amp else None), kvstore=kv)

    if args.amp:
        amp.init_trainer(trainer)

    # lr decay policy
    lr_decay = float(args.lr_decay)
    lr_steps = sorted([float(ls) for ls in args.lr_decay_epoch.split(',') if ls.strip()])
    lr_warmup = float(args.lr_warmup)  # avoid int division

    # TODO(zhreshold) losses?
    rpn_cls_loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
    rpn_box_loss = mx.gluon.loss.HuberLoss(rho=args.rpn_smoothl1_rho)  # == smoothl1
    rcnn_cls_loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()
    rcnn_box_loss = mx.gluon.loss.HuberLoss(rho=args.rcnn_smoothl1_rho)  # == smoothl1
    metrics = [mx.metric.Loss('RPN_Conf'),
               mx.metric.Loss('RPN_SmoothL1'),
               mx.metric.Loss('RCNN_CrossEntropy'),
               mx.metric.Loss('RCNN_SmoothL1'), ]

    rpn_acc_metric = RPNAccMetric()
    rpn_bbox_metric = RPNL1LossMetric()
    rcnn_acc_metric = RCNNAccMetric()
    rcnn_bbox_metric = RCNNL1LossMetric()
    metrics2 = [rpn_acc_metric, rpn_bbox_metric, rcnn_acc_metric, rcnn_bbox_metric]

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
    if args.custom_model:
        logger.info('Custom model enabled. Expert Only!! Currently non-FPN model is not supported!!'
                    ' Default setting is for MS-COCO.')
    logger.info(args)

    if args.verbose:
        logger.info('Trainable parameters:')
        logger.info(net.collect_train_params().keys())
    logger.info('Start training from [Epoch {}]'.format(args.start_epoch))
    best_map = [0]
    for epoch in range(args.start_epoch, args.epochs):
        rcnn_task = ForwardBackwardTask(net, trainer, rpn_cls_loss, rpn_box_loss, rcnn_cls_loss,
                                        rcnn_box_loss, mix_ratio=1.0, amp_enabled=args.amp)
        executor = Parallel(args.executor_threads, rcnn_task) if not args.horovod else None
        mix_ratio = 1.0
        if not args.disable_hybridization:
            net.hybridize(static_alloc=args.static_alloc)
        if args.mixup:
            # TODO(zhreshold) only support evenly mixup now, target generator needs to be modified otherwise
            train_data._dataset._data.set_mixup(np.random.uniform, 0.5, 0.5)
            mix_ratio = 0.5
            if epoch >= args.epochs - args.no_mixup_epochs:
                train_data._dataset._data.set_mixup(None)
                mix_ratio = 1.0
        while lr_steps and epoch >= lr_steps[0]:
            new_lr = trainer.learning_rate * lr_decay
            lr_steps.pop(0)
            trainer.set_learning_rate(new_lr)
            logger.info("[Epoch {}] Set learning rate to {}".format(epoch, new_lr))
        for metric in metrics:
            metric.reset()
        tic = time.time()
        btic = time.time()
        base_lr = trainer.learning_rate
        rcnn_task.mix_ratio = mix_ratio
        for i, batch in enumerate(train_data):
            if epoch == 0 and i <= lr_warmup:
                # adjust based on real percentage
                new_lr = base_lr * get_lr_at_iter(i / lr_warmup, args.lr_warmup_factor)
                if new_lr != trainer.learning_rate:
                    if i % args.log_interval == 0:
                        logger.info(
                            '[Epoch 0 Iteration {}] Set learning rate to {}'.format(i, new_lr))
                    trainer.set_learning_rate(new_lr)
            batch = split_and_load(batch, ctx_list=ctx)
            metric_losses = [[] for _ in metrics]
            add_losses = [[] for _ in metrics2]
            if executor is not None:
                for data in zip(*batch):
                    executor.put(data)
            for j in range(len(ctx)):
                if executor is not None:
                    result = executor.get()
                else:
                    result = rcnn_task.forward_backward(list(zip(*batch))[0])
                if (not args.horovod) or hvd.rank() == 0:
                    for k in range(len(metric_losses)):
                        metric_losses[k].append(result[k])
                    for k in range(len(add_losses)):
                        add_losses[k].append(result[len(metric_losses) + k])
            trainer.step(batch_size)

            for metric, record in zip(metrics, metric_losses):
                metric.update(0, record)
            for metric, records in zip(metrics2, add_losses):
                for pred in records:
                    metric.update(pred[0], pred[1])

            # update metrics
            if (not args.horovod or hvd.rank() == 0) and args.log_interval \
                    and not (i + 1) % args.log_interval:
                msg = ','.join(
                    ['{}={:.3f}'.format(*metric.get()) for metric in metrics + metrics2])
                logger.info('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}'.format(
                    epoch, i, args.log_interval * args.batch_size / (time.time() - btic), msg))
                btic = time.time()

        if (not args.horovod) or hvd.rank() == 0:
            msg = ','.join(['{}={:.3f}'.format(*metric.get()) for metric in metrics])
            logger.info('[Epoch {}] Training cost: {:.3f}, {}'.format(
                epoch, (time.time() - tic), msg))
            if not (epoch + 1) % args.val_interval:
                # consider reduce the frequency of validation to save time
                map_name, mean_ap = validate(net, val_data, ctx, eval_metric, args)
                val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
                logger.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
                current_map = float(mean_ap[-1])
            else:
                current_map = 0.
            save_params(net, logger, best_map, current_map, epoch, args.save_interval,
                        args.save_prefix)


if __name__ == '__main__':
    import sys

    sys.setrecursionlimit(1100)
    args = parse_args()
    # fix seed for mxnet, numpy and python builtin random generator.
    gutils.random.seed(args.seed)

    if args.amp:
        amp.init()

    # training contexts
    if args.horovod:
        ctx = [mx.gpu(hvd.local_rank())]
    else:
        ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
        ctx = ctx if ctx else [mx.cpu()]

    # training data
    train_dataset, val_dataset, eval_metric = get_dataset(args.dataset, args)

    # network
    kwargs = {}
    module_list = []
    if args.use_fpn:
        module_list.append('fpn')
    if args.norm_layer is not None:
        module_list.append(args.norm_layer)
        if args.norm_layer == 'syncbn':
            kwargs['num_devices'] = len(ctx)

    num_gpus = hvd.size() if args.horovod else len(ctx)
    net_name = '_'.join(('faster_rcnn', *module_list, args.network, args.dataset))
    if args.custom_model:
        args.use_fpn = True
        net_name = '_'.join(('custom_faster_rcnn_fpn', args.network, args.dataset))
        if args.norm_layer == 'syncbn':
            norm_layer = gluon.contrib.nn.SyncBatchNorm
            norm_kwargs = {'num_devices': len(ctx)}
            sym_norm_layer = mx.sym.contrib.SyncBatchNorm
            sym_norm_kwargs = {'ndev': len(ctx)}
        elif args.norm_layer == 'gn':
            norm_layer = gluon.nn.GroupNorm
            norm_kwargs = {'groups': 8}
            sym_norm_layer = mx.sym.GroupNorm
            sym_norm_kwargs = {'groups': 8}
        else:
            norm_layer = gluon.nn.BatchNorm
            norm_kwargs = None
            sym_norm_layer = None
            sym_norm_kwargs = None
        classes = train_dataset.CLASSES
        net = get_model('custom_faster_rcnn_fpn', classes=classes, transfer=None,
                        dataset=args.dataset, pretrained_base=not args.no_pretrained_base,
                        base_network_name=args.network, norm_layer=norm_layer,
                        norm_kwargs=norm_kwargs, sym_norm_kwargs=sym_norm_kwargs,
                        num_fpn_filters=args.num_fpn_filters,
                        num_box_head_conv=args.num_box_head_conv,
                        num_box_head_conv_filters=args.num_box_head_conv_filters,
                        num_box_head_dense_filters=args.num_box_head_dense_filters,
                        short=args.image_short, max_size=args.image_max_size, min_stage=2,
                        max_stage=6, nms_thresh=args.nms_thresh, nms_topk=args.nms_topk,
                        post_nms=args.post_nms, roi_mode=args.roi_mode, roi_size=args.roi_size,
                        strides=args.strides, clip=args.clip, rpn_channel=args.rpn_channel,
                        base_size=args.anchor_base_size, scales=args.anchor_scales,
                        ratios=args.anchor_aspect_ratio, alloc_size=args.anchor_alloc_size,
                        rpn_nms_thresh=args.rpn_nms_thresh,
                        rpn_train_pre_nms=args.rpn_train_pre_nms,
                        rpn_train_post_nms=args.rpn_train_post_nms,
                        rpn_test_pre_nms=args.rpn_test_pre_nms,
                        rpn_test_post_nms=args.rpn_test_post_nms, rpn_min_size=args.rpn_min_size,
                        per_device_batch_size=args.batch_size // num_gpus,
                        num_sample=args.rcnn_num_samples, pos_iou_thresh=args.rcnn_pos_iou_thresh,
                        pos_ratio=args.rcnn_pos_ratio, max_num_gt=args.max_num_gt)
    else:
        net = get_model(net_name, pretrained_base=True,
                        per_device_batch_size=args.batch_size // num_gpus, **kwargs)
    args.save_prefix += net_name
    if args.resume.strip():
        net.load_parameters(args.resume.strip())
    else:
        for param in net.collect_params().values():
            if param._data is not None:
                continue
            param.initialize()
    net.collect_params().reset_ctx(ctx)

    if args.amp:
        # Cast both weights and gradients to 'float16'
        net.cast('float16')
        # These layers don't support type 'float16'
        net.collect_params('.*batchnorm.*').setattr('dtype', 'float32')
        net.collect_params('.*normalizedperclassboxcenterencoder.*').setattr('dtype', 'float32')

    # dataloader
    batch_size = args.batch_size // num_gpus if args.horovod else args.batch_size
    train_data, val_data = get_dataloader(
        net, train_dataset, val_dataset, FasterRCNNDefaultTrainTransform,
        FasterRCNNDefaultValTransform, batch_size, len(ctx), args)

    # training
    train(net, train_data, val_data, eval_metric, batch_size, ctx, args)
