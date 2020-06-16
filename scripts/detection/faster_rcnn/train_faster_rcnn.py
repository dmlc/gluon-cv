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

from mxnet.contrib import amp
import gluoncv as gcv

gcv.utils.check_version('0.7.0')
from gluoncv import utils as gutils
from gluoncv.pipelines.estimators.rcnn import FasterRCNNEstimator

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


if __name__ == '__main__':
    import sys

    sys.setrecursionlimit(1100)
    args = parse_args()
    # fix seed for mxnet, numpy and python builtin random generator.
    gutils.random.seed(args.seed)

    if args.amp:
        amp.init()

    frcnn_estimator = FasterRCNNEstimator(args)

    # training
    frcnn_estimator.fit()
