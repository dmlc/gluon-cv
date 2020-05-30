"""CenterNet Estimator"""
import os
import warnings
from yacs.config import CfgNode as CN
import mxnet as mx
from mxnet import gluon
from .base import BaseEstimator, set_default
from ..data import COCODetection, VOCDetection
from ..data.transforms.presets.center_net import CenterNetDefaultTrainTransform
from ..data.transforms.presets.center_net import CenterNetDefaultValTransform, get_post_transform
from ..data.batchify import Tuple, Stack, Pad
from ..utils.metric import COCODetectionMetric
from ..utils.metrics.accuracy import Accuracy
from ..utils import LRScheduler, LRSequential
from ..utils import random as _random
from ..model_zoo.center_net import get_center_net
from ..loss import *


def get_center_net_defaults():
    cfg = CN()

    cfg.MODEL = CN()
    # base feature network
    cfg.MODEL.BASE_NETWORK = 'dla34_deconv_dcnv2'
    cfg.MODEL.HEADS = CN()
    # init bias for conv
    cfg.MODEL.HEADS.BIAS = -2.19  # use bias = -log((1 - 0.1) / 0.1)
    # wh head channel
    cfg.MODEL.HEADS.WH_OUTPUTS = 2
    # regression head channel
    cfg.MODEL.HEADS.REG_OUTPUTS = 2
    # additional conv channel
    cfg.MODEL.HEADS.HEAD_CONV_CHANNEL = 64
    # output vs input scaling ratio, e.g., input_h // feature_h
    cfg.MODEL.SCALE = 4.0
    # topk detection results will be kept after inference
    cfg.MODEL.TOPK = 100
    # model zoo root dir
    cfg.MODEL.ROOT = os.path.join('~', '.mxnet', 'models')

    cfg.DATA = CN()
    # dataset root
    cfg.DATA.ROOT = os.path.join('~', '.mxnet', 'datasets')
    # dataset type
    cfg.DATA.TYPE = 'coco'  # coco json style
    cfg.DATA.TRAIN_SPLITS = 'instances_train2017'
    cfg.DATA.VALID_SPLITS = 'instances_val2017'
    cfg.DATA.VALID_SKIP_EMPTY = False

    cfg.SYSTEM = CN()
    # GPU IDs to use in the experiment
    cfg.SYSTEM.GPUS = (0, 1, 2, 3, 4, 5, 6, 7)
    # Number of workers for doing things
    cfg.SYSTEM.NUM_WORKERS = 16

    cfg.TRAIN = CN()
    # whether load the imagenet pre-trained base
    cfg.TRAIN.PRETRAINED_BASE = True
    # whether finetune from other models, expect a dataset name, e.g., 'coco'
    cfg.TRAIN.TRANSFER_FROM = None
    # A very important hyperparameter
    cfg.TRAIN.BATCH_SIZE = 32
    # training data shape
    cfg.TRAIN.DATA_SHAPE = 512
    # epochs
    cfg.TRAIN.EPOCHS = 140
    # Resume from previously saved parameters if not empty
    cfg.TRAIN.RESUME = ''
    # Starting epoch for resuming, default is 0 for new training
    cfg.TRAIN.START_EPOCH = 0
    # Learning rate
    cfg.TRAIN.LR = 1.25e-4
    # decay rate of learning rate.
    cfg.TRAIN.LR_DECAY = 0.1
    # epochs at which learning rate decays
    cfg.TRAIN.LR_DECAY_EPOCH = (90, 120)
    # learning rate scheduler mode. options are step, poly and cosine
    cfg.TRAIN.LR_MODE = 'step'
    # starting warmup learning rate.
    cfg.TRAIN.WARMUP_LR = 0.0
    # number of warmup epochs
    cfg.TRAIN.WARMUP_EPOCHS = 0
    # SGD momentum, default is 0.9
    cfg.TRAIN.MOMENTUM = 0.9
    # Weight decay, default is 1e-4
    cfg.TRAIN.WD = 1e-4
    # random seed
    cfg.TRAIN.SEED = 123
    # Loss weight for width/height
    cfg.TRAIN.WH_WEIGHT = 0.1
    # Center regression loss weight
    cfg.TRAIN.CENTER_REG_WEIGHT = 1.0
    # Saving parameters epoch interval, best model will always be saved
    cfg.TRAIN.SAVE_INTERVAL = 10

    cfg.VALID = CN()
    # enable flip test
    cfg.VALID.FLIP_TEST = True
    # nms
    cfg.VALID.NMS_THRESH = 0  # 0 means disable
    # pre nms topk
    cfg.VALID.NMS_TOPK = 400
    # post nms topk
    cfg.VALID.POST_NMS = 100
    # Epoch interval for validation, increase the number will reduce the training time if validation is slow
    cfg.VALID.INTERVAL = 10
    return cfg.clone()

@set_default(get_center_net_defaults)
class CenterNetEstimator(BaseEstimator):
    """CenterNet Estimator."""
    def __init__(self, config=None, reporter=None, logdir=None):
        super(CenterNetEstimator, self).__init__(config, reporter, logdir)

    def fit(self, train_data):
        if isinstance(train_data, str) and train_data.lower() == 'coco':
            self.config.DATA.TRAIN.TYPE = 'coco'
            self.config.DATA.TRAIN.SPLITS = 'instances_train2017'
            self.config.DATA.VALID.TYPE = 'coco'
            self.config.DATA.VALID.SPLITS = 'instances_val2017'
        elif isinstance(train_data, str) and train_data.lower() == 'voc':
            self.config.DATA.TRAIN.TYPE = 'voc'
            self.config.DATA.TRAIN.SPLITS = [(2007, 'trainval'), (2012, 'trainval')]
            self.config.DATA.VALID.TYPE = 'voc'
            self.config.DATA.VALID.SPLITS = [(2007, 'test')]
        else:
            raise NotImplementedError('CenterNetEstimator.fit now only supports coco and voc')
        self.finalize_config()

        # dataset
        if self.config.DATA.TYPE == 'coco':
            train_dataset = COCODetection(root=os.path.join(self.config.DATA.ROOT, 'coco'),
                                          splits=self.config.DATA.TRAIN_SPLITS)
            val_dataset = COCODetection(root=os.path.join(self.config.DATA.ROOT, 'coco'),
                                        splits=self.config.DATA.VALID_SPLITS,
                                        skip_empty=self.config.DATA.VALID_SKIP_EMPTY)
            val_metric = COCODetectionMetric(
                val_dataset, os.path.join(self.logdir, 'coco_eval'), cleanup=True,
                data_shape=(self.config.TRAIN.DATA_SHAPE, self.config.TRAIN.DATA_SHAPE),
                post_affine=get_post_transform)
        elif self.config.DATA.TYPE == 'voc':
            train_dataset = VOCDetection(splits=self.config.DATA.TRAIN_SPLITS)
            val_dataset = VOCDetection(splits=self.config.DATA.VALID_SPLITS)
            val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
        else:
            raise NotImplementedError('Invalid dataset type: {}'.format(self.config.DATA.TYPE))

        # network
        ctx = [mx.gpu(int(i)) for i in self.config.SYSTEM.GPUS]
        ctx = ctx if ctx else [mx.cpu()]

        net_name = '_'.join(('center_net', self.config.MODEL.BASE_NETWORK, self.config.DATA.TYPE))
        heads = OrderedDict([
            ('heatmap', {'num_output': train_dataset.num_class, 'bias': self.config.MODEL.HEADS.BIAS}),
            ('wh', {'num_output': self.config.MODEL.HEADS.WH_OUTPUTS}),
            ('reg', {'num_output': self.config.MODEL.HEADS.REG_OUTPUTS})
            ])
        net = get_center_net(self.config.MODEL.BASE_NETWORK,
                             self.config.DATA.TYPE,
                             base_network=self.config.MODEL.BASE_NETWORK,
                             heads=heads,
                             head_conv_channel=self.config.MODEL.HEADS.HEAD_CONV_CHANNEL,
                             classes=train_dataset.classes,
                             scale=self.config.MODEL.SCALE,
                             topk=self.config.MODEL.TOPK,
                             pretrained_base=self.config.TRAIN.PRETRAINED_BASE,
                             norm_layer=gluon.nn.BatchNorm)
        if self.config.TRAIN.RESUME.strip():
            net.load_parameters(self.config.TRAIN.RESUME.strip())
        else:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                net.initialize()

        # dataloader
        _random.seed(self.config.TRAIN.SEED)
        batch_size = self.config.TRAIN.BATCH_SIZE
        width, height = self.config.TRAIN.DATA_SHAPE, self.config.TRAIN.DATA_SHAPE
        num_class = len(train_dataset.classes)
        batchify_fn = Tuple([Stack() for _ in range(6)])  # stack image, cls_targets, box_targets
        train_loader = gluon.data.DataLoader(
            train_dataset.transform(CenterNetDefaultTrainTransform(
                width, height, num_class=num_class, scale_factor=net.scale)),
            batch_size, True, batchify_fn=batchify_fn, last_batch='rollover',
            num_workers=self.config.SYSTEM.NUM_WORKERS)
        val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
        val_loader = gluon.data.DataLoader(
            val_dataset.transform(CenterNetDefaultValTransform(width, height)),
            batch_size, False, batchify_fn=val_batchify_fn, last_batch='keep',
            num_workers=self.config.SYSTEM.NUM_WORKERS)

    def _train(self, net, train_data, val_data, eval_metric, ctx):
        net.collect_params().reset_ctx(ctx)
        # lr decay policy
        lr_decay = float(self.config.TRAIN.LR_DECAY)
        lr_steps = sorted(self.config.TRAIN.LR_DECAY_EPOCH)
        lr_decay_epoch = [e - self.config.TRAIN.WARMUP_EPOCHS for e in lr_steps]
        num_batches = len(train_data) // self.config.TRAIN.BATCH_SIZE
        lr_scheduler = LRSequential([
            LRScheduler('linear', base_lr=0, target_lr=self.config.TRAIN.LR,
                        nepochs=self.config.TRAIN.WARMUP_EPOCHS, iters_per_epoch=num_batches),
            LRScheduler(self.config.TRAIN.LR_MODE, base_lr=self.config.TRAIN.LR,
                        nepochs=self.config.TRAIN.EPOCHS - self.config.TRAIN.WARMUP_EPOCHS,
                        iters_per_epoch=num_batches,
                        step_epoch=lr_decay_epoch,
                        step_factor=self.config.TRAIN.LR_DECAY, power=2),
        ])

        for k, v in net.collect_params('.*bias').items():
            v.wd_mult = 0.0
        trainer = gluon.Trainer(
                    net.collect_params(), 'adam',
                    {'learning_rate': self.config.TRAIN.LR, 'wd': self.config.TRAIN.WD,
                     'lr_scheduler': lr_scheduler})

        heatmap_loss = HeatmapFocalLoss(from_logits=True)
        wh_loss = MaskedL1Loss(weight=args.wh_weight)
        center_reg_loss = MaskedL1Loss(weight=args.center_reg_weight)
        heatmap_loss_metric = mx.metric.Loss('HeatmapFocal')
        wh_metric = mx.metric.Loss('WHL1')
        center_reg_metric = mx.metric.Loss('CenterRegL1')
