"""CenterNet Estimator"""
# pylint: disable=unused-variable,missing-function-docstring
import os
import time
import warnings
from collections import OrderedDict

import pandas as pd
import mxnet as mx
from mxnet import autograd
from mxnet import gluon
from sacred import Experiment, Ingredient

from ..base_estimator import BaseEstimator, set_default
from ..common import logging
from ...data.coco_detection import coco_detection, load_coco_detection
from ....data.batchify import Tuple, Stack, Pad
from ....data.transforms.presets.center_net import CenterNetDefaultTrainTransform
from ....data.transforms.presets.center_net import CenterNetDefaultValTransform, get_post_transform
from ....loss import MaskedL1Loss, HeatmapFocalLoss
from ....model_zoo.center_net import get_center_net, get_base_network
from ....utils import LRScheduler, LRSequential
from ....utils.metrics import VOCMApMetric

__all__ = ['CenterNetEstimator']

center_net = Ingredient('center_net')
train = Ingredient('train')
validation = Ingredient('validation')


@center_net.config
def center_net_default():
    base_network = 'dla34_deconv'  # base feature network
    heads = {
        'bias': -2.19,  # use bias = -log((1 - 0.1) / 0.1)
        'wh_outputs': 2,  # wh head channel
        'reg_outputs': 2,  # regression head channel
        'head_conv_channel': 64,  # additional conv channel
    }
    scale = 4.0  # output vs input scaling ratio, e.g., input_h // feature_h
    topk = 100  # topk detection results will be kept after inference
    root = os.path.expanduser(os.path.join('~', '.mxnet', 'models'))  # model zoo root dir
    wh_weight = 0.1  # Loss weight for width/height
    center_reg_weight = 1.0  # Center regression loss weight
    data_shape = (512, 512)


@train.config
def train_config():
    gpus = (0, 1, 2, 3, 4, 5, 6, 7)  # gpu individual ids, not necessarily consecutive
    pretrained_base = True  # whether load the imagenet pre-trained base
    batch_size = 128
    epochs = 140
    lr = 1.25e-4  # learning rate
    lr_decay = 0.1  # decay rate of learning rate.
    lr_decay_epoch = (90, 120)  # epochs at which learning rate decays
    lr_mode = 'step'  # learning rate scheduler mode. options are step, poly and cosine
    warmup_lr = 0.0  # starting warmup learning rate.
    warmup_epochs = 0  # number of warmup epochs
    num_workers = 16  # cpu workers, the larger the more processes used
    resume = ''
    auto_resume = True  # try to automatically resume last trial if config is default
    start_epoch = 0
    momentum = 0.9  # SGD momentum
    wd = 1e-4  # weight decay
    save_interval = 10  # Saving parameters epoch interval, best model will always be saved
    log_interval = 100  # logging interval


@validation.config
def valid_config():
    flip_test = True  # use flip in validation test
    nms_thresh = 0  # 0 means disable
    nms_topk = 400  # pre nms topk
    post_nms = 100  # post nms topk
    num_workers = 32  # cpu workers, the larger the more processes used
    batch_size = 32  # validation batch size
    interval = 10  # validation epoch interval, for slow validations


ex = Experiment('center_net_default',
                ingredients=[logging, coco_detection, train, validation, center_net])


@ex.config
def default_configs():
    dataset = 'coco'


@set_default(ex)
class CenterNetEstimator(BaseEstimator):
    """Estimator implementation for CenterNet.

    Parameters
    ----------
    config : dict
        Config in nested dict.
    logger : logging.Logger
        Optional logger for this estimator, can be `None` when default setting is used.
    reporter : callable
        The reporter for metric checkpointing.

    Attributes
    ----------
    _logger : logging.Logger
        The customized/default logger for this estimator.
    _logdir : str
        The temporary dir for logs.
    _cfg : ConfigDict
        The configurations.

    """
    def __init__(self, config, logger=None, reporter=None):
        super(CenterNetEstimator, self).__init__(config, logger, reporter=reporter, name=None)
        self.last_train = None

    def _fit(self, train_data, val_data):
        self._best_map = 0
        self.epoch = 0
        if not isinstance(train_data, pd.DataFrame):
            self.last_train = len(train_data)
        else:
            self.last_train = train_data
        self._init_trainer()
        self._resume_fit(train_data, val_data)

    def _resume_fit(self, train_data, val_data):
        if not self.classes or not self.num_class:
            raise ValueError('Unable to determine classes of dataset')
        train_dataset = train_data.to_mxnet()
        val_dataset = val_data.to_mxnet()
        val_metric = VOCMApMetric(class_names=self.classes)

        # dataloader
        batch_size = self._cfg.train.batch_size
        width, height = self._cfg.center_net.data_shape
        num_class = len(train_dataset.classes)
        batchify_fn = Tuple([Stack() for _ in range(6)])  # stack image, cls_targets, box_targets
        train_loader = gluon.data.DataLoader(
            train_dataset.transform(CenterNetDefaultTrainTransform(
                width, height, num_class=num_class, scale_factor=self.net.scale)),
            batch_size, True, batchify_fn=batchify_fn, last_batch='rollover',
            num_workers=self._cfg.train.num_workers)
        val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
        val_loader = gluon.data.DataLoader(
            val_dataset.transform(CenterNetDefaultValTransform(width, height)),
            self._cfg.validation.batch_size, False, batchify_fn=val_batchify_fn, last_batch='keep',
            num_workers=self._cfg.validation.num_workers)

        self._train_loop(train_loader, val_loader, val_metric)

    def _train_loop(self, train_data, val_data, val_metric):
        wh_loss = MaskedL1Loss(weight=self._cfg.center_net.wh_weight)
        heatmap_loss = HeatmapFocalLoss(from_logits=True)
        center_reg_loss = MaskedL1Loss(weight=self._cfg.center_net.center_reg_weight)
        heatmap_loss_metric = mx.metric.Loss('HeatmapFocal')
        wh_metric = mx.metric.Loss('WHL1')
        center_reg_metric = mx.metric.Loss('CenterRegL1')

        for self.epoch in range(max(self._cfg.train.start_epoch, self.epoch), self._cfg.train.epochs):
            wh_metric.reset()
            center_reg_metric.reset()
            heatmap_loss_metric.reset()
            tic = time.time()
            btic = time.time()
            self.net.hybridize()
            epoch = self.epoch

            for i, batch in enumerate(train_data):
                split_data = [
                    gluon.utils.split_and_load(batch[ind], ctx_list=self.ctx, batch_axis=0) for ind
                    in range(6)]
                data, heatmap_targets, wh_targets, wh_masks, center_reg_targets, center_reg_masks = split_data
                batch_size = self._cfg.train.batch_size
                with autograd.record():
                    sum_losses = []
                    heatmap_losses = []
                    wh_losses = []
                    center_reg_losses = []
                    wh_preds = []
                    center_reg_preds = []
                    for x, heatmap_target, wh_target, wh_mask, center_reg_target, center_reg_mask in zip(
                            *split_data):
                        heatmap_pred, wh_pred, center_reg_pred = self.net(x)
                        wh_preds.append(wh_pred)
                        center_reg_preds.append(center_reg_pred)
                        wh_losses.append(wh_loss(wh_pred, wh_target, wh_mask))
                        center_reg_losses.append(
                            center_reg_loss(center_reg_pred, center_reg_target, center_reg_mask))
                        heatmap_losses.append(heatmap_loss(heatmap_pred, heatmap_target))
                        curr_loss = heatmap_losses[-1] + wh_losses[-1] + center_reg_losses[-1]
                        sum_losses.append(curr_loss)
                    autograd.backward(sum_losses)
                self.trainer.step(len(sum_losses))  # step with # gpus

                heatmap_loss_metric.update(0, heatmap_losses)
                wh_metric.update(0, wh_losses)
                center_reg_metric.update(0, center_reg_losses)
                if self._cfg.train.log_interval and not (i + 1) % self._cfg.train.log_interval:
                    name2, loss2 = wh_metric.get()
                    name3, loss3 = center_reg_metric.get()
                    name4, loss4 = heatmap_loss_metric.get()
                    self._logger.info(
                        '[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, '
                        'LR={}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
                            epoch, i, batch_size / (time.time() - btic),
                            self.trainer.learning_rate, name2, loss2, name3, loss3, name4, loss4))
                btic = time.time()

            name2, loss2 = wh_metric.get()
            name3, loss3 = center_reg_metric.get()
            name4, loss4 = heatmap_loss_metric.get()
            self._logger.info(
                '[Epoch {}] Training cost: {:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
                    epoch, (time.time() - tic), name2, loss2, name3, loss3, name4, loss4))
            if (epoch % self._cfg.validation.interval == 0) or \
                    (self._cfg.train.save_interval and epoch % self._cfg.train.save_interval == 0) or \
                    (epoch == self._cfg.train.epochs - 1):
                # consider reduce the frequency of validation to save time
                map_name, mean_ap = self._evaluate(val_data, val_metric)
                val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
                self._logger.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
                current_map = float(mean_ap[-1])
                if current_map > self._best_map:
                    self._logger.info('[Epoch %d] Current best map: %f vs previous %f',
                        self.epoch, current_map, self._best_map)
                    self._best_map = current_map

    def _evaluate(self, val_data, eval_metric):
        """Test on validation dataset."""
        eval_metric.reset()
        self.net.flip_test = self._cfg.validation.flip_test
        mx.nd.waitall()
        self.net.hybridize()
        for batch in val_data:
            data = gluon.utils.split_and_load(batch[0], ctx_list=self.ctx, batch_axis=0,
                                              even_split=False)
            label = gluon.utils.split_and_load(batch[1], ctx_list=self.ctx, batch_axis=0,
                                               even_split=False)
            det_bboxes = []
            det_ids = []
            det_scores = []
            gt_bboxes = []
            gt_ids = []
            gt_difficults = []
            for x, y in zip(data, label):
                # get prediction results
                ids, scores, bboxes = self.net(x)
                det_ids.append(ids)
                det_scores.append(scores)
                # clip to image size
                det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
                # split ground truths
                gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
                gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
                gt_difficults.append(
                    y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)

            # update metric
            eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids,
                                     gt_difficults)
        return eval_metric.get()

    def _init_network(self):
        if not self.num_class:
            raise ValueError('Unable to create network when `num_class` is unknown. \
                It should be inferred from dataset or resumed from saved states.')
        assert len(self.classes) == self.num_class
        # network
        ctx = [mx.gpu(int(i)) for i in self._cfg.train.gpus]
        ctx = ctx if ctx else [mx.cpu()]
        self.ctx = ctx
        net_name = '_'.join(('center_net', self._cfg.center_net.base_network, self._cfg.dataset))
        heads = OrderedDict([
            ('heatmap',
             {'num_output': self.num_class, 'bias': self._cfg.center_net.heads.bias}),
            ('wh', {'num_output': self._cfg.center_net.heads.wh_outputs}),
            ('reg', {'num_output': self._cfg.center_net.heads.reg_outputs})])
        base_network = get_base_network(self._cfg.center_net.base_network,
                                        pretrained=self._cfg.train.pretrained_base)
        net = get_center_net(self._cfg.center_net.base_network,
                             self.dataset,
                             base_network=base_network,
                             heads=heads,
                             head_conv_channel=self._cfg.center_net.heads.head_conv_channel,
                             classes=self.classes,
                             scale=self._cfg.center_net.scale,
                             topk=self._cfg.center_net.topk,
                             norm_layer=gluon.nn.BatchNorm)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            net.initialize()
        self.net = net
        for k, v in self.net.collect_params('.*bias').items():
            v.wd_mult = 0.0
        self.net.collect_params().reset_ctx(self.ctx)

    def _init_trainer(self):
        if self.last_train is None:
            raise RuntimeError('Cannot init trainer without knowing the size of training data')
        if isinstance(self.last_train, pd.DataFrame):
            train_size = len(self.last_train)
        elif isinstance(self.last_train, int):
            train_size = self.last_train
        else:
            raise ValueError("Unknown type of self.last_train: {}".format(type(self.last_train)))
        lr_decay = float(self._cfg.train.lr_decay)
        lr_steps = sorted(self._cfg.train.lr_decay_epoch)
        lr_decay_epoch = [e - self._cfg.train.warmup_epochs for e in lr_steps]
        num_batches = train_size // self._cfg.train.batch_size
        lr_scheduler = LRSequential([
            LRScheduler('linear', base_lr=0, target_lr=self._cfg.train.lr,
                        nepochs=self._cfg.train.warmup_epochs, iters_per_epoch=num_batches),
            LRScheduler(self._cfg.train.lr_mode, base_lr=self._cfg.train.lr,
                        nepochs=self._cfg.train.epochs - self._cfg.train.warmup_epochs,
                        iters_per_epoch=num_batches,
                        step_epoch=lr_decay_epoch,
                        step_factor=self._cfg.train.lr_decay, power=2),
        ])
        self.trainer = gluon.Trainer(
            self.net.collect_params(), 'adam',
            {'learning_rate': self._cfg.train.lr, 'wd': self._cfg.train.wd,
             'lr_scheduler': lr_scheduler})

@ex.automain
def main(_config, _log):
    # main is the commandline entry for user w/o coding
    c = CenterNetEstimator(_config, _log)
    c.fit()