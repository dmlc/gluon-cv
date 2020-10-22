"""SSD Estimator."""
# pylint: disable=logging-format-interpolation
import os
import time
import warnings

import pandas as pd
import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
from mxnet.contrib import amp

from .... import utils as gutils
from ....utils.metrics.voc_detection import VOC07MApMetric
from ....utils.metrics.coco_detection import COCODetectionMetric
from ....model_zoo import get_model
from ....model_zoo import custom_ssd
from ....data.transforms import presets
from ....loss import SSDMultiBoxLoss
from .utils import _get_dataset, _get_dataloader, _get_dali_dataset, _get_dali_dataloader, _save_params
from ..base_estimator import BaseEstimator, set_default
from .default import ex

try:
    import horovod.mxnet as hvd
except ImportError:
    hvd = None

try:
    dali_found = True
except ImportError:
    dali_found = False

__all__ = ['SSDEstimator']


@set_default(ex)
class SSDEstimator(BaseEstimator):
    """Estimator implementation for SSD.

    Parameters
    ----------
    config : dict
        Config in nested dict.
    logger : logging.Logger, default is None
        Optional logger for this estimator, can be `None` when default setting is used.
    reporter : callable, default is None
        If set, use reporter callback to report the metrics of the current estimator.

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
        super(SSDEstimator, self).__init__(config, logger, reporter, name=self.__class__.__name__)

        if self._cfg.ssd.amp:
            amp.init()
        if self._cfg.horovod:
            hvd.init()
        # fix seed for mxnet, numpy and python builtin random generator.
        gutils.random.seed(self._cfg.train.seed)

    def _fit(self, train_data, val_data):
        """Fit SSD model."""
        self._best_map = 0
        self.epoch = 0
        self.net.collect_params().reset_ctx(self.ctx)
        self._init_trainer()
        self._resume_fit(train_data, val_data)

    def _resume_fit(self, train_data, val_data):
        if not self.classes or not self.num_class:
            raise ValueError('Unable to determine classes of dataset')

        # dataset
        devices = [int(i) for i in self._cfg.gpus]
        # if self._cfg.train.dali:
        #     if not dali_found:
        #         raise SystemExit("DALI not found, please check if you installed it correctly.")
        #     train_dataset, val_dataset, eval_metric = _get_dali_dataset(self._cfg.dataset, devices, self._cfg)
        # else:
        #     train_dataset, val_dataset, eval_metric = _get_dataset(self._cfg.dataset, self._cfg)
        train_dataset = train_data.to_mxnet()
        val_dataset = val_data.to_mxnet()

        # dataloader
        if self._cfg.train.dali:
            if not dali_found:
                raise SystemExit("DALI not found, please check if you installed it correctly.")
            train_loader, val_loader = _get_dali_dataloader(
                self.async_net, train_dataset, val_dataset, self._cfg.ssd.data_shape,
                self._cfg.train.batch_size, self._cfg.num_workers,
                devices, self.ctx[0], self._cfg.horovod)
        else:
            self.batch_size = self._cfg.train.batch_size // hvd.size() \
                if self._cfg.horovod else self._cfg.train.batch_size
            train_loader, val_loader = _get_dataloader(
                self.async_net, train_dataset, val_dataset, self._cfg.ssd.data_shape,
                self.batch_size, self._cfg.num_workers)

        self._train_loop(train_loader, val_loader)

    def _train_loop(self, train_data, val_data):
        # loss and metric
        mbox_loss = SSDMultiBoxLoss()
        ce_metric = mx.metric.Loss('CrossEntropy')
        smoothl1_metric = mx.metric.Loss('SmoothL1')

        # lr decay policy
        lr_decay = float(self._cfg.train.lr_decay)
        lr_steps = sorted([float(ls) for ls in self._cfg.train.lr_decay_epoch])

        self._logger.info('Start training from [Epoch {}]'.format(self._cfg.train.start_epoch))
        best_map = [0]

        self.net.collect_params().reset_ctx(self.ctx)
        for self.epoch in range(max(self._cfg.train.start_epoch, self.epoch), self._cfg.train.epochs):
            epoch = self.epoch
            while lr_steps and epoch >= lr_steps[0]:
                new_lr = self.trainer.learning_rate * lr_decay
                lr_steps.pop(0)
                self.trainer.set_learning_rate(new_lr)
                self._logger.info("[Epoch {}] Set learning rate to {}".format(epoch, new_lr))
            ce_metric.reset()
            smoothl1_metric.reset()
            tic = time.time()
            btic = time.time()
            self.net.hybridize(static_alloc=True, static_shape=True)

            for i, batch in enumerate(train_data):
                if self._cfg.train.dali:
                    # dali iterator returns a mxnet.io.DataBatch
                    data = [d.data[0] for d in batch]
                    box_targets = [d.label[0] for d in batch]
                    cls_targets = [nd.cast(d.label[1], dtype='float32') for d in batch]
                else:
                    data = gluon.utils.split_and_load(batch[0], ctx_list=self.ctx, batch_axis=0)
                    cls_targets = gluon.utils.split_and_load(batch[1], ctx_list=self.ctx, batch_axis=0)
                    box_targets = gluon.utils.split_and_load(batch[2], ctx_list=self.ctx, batch_axis=0)

                with autograd.record():
                    cls_preds = []
                    box_preds = []
                    for x in data:
                        cls_pred, box_pred, _ = self.net(x)
                        cls_preds.append(cls_pred)
                        box_preds.append(box_pred)
                    sum_loss, cls_loss, box_loss = mbox_loss(
                        cls_preds, box_preds, cls_targets, box_targets)
                    if self._cfg.ssd.amp:
                        with amp.scale_loss(sum_loss, self.trainer) as scaled_loss:
                            autograd.backward(scaled_loss)
                    else:
                        autograd.backward(sum_loss)
                # since we have already normalized the loss, we don't want to normalize
                # by batch-size anymore
                self.trainer.step(1)

                if not self._cfg.horovod or hvd.rank() == 0:
                    local_batch_size = int(self._cfg.train.batch_size // (hvd.size() if self._cfg.horovod else 1))
                    ce_metric.update(0, [l * local_batch_size for l in cls_loss])
                    smoothl1_metric.update(0, [l * local_batch_size for l in box_loss])
                    if self._cfg.train.log_interval and not (i + 1) % self._cfg.train.log_interval:
                        name1, loss1 = ce_metric.get()
                        name2, loss2 = smoothl1_metric.get()
                        self._logger.info('[Epoch %d][Batch %d], Speed: %f samples/sec, %s=%f, %s=%f',
                                          epoch, i, self._cfg.train.batch_size/(time.time()-btic), name1, loss1, name2, loss2)
                    btic = time.time()

            if not self._cfg.horovod or hvd.rank() == 0:
                name1, loss1 = ce_metric.get()
                name2, loss2 = smoothl1_metric.get()
                self._logger.info('[Epoch %d] Training cost: %f, %s=%f, %s=%f',
                                  epoch, (time.time()-tic), name1, loss1, name2, loss2)
                if (epoch % self._cfg.validation.val_interval == 0) or \
                    (self._cfg.save_interval and epoch % self._cfg.save_interval == 0):
                    # consider reduce the frequency of validation to save time
                    map_name, mean_ap = self._evaluate(val_data)
                    val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
                    self._logger.info('[Epoch %d] Validation: \n%s', epoch, str(val_msg))
                    current_map = float(mean_ap[-1])
                    if current_map > self._best_map:
                        self._logger.info('[Epoch %d] Current best map: %f vs previous %f',
                                          self.epoch, current_map, self._best_map)
                        self._best_map = current_map
                if self._reporter:
                    self._reporter(epoch=epoch, map_reward=current_map)

    def _evaluate(self, val_data):
        """Evaluate on validation dataset."""
        eval_metric = VOC07MApMetric(iou_thresh=0.5, class_names=self.classes)

        # if self._cfg.dataset.lower() == 'voc' or 'voc_tiny':
        #     eval_metric = VOC07MApMetric(iou_thresh=0.5, class_names=self.classes)
        # elif self._cfg.dataset.lower() == 'coco':
        #     eval_metric = COCODetectionMetric(
        #         self.val_dataset, os.path.join(self._cfg.logdir, self._cfg.save_prefix + '_eval'), cleanup=True,
        #         data_shape=(self._cfg.ssd.data_shape, self._cfg.ssd.data_shape))

        # set nms threshold and topk constraint
        self.net.set_nms(nms_thresh=0.45, nms_topk=400)
        self.net.collect_params().reset_ctx(self.ctx)
        self.net.hybridize(static_alloc=True, static_shape=True)
        for batch in val_data:
            data = gluon.utils.split_and_load(batch[0], ctx_list=self.ctx, batch_axis=0, even_split=False)
            label = gluon.utils.split_and_load(batch[1], ctx_list=self.ctx, batch_axis=0, even_split=False)
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
                gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)
            # update metric
            eval_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
        return eval_metric.get()

    def _predict(self, x):
        """Predict an individual example."""
        x, _ = presets.ssd.transform_test(x, short=512)
        x = x.as_in_context(self.ctx[0])
        ids, scores, bboxes = [xx[0].asnumpy() for xx in self.net(x)]
        return ids, scores, bboxes

    def _init_network(self):
        if not self.num_class:
            raise ValueError('Unable to create network when `num_class` is unknown. \
                It should be inferred from dataset or resumed from saved states.')
        assert len(self.classes) == self.num_class

        # training contexts
        if self._cfg.horovod:
            self.ctx = [mx.gpu(hvd.local_rank())]
        else:
            ctx = [mx.gpu(int(i)) for i in self._cfg.gpus]
            self.ctx = ctx if ctx else [mx.cpu()]

        # network
        # net_name = '_'.join(('ssd', str(self._cfg.ssd.data_shape), self._cfg.ssd.backbone, self._cfg.dataset))
        # self._cfg.save_prefix += net_name

        if self._cfg.ssd.transfer is not None:
            assert isinstance(self._cfg.ssd.transfer, str)
            self._logger.info(
                f'Using transfer learning from {self._cfg.ssd.transfer}, the other network parameters are ignored.')
            if self._cfg.ssd.syncbn and len(self.ctx) > 1:
                self.net = get_model(self._cfg.ssd.transfer, pretrained=True, norm_layer=gluon.contrib.nn.SyncBatchNorm,
                                     norm_kwargs={'num_devices': len(self.ctx)})
                self.async_net = get_model(self._cfg.ssd.transfer, pretrained=True)  # used by cpu worker
                self.net.reset_class(self.classes,
                                     reuse_weights=[cname for cname in self.classes if cname in self.net.classes])
            else:
                self.net = get_model(self._cfg.ssd.transfer, pretrained=True, norm_layer=gluon.nn.BatchNorm)
                self.async_net = get_model(self._cfg.ssd.transfer, pretrained=True, norm_layer=gluon.nn.BatchNorm)
                self.net.reset_class(self.classes,
                                     reuse_weights=[cname for cname in self.classes if cname in self.net.classes])
        # elif self._cfg.ssd.custom_model:
        else:
            if self._cfg.ssd.syncbn and len(self.ctx) > 1:
                self.net = custom_ssd(base_network_name=self._cfg.ssd.backbone,
                                      base_size=self._cfg.ssd.data_shape,
                                      filters=self._cfg.ssd.filters,
                                      sizes=self._cfg.ssd.sizes,
                                      ratios=self._cfg.ssd.ratios,
                                      steps=self._cfg.ssd.steps,
                                      classes=self.classes,
                                      dataset='auto',
                                      pretrained_base=True,
                                      norm_layer=gluon.contrib.nn.SyncBatchNorm,
                                      norm_kwargs={'num_devices': len(self.ctx)})
                self.async_net = custom_ssd(base_network_name=self._cfg.ssd.backbone,
                                            base_size=self._cfg.ssd.data_shape,
                                            filters=self._cfg.ssd.filters,
                                            sizes=self._cfg.ssd.sizes,
                                            ratios=self._cfg.ssd.ratios,
                                            steps=self._cfg.ssd.steps,
                                            classes=self.classes,
                                            dataset='auto',
                                            pretrained_base=False)
            else:
                self.net = custom_ssd(base_network_name=self._cfg.ssd.backbone,
                                      base_size=self._cfg.ssd.data_shape,
                                      filters=self._cfg.ssd.filters,
                                      sizes=self._cfg.ssd.sizes,
                                      ratios=self._cfg.ssd.ratios,
                                      steps=self._cfg.ssd.steps,
                                      classes=self.classes,
                                      dataset=self._cfg.dataset,
                                      pretrained_base=True,
                                      norm_layer=gluon.nn.BatchNorm)
                self.async_net = self.net

        # if self._cfg.resume.strip():
        #     self.net.load_parameters(self._cfg.resume.strip())
        #     self.async_net.load_parameters(self._cfg.resume.strip())
        # else:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.net.initialize()
            self.async_net.initialize()
            # needed for net to be first gpu when using AMP
            self.net.collect_params().reset_ctx(self.ctx[0])

    def _init_trainer(self):
        if self._cfg.horovod:
            hvd.broadcast_parameters(self.net.collect_params(), root_rank=0)
            self.trainer = hvd.DistributedTrainer(
                self.net.collect_params(), 'sgd',
                {'learning_rate': self._cfg.train.lr, 'wd': self._cfg.train.wd,
                 'momentum': self._cfg.train.momentum})
        else:
            self.trainer = gluon.Trainer(
                self.net.collect_params(), 'sgd',
                {'learning_rate': self._cfg.train.lr, 'wd': self._cfg.train.wd,
                 'momentum': self._cfg.train.momentum},
                update_on_kvstore=(False if self._cfg.ssd.amp else None))

        if self._cfg.ssd.amp:
            amp.init_trainer(self.trainer)


@ex.automain
def main(_config, _log):
    # main is the commandline entry for user w/o coding
    c = SSDEstimator(_config, _log)
    c.fit()