"""YOLO Estimator."""
# pylint: disable=logging-format-interpolation
import os
import logging
import time
import warnings
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet import autograd
from mxnet.contrib import amp

from .... import utils as gutils
from ....data.transforms import presets
from ....model_zoo import get_model
from ....model_zoo import custom_yolov3
from ....utils import LRScheduler, LRSequential

from ..base_estimator import BaseEstimator, set_default
from .utils import _get_dataset, _get_dataloader, _save_params

try:
    import horovod.mxnet as hvd
except ImportError:
    hvd = None

from .default import ex

__all__ = ['YOLOEstimator']


@set_default(ex)
class YOLOEstimator(BaseEstimator):
    """Estimator implementation for YOLOv3.

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
        super(YOLOEstimator, self).__init__(config, logger, reporter)

        if self._cfg.yolo3.amp:
            amp.init()

        if self._cfg.horovod:
            if hvd is None:
                raise SystemExit("Horovod not found, please check if you installed it correctly.")
            hvd.init()

        # fix seed for mxnet, numpy and python builtin random generator.
        gutils.random.seed(self._cfg.train.seed)

        # training contexts
        if self._cfg.horovod:
            self.ctx = [mx.gpu(hvd.local_rank())]
        else:
            ctx = [mx.gpu(int(i)) for i in self._cfg.gpus]
            self.ctx = ctx if ctx else [mx.cpu()]

        # training dataset
        self.train_dataset, self.val_dataset, self.eval_metric = _get_dataset(self._cfg.dataset, self._cfg)

        # network
        net_name = '_'.join(('yolo3', self._cfg.yolo3.backbone, self._cfg.dataset))
        self._cfg.save_prefix += net_name

        if self._cfg.yolo3.custom_model:
            classes = self.train_dataset.CLASSES
            # use sync bn if specified
            if self._cfg.yolo3.syncbn and len(self.ctx) > 1:
                self.net = custom_yolov3(base_network_name=self._cfg.yolo3.backbone,
                                         filters=self._cfg.yolo3.filters,
                                         anchors=self._cfg.yolo3.anchors,
                                         strides=self._cfg.yolo3.strides,
                                         classes=classes,
                                         dataset=self._cfg.dataset,
                                         pretrained_base=True,
                                         norm_layer=gluon.contrib.nn.SyncBatchNorm,
                                         norm_kwargs={'num_devices': len(self.ctx)})
                self.async_net = custom_yolov3(base_network_name=self._cfg.yolo3.backbone,
                                               filters=self._cfg.yolo3.filters,
                                               anchors=self._cfg.yolo3.anchors,
                                               strides=self._cfg.yolo3.strides,
                                               classes=classes,
                                               dataset=self._cfg.dataset,
                                               pretrained_base=False)
            else:
                self.net = custom_yolov3(base_network_name=self._cfg.yolo3.backbone,
                                         filters=self._cfg.yolo3.filters,
                                         anchors=self._cfg.yolo3.anchors,
                                         strides=self._cfg.yolo3.strides,
                                         classes=classes,
                                         dataset=self._cfg.dataset,
                                         pretrained_base=True)
                self.async_net = self.net
        else:
            # use sync bn if specified
            if self._cfg.yolo3.syncbn and len(self.ctx) > 1:
                self.net = get_model(net_name, pretrained_base=True, norm_layer=gluon.contrib.nn.SyncBatchNorm,
                                     norm_kwargs={'num_devices': len(self.ctx)})
                self.async_net = get_model(net_name, pretrained_base=False)  # used by cpu worker
            else:
                self.net = get_model(net_name, pretrained_base=True)
                self.async_net = self.net

        if self._cfg.resume.strip():
            self.net.load_parameters(self._cfg.resume.strip())
            self.async_net.load_parameters(self._cfg.resume.strip())
        else:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                self.net.initialize()
                self.async_net.initialize()

        # training dataloader
        self.batch_size = self._cfg.train.batch_size // hvd.size() if self._cfg.horovod else self._cfg.train.batch_size
        self._train_data, self._val_data = _get_dataloader(
            self.async_net, self.train_dataset, self.val_dataset, self._cfg.yolo3.data_shape,
            self.batch_size, self._cfg.num_workers, self._cfg)

        # targets
        self.sigmoid_ce = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
        self.l1_loss = gluon.loss.L1Loss()

        # metrics
        self.obj_metrics = mx.metric.Loss('ObjLoss')
        self.center_metrics = mx.metric.Loss('BoxCenterLoss')
        self.scale_metrics = mx.metric.Loss('BoxScaleLoss')
        self.cls_metrics = mx.metric.Loss('ClassLoss')

        # set up logger
        logging.basicConfig()
        self._logger = logging.getLogger()
        self._logger.setLevel(logging.INFO)
        log_file_path = self._cfg.save_prefix + '_train.log'
        log_dir = os.path.dirname(log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        fh = logging.FileHandler(log_file_path)
        self._logger.addHandler(fh)
        self._logger.info(self._cfg)

    def _validate(self, val_data, ctx, eval_metric):
        """Test on validation dataset."""
        eval_metric.reset()
        # set nms threshold and topk constraint
        self.net.set_nms(nms_thresh=0.45, nms_topk=400)
        mx.nd.waitall()
        self.net.hybridize()
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

    def _fit(self):
        """Fit YOLO models."""
        self.net.collect_params().reset_ctx(self.ctx)
        if self._cfg.train.no_wd:
            for _, v in self.net.collect_params('.*beta|.*gamma|.*bias').items():
                v.wd_mult = 0.0

        if self._cfg.train.label_smooth:
            self.net._target_generator._label_smooth = True

        if self._cfg.train.lr_decay_period > 0:
            lr_decay_epoch = list(range(self._cfg.train.lr_decay_period,
                                        self._cfg.train.epochs,
                                        self._cfg.train.lr_decay_period))
        else:
            lr_decay_epoch = [int(i) for i in self._cfg.train.lr_decay_epoch]
        lr_decay_epoch = [e - self._cfg.train.warmup_epochs for e in lr_decay_epoch]
        num_batches = self._cfg.train.num_samples // self._cfg.train.batch_size
        lr_scheduler = LRSequential([
            LRScheduler('linear', base_lr=0, target_lr=self._cfg.train.lr,
                        nepochs=self._cfg.train.warmup_epochs, iters_per_epoch=num_batches),
            LRScheduler(self._cfg.train.lr_mode, base_lr=self._cfg.train.lr,
                        nepochs=self._cfg.train.epochs - self._cfg.train.warmup_epochs,
                        iters_per_epoch=num_batches,
                        step_epoch=lr_decay_epoch,
                        step_factor=self._cfg.train.lr_decay, power=2),
        ])

        if self._cfg.horovod:
            hvd.broadcast_parameters(self.net.collect_params(), root_rank=0)
            trainer = hvd.DistributedTrainer(
                self.net.collect_params(), 'sgd',
                {'wd': self._cfg.train.wd, 'momentum': self._cfg.train.momentum, 'lr_scheduler': lr_scheduler})
        else:
            trainer = gluon.Trainer(
                self.net.collect_params(), 'sgd',
                {'wd': self._cfg.train.wd, 'momentum': self._cfg.train.momentum, 'lr_scheduler': lr_scheduler},
                kvstore='local', update_on_kvstore=(False if self._cfg.yolo3.amp else None))

        if self._cfg.yolo3.amp:
            amp.init_trainer(trainer)

        self._logger.info('Start training from [Epoch %d]', self._cfg.train.start_epoch)
        best_map = [0]
        for epoch in range(self._cfg.train.start_epoch, self._cfg.train.epochs):
            if self._cfg.train.mixup:
                # TODO(zhreshold): more elegant way to control mixup during runtime
                try:
                    self._train_data._dataset.set_mixup(np.random.beta, 1.5, 1.5)
                except AttributeError:
                    self._train_data._dataset._data.set_mixup(np.random.beta, 1.5, 1.5)
                if epoch >= self._cfg.train.epochs - self._cfg.train.no_mixup_epochs:
                    try:
                        self._train_data._dataset.set_mixup(None)
                    except AttributeError:
                        self._train_data._dataset._data.set_mixup(None)

            tic = time.time()
            btic = time.time()
            mx.nd.waitall()
            self.net.hybridize()
            for i, batch in enumerate(self._train_data):
                data = gluon.utils.split_and_load(batch[0], ctx_list=self.ctx, batch_axis=0)
                # objectness, center_targets, scale_targets, weights, class_targets
                fixed_targets = [gluon.utils.split_and_load(batch[it], ctx_list=self.ctx, batch_axis=0) for it in
                                 range(1, 6)]
                gt_boxes = gluon.utils.split_and_load(batch[6], ctx_list=self.ctx, batch_axis=0)
                sum_losses = []
                obj_losses = []
                center_losses = []
                scale_losses = []
                cls_losses = []
                with autograd.record():
                    for ix, x in enumerate(data):
                        obj_loss, center_loss, scale_loss, cls_loss = self.net(x, gt_boxes[ix],
                                                                               *[ft[ix] for ft in fixed_targets])
                        sum_losses.append(obj_loss + center_loss + scale_loss + cls_loss)
                        obj_losses.append(obj_loss)
                        center_losses.append(center_loss)
                        scale_losses.append(scale_loss)
                        cls_losses.append(cls_loss)
                    if self._cfg.yolo3.amp:
                        with amp.scale_loss(sum_losses, trainer) as scaled_loss:
                            autograd.backward(scaled_loss)
                    else:
                        autograd.backward(sum_losses)
                trainer.step(self.batch_size)
                if (not self._cfg.horovod or hvd.rank() == 0):
                    self.obj_metrics.update(0, obj_losses)
                    self.center_metrics.update(0, center_losses)
                    self.scale_metrics.update(0, scale_losses)
                    self.cls_metrics.update(0, cls_losses)
                    if self._cfg.train.log_interval and not (i + 1) % self._cfg.train.log_interval:
                        name1, loss1 = self.obj_metrics.get()
                        name2, loss2 = self.center_metrics.get()
                        name3, loss3 = self.scale_metrics.get()
                        name4, loss4 = self.cls_metrics.get()
                        self._logger.info(
                            '[Epoch {}][Batch {}], LR: {:.2E}, Speed: {:.3f} samples/sec,'
                            ' {}={:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
                                epoch, i, trainer.learning_rate, self._cfg.train.batch_size / (time.time() - btic),
                                name1, loss1, name2, loss2, name3, loss3, name4, loss4))
                    btic = time.time()

            if (not self._cfg.horovod or hvd.rank() == 0):
                name1, loss1 = self.obj_metrics.get()
                name2, loss2 = self.center_metrics.get()
                name3, loss3 = self.scale_metrics.get()
                name4, loss4 = self.cls_metrics.get()
                self._logger.info('[Epoch {}] Training cost: {:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
                    epoch, (time.time() - tic), name1, loss1, name2, loss2, name3, loss3, name4, loss4))
                if not (epoch + 1) % self._cfg.validation.val_interval:
                    # consider reduce the frequency of validation to save time
                    map_name, mean_ap = self._validate(self._val_data, self.ctx, self.eval_metric)
                    val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
                    self._logger.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
                    current_map = float(mean_ap[-1])
                else:
                    current_map = 0.
                _save_params(self.net, best_map, current_map, epoch, self._cfg.save_interval, self._cfg.save_prefix)
                if self._reporter:
                    self._reporter(epoch=epoch, map_reward=current_map)

    def _evaluate(self):
        """Evaluate the current model on dataset."""
        self.net.collect_params().reset_ctx(self.ctx)
        return self._validate(self._val_data, self.ctx, self.eval_metric)

    def predict(self, x):
        """Predict an individual example.

        Parameters
        ----------
        x : file
            An image.
        """
        x, _ = presets.yolo.transform_test(x, short=512)
        x = x.as_in_context(self.ctx[0])
        ids, scores, bboxes = [xx[0].asnumpy() for xx in self.net(x)]
        return ids, scores, bboxes
