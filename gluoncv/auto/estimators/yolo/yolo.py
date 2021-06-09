"""YOLO Estimator."""
# pylint: disable=logging-format-interpolation,abstract-method,arguments-differ,logging-not-lazy
import os
import math
import time
import warnings

from PIL import Image
import pandas as pd
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet import autograd
from mxnet.contrib import amp

from .... import utils as gutils
from ....data.batchify import Tuple, Stack, Pad
from ....data.transforms.presets.yolo import YOLO3DefaultValTransform
from ....data.transforms.presets.yolo import load_test, transform_test
from ....model_zoo import get_model
from ....model_zoo import custom_yolov3
from ....utils.metrics.voc_detection import VOC07MApMetric, VOCMApMetric
from ....utils import LRScheduler, LRSequential

from ..base_estimator import BaseEstimator, set_default
from .utils import _get_dataloader
from ...data.dataset import ObjectDetectionDataset
from ..conf import _BEST_CHECKPOINT_FILE
from ..utils import EarlyStopperOnPlateau

try:
    import horovod.mxnet as hvd
except ImportError:
    hvd = None

from .default import YOLOv3Cfg

__all__ = ['YOLOv3Estimator']


@set_default(YOLOv3Cfg())
class YOLOv3Estimator(BaseEstimator):
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
    _cfg : autocfg.dataclass
        The configurations.
    """
    Dataset = ObjectDetectionDataset
    def __init__(self, config, logger=None, reporter=None):
        super(YOLOv3Estimator, self).__init__(config, logger, reporter)
        self.last_train = None
        if self._cfg.yolo3.amp:
            amp.init()
        if self._cfg.horovod:
            if hvd is None:
                raise SystemExit("Horovod not found, please check if you installed it correctly.")
            hvd.init()

    def _fit(self, train_data, val_data, time_limit=math.inf):
        """Fit YOLO3 model."""
        tic = time.time()
        self._best_map = 0
        self.epoch = 0
        self._time_elapsed = 0
        if max(self._cfg.train.start_epoch, self.epoch) >= self._cfg.train.epochs:
            return {'time', self._time_elapsed}
        if not isinstance(train_data, pd.DataFrame):
            self.last_train = len(train_data)
        else:
            self.last_train = train_data
        self.net.collect_params().reset_ctx(self.ctx)
        self._init_trainer()
        self._time_elapsed += time.time() - tic
        return self._resume_fit(train_data, val_data, time_limit=time_limit)

    def _resume_fit(self, train_data, val_data, time_limit=math.inf):
        tic = time.time()
        if max(self._cfg.train.start_epoch, self.epoch) >= self._cfg.train.epochs:
            return {'time', self._time_elapsed}
        if not self.classes or not self.num_class:
            raise ValueError('Unable to determine classes of dataset')

        # training dataset
        train_dataset = train_data.to_mxnet()
        val_dataset = val_data.to_mxnet()
        # training dataloader
        self.batch_size = self._cfg.train.batch_size // hvd.size() if self._cfg.horovod else self._cfg.train.batch_size
        train_loader, val_loader, train_eval_loader = _get_dataloader(
            self.async_net, train_dataset, val_dataset, self._cfg.yolo3.data_shape,
            self.batch_size, self._cfg.num_workers, self._cfg)

        if self._cfg.train.no_wd:
            for _, v in self.net.collect_params('.*beta|.*gamma|.*bias').items():
                v.wd_mult = 0.0
        if self._cfg.train.label_smooth:
            self.net._target_generator._label_smooth = True
        self._time_elapsed += time.time() - tic
        return self._train_loop(train_loader, val_loader, train_eval_loader, time_limit=time_limit)

    def _train_loop(self, train_data, val_data, train_eval_data, time_limit=math.inf):
        start_tic = time.time()
        # fix seed for mxnet, numpy and python builtin random generator.
        gutils.random.seed(self._cfg.train.seed)

        # metrics
        obj_metrics = mx.metric.Loss('ObjLoss')
        center_metrics = mx.metric.Loss('BoxCenterLoss')
        scale_metrics = mx.metric.Loss('BoxScaleLoss')
        cls_metrics = mx.metric.Loss('ClassLoss')
        trainer = self.trainer
        self._logger.info('Start training from [Epoch %d]', max(self._cfg.train.start_epoch, self.epoch))
        early_stopper = EarlyStopperOnPlateau(
            patience=self._cfg.train.early_stop_patience,
            min_delta=self._cfg.train.early_stop_min_delta,
            baseline_value=self._cfg.train.early_stop_baseline,
            max_value=self._cfg.train.early_stop_max_value)
        mean_ap = [-1]
        cp_name = ''
        self._time_elapsed += time.time() - start_tic
        for self.epoch in range(max(self._cfg.train.start_epoch, self.epoch), self._cfg.train.epochs):
            epoch = self.epoch
            if self._best_map >= 1.0:
                self._logger.info('[Epoch {}] Early stopping as mAP is reaching 1.0'.format(epoch))
                break
            should_stop, stop_message = early_stopper.get_early_stop_advice()
            if should_stop:
                self._logger.info('[Epoch {}] '.format(epoch) + stop_message)
                break
            tic = time.time()
            last_tic = time.time()
            if self._cfg.train.mixup:
                # TODO(zhreshold): more elegant way to control mixup during runtime
                try:
                    train_data._dataset.set_mixup(np.random.beta, 1.5, 1.5)
                except AttributeError:
                    train_data._dataset._data.set_mixup(np.random.beta, 1.5, 1.5)
                if epoch >= self._cfg.train.epochs - self._cfg.train.no_mixup_epochs:
                    try:
                        train_data._dataset.set_mixup(None)
                    except AttributeError:
                        train_data._dataset._data.set_mixup(None)

            mx.nd.waitall()
            self.net.hybridize()
            for i, batch in enumerate(train_data):
                btic = time.time()
                if self._time_elapsed > time_limit:
                    self._logger.warning(f'`time_limit={time_limit}` reached, exit early...')
                    return {'train_map': float(mean_ap[-1]), 'valid_map': self._best_map,
                            'time': self._time_elapsed, 'checkpoint': cp_name}
                data = gluon.utils.split_and_load(batch[0], ctx_list=self.ctx, batch_axis=0, even_split=False)
                # objectness, center_targets, scale_targets, weights, class_targets
                fixed_targets = [gluon.utils.split_and_load(batch[it], ctx_list=self.ctx,
                                                            batch_axis=0, even_split=False) for it in range(1, 6)]
                gt_boxes = gluon.utils.split_and_load(batch[6], ctx_list=self.ctx, batch_axis=0, even_split=False)
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
                    obj_metrics.update(0, obj_losses)
                    center_metrics.update(0, center_losses)
                    scale_metrics.update(0, scale_losses)
                    cls_metrics.update(0, cls_losses)
                    if self._cfg.train.log_interval and not (i + 1) % self._cfg.train.log_interval:
                        name1, loss1 = obj_metrics.get()
                        name2, loss2 = center_metrics.get()
                        name3, loss3 = scale_metrics.get()
                        name4, loss4 = cls_metrics.get()
                        self._logger.info(
                            '[Epoch {}][Batch {}], LR: {:.2E}, Speed: {:.3f} samples/sec,'
                            ' {}={:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
                                epoch, i, trainer.learning_rate, self._cfg.train.batch_size / (time.time() - last_tic),
                                name1, loss1, name2, loss2, name3, loss3, name4, loss4))
                        last_tic = time.time()
                    self._time_elapsed += time.time() - btic

            post_tic = time.time()
            if (not self._cfg.horovod or hvd.rank() == 0):
                name1, loss1 = obj_metrics.get()
                name2, loss2 = center_metrics.get()
                name3, loss3 = scale_metrics.get()
                name4, loss4 = cls_metrics.get()
                self._logger.info('[Epoch {}] Training cost: {:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
                    epoch, (time.time() - tic), name1, loss1, name2, loss2, name3, loss3, name4, loss4))
                if not (epoch + 1) % self._cfg.valid.val_interval:
                    # consider reduce the frequency of validation to save time
                    map_name, mean_ap = self._evaluate(val_data)
                    val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
                    self._logger.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
                    current_map = float(mean_ap[-1])
                    if current_map > self._best_map:
                        cp_name = os.path.join(self._logdir, _BEST_CHECKPOINT_FILE)
                        self._logger.info('[Epoch %d] Current best map: %f vs previous %f, saved to %s',
                                          self.epoch, current_map, self._best_map, cp_name)
                        self.save(cp_name)
                        self._best_map = current_map
                    if self._reporter:
                        self._reporter(epoch=epoch, map_reward=current_map)
                    early_stopper.update(current_map, epoch=epoch)
            self._time_elapsed += time.time() - post_tic

        # map on train data
        tic = time.time()
        map_name, mean_ap = self._evaluate(train_eval_data)
        self._time_elapsed += time.time() - tic
        return {'train_map': float(mean_ap[-1]), 'valid_map': self._best_map,
                'time': self._time_elapsed, 'checkpoint': cp_name}

    def _evaluate(self, val_data):
        """Evaluate the current model on dataset."""
        if not isinstance(val_data, gluon.data.DataLoader):
            if hasattr(val_data, 'to_mxnet'):
                val_data = val_data.to_mxnet()
            val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1))
            val_data = gluon.data.DataLoader(
                val_data.transform(YOLO3DefaultValTransform(self._cfg.yolo3.data_shape, self._cfg.yolo3.data_shape)),
                self._cfg.valid.batch_size, False, batchify_fn=val_batchify_fn, last_batch='keep',
                num_workers=self._cfg.num_workers)
        if self._cfg.valid.metric == 'voc07':
            eval_metric = VOC07MApMetric(iou_thresh=self._cfg.valid.iou_thresh, class_names=self.classes)
        elif self._cfg.valid.metric == 'voc':
            eval_metric = VOCMApMetric(iou_thresh=self._cfg.valid.iou_thresh, class_names=self.classes)
        else:
            raise ValueError(f'Invalid metric type: {self._cfg.valid.metric}')
        self.net.collect_params().reset_ctx(self.ctx)
        # set nms threshold and topk constraint
        self.net.set_nms(nms_thresh=self._cfg.yolo3.nms_thresh, nms_topk=self._cfg.yolo3.nms_topk)
        mx.nd.waitall()
        self.net.hybridize()
        for batch in val_data:
            val_ctx = self.ctx
            if batch[0].shape[0] < len(val_ctx):
                val_ctx = val_ctx[:batch[0].shape[0]]
            data = gluon.utils.split_and_load(batch[0], ctx_list=val_ctx, batch_axis=0, even_split=False)
            label = gluon.utils.split_and_load(batch[1], ctx_list=val_ctx, batch_axis=0, even_split=False)
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

    def _predict(self, x, ctx_id=0):
        """Predict an individual example."""
        short_size = int(self._cfg.yolo3.data_shape)
        if isinstance(x, str):
            x = load_test(x, short=short_size, max_size=1024)[0]
        elif isinstance(x, Image.Image):
            return self._predict(np.array(x))
        elif isinstance(x, np.ndarray):
            return self._predict(mx.nd.array(x))
        elif isinstance(x, mx.nd.NDArray):
            if len(x.shape) != 3 or x.shape[-1] != 3:
                raise ValueError('array input with shape (h, w, 3) is required for predict')
            x = transform_test(x, short=short_size, max_size=1024)[0]
        elif isinstance(x, pd.DataFrame):
            assert 'image' in x.columns, "Expect column `image` for input images"
            def _predict_merge(x, ctx_id=0):
                y = self._predict(x, ctx_id=ctx_id)
                y['image'] = x
                return y
            return pd.concat([_predict_merge(xx, ctx_id=ii % len(self.ctx)) \
                for ii, xx in enumerate(x['image'])]).reset_index(drop=True)
        elif isinstance(x, (list, tuple)):
            return pd.concat([self._predict(xx, ctx_id=ii % len(self.ctx)) \
                for ii, xx in enumerate(x)]).reset_index(drop=True)
        else:
            raise ValueError('Input is not supported: {}'.format(type(x)))
        height, width = x.shape[2:4]
        x = x.as_in_context(self.ctx[ctx_id])
        ids, scores, bboxes = [xx[0].asnumpy() for xx in self.net(x)]
        bboxes[:, (0, 2)] /= width
        bboxes[:, (1, 3)] /= height
        bboxes = np.clip(bboxes, 0.0, 1.0).tolist()
        df = pd.DataFrame({'predict_class': [self.classes[int(id)] for id in ids], 'predict_score': scores.flatten(),
                           'predict_rois': [{'xmin': bbox[0], 'ymin': bbox[1], 'xmax': bbox[2], 'ymax': bbox[3]} \
                                for bbox in bboxes]})
        # filter out invalid (scores < 0) rows
        valid_df = df[df['predict_score'] > 0].reset_index(drop=True)
        return valid_df

    def _init_network(self, **kwargs):
        load_only = kwargs.get('load_only', False)
        if not self.num_class:
            raise ValueError('Unable to create network when `num_class` is unknown. \
                It should be inferred from dataset or resumed from saved states.')
        assert len(self.classes) == self.num_class

        # training contexts
        if self._cfg.horovod:
            self.ctx = [mx.gpu(hvd.local_rank())]
        else:
            valid_gpus = []
            if self._cfg.gpus:
                valid_gpus = self._validate_gpus(self._cfg.gpus)
                if not valid_gpus:
                    self._logger.warning(
                        'No gpu detected, fallback to cpu. You can ignore this warning if this is intended.')
                elif len(valid_gpus) != len(self._cfg.gpus):
                    self._logger.warning(
                        f'Loaded on gpu({valid_gpus}), different from gpu({self._cfg.gpus}).')
            ctx = [mx.gpu(int(i)) for i in valid_gpus]
            self.ctx = ctx if ctx else [mx.cpu()]

        if self._cfg.yolo3.transfer is None:
            # use sync bn if specified
            if self._cfg.yolo3.syncbn and len(self.ctx) > 1:
                self.net = custom_yolov3(base_network_name=self._cfg.yolo3.base_network,
                                         filters=self._cfg.yolo3.filters,
                                         anchors=self._cfg.yolo3.anchors,
                                         strides=self._cfg.yolo3.strides,
                                         classes=self.classes,
                                         dataset=self._cfg.dataset,
                                         pretrained_base=(not load_only),
                                         norm_layer=gluon.contrib.nn.SyncBatchNorm,
                                         norm_kwargs={'num_devices': len(self.ctx)})
                self.async_net = custom_yolov3(base_network_name=self._cfg.yolo3.base_network,
                                               filters=self._cfg.yolo3.filters,
                                               anchors=self._cfg.yolo3.anchors,
                                               strides=self._cfg.yolo3.strides,
                                               classes=self.classes,
                                               dataset=self._cfg.dataset,
                                               pretrained_base=False)
            else:
                self.net = custom_yolov3(base_network_name=self._cfg.yolo3.base_network,
                                         filters=self._cfg.yolo3.filters,
                                         anchors=self._cfg.yolo3.anchors,
                                         strides=self._cfg.yolo3.strides,
                                         classes=self.classes,
                                         dataset=self._cfg.dataset,
                                         pretrained_base=(not load_only))
                self.async_net = self.net
        else:
            assert isinstance(self._cfg.yolo3.transfer, str)
            self._logger.info(
                f'Using transfer learning from {self._cfg.yolo3.transfer}, the other network parameters are ignored.')
            # use sync bn if specified
            if self._cfg.yolo3.syncbn and len(self.ctx) > 1:
                self.net = get_model(self._cfg.yolo3.transfer, pretrained=(not load_only),
                                     norm_layer=gluon.contrib.nn.SyncBatchNorm,
                                     norm_kwargs={'num_devices': len(self.ctx)})
                self.async_net = get_model(self._cfg.yolo3.transfer, pretrained=(not load_only))  # used by cpu worker
            else:
                self.net = get_model(self._cfg.yolo3.transfer, pretrained=(not load_only))
                self.async_net = self.net
            if load_only:
                self.net.initialize()
                self.net.set_nms(nms_thresh=0)
                self.net(mx.nd.zeros((1, 3, self._cfg.yolo3.data_shape, self._cfg.yolo3.data_shape)))
                self.net.set_nms(nms_thresh=self._cfg.yolo3.nms_thresh, nms_topk=self._cfg.yolo3.nms_topk)
            self.net.reset_class(self.classes,
                                 reuse_weights=[cname for cname in self.classes if cname in self.net.classes])

        with warnings.catch_warnings(record=True) as _:
            warnings.simplefilter("always")
            self.net.initialize()
            self.async_net.initialize()
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

        if self._cfg.train.lr_decay_period > 0:
            lr_decay_epoch = list(range(self._cfg.train.lr_decay_period,
                                        self._cfg.train.epochs,
                                        self._cfg.train.lr_decay_period))
        else:
            lr_decay_epoch = [int(i) for i in self._cfg.train.lr_decay_epoch]
        lr_decay_epoch = [e - self._cfg.train.warmup_epochs for e in lr_decay_epoch]
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

        if self._cfg.horovod:
            hvd.broadcast_parameters(self.net.collect_params(), root_rank=0)
            self.trainer = hvd.DistributedTrainer(
                self.net.collect_params(), 'sgd',
                {'wd': self._cfg.train.wd, 'momentum': self._cfg.train.momentum, 'lr_scheduler': lr_scheduler})
        else:
            self.trainer = gluon.Trainer(
                self.net.collect_params(), 'sgd',
                {'wd': self._cfg.train.wd, 'momentum': self._cfg.train.momentum, 'lr_scheduler': lr_scheduler},
                kvstore='local', update_on_kvstore=(False if self._cfg.yolo3.amp else None))

        if self._cfg.yolo3.amp:
            amp.init_trainer(self.trainer)
