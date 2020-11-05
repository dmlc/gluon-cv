"""Faster RCNN Estimator."""
# pylint: disable=logging-not-lazy
import os
import time

import pandas as pd
import numpy as np
import mxnet as mx
from mxnet import gluon

from ....data.batchify import Tuple, Append
from ....data.transforms.presets.rcnn import load_test, transform_test
from ....data.transforms.presets.rcnn import FasterRCNNDefaultTrainTransform, FasterRCNNDefaultValTransform
from ....model_zoo import get_model
from ....model_zoo.rcnn.faster_rcnn.data_parallel import ForwardBackwardTask
from ....nn.bbox import BBoxClipToImage
from ....utils.parallel import Parallel
from ....utils.metrics.rcnn import RPNAccMetric, RPNL1LossMetric, RCNNAccMetric, RCNNL1LossMetric
from ....utils.metrics.voc_detection import VOC07MApMetric, VOCMApMetric
from ..base_estimator import BaseEstimator, set_default
from .utils import _get_lr_at_iter, _get_dataloader, _split_and_load

try:
    import horovod.mxnet as hvd
except ImportError:
    hvd = None

from .default import FasterRCNNCfg

__all__ = ['FasterRCNNEstimator']


@set_default(FasterRCNNCfg())
class FasterRCNNEstimator(BaseEstimator):
    """Estimator implementation for Faster-RCNN.

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
    def __init__(self, config, logger=None, reporter=None):
        super(FasterRCNNEstimator, self).__init__(config, logger, reporter)

    def _fit(self, train_data, val_data):
        """Fit Faster R-CNN model."""
        self._best_map = 0
        self.epoch = 0
        self._time_elapsed = 0
        if max(self._cfg.train.start_epoch, self.epoch) >= self._cfg.train.epochs:
            return {'time', self._time_elapsed}
        self.net.collect_params().setattr('grad_req', 'null')
        self.net.collect_train_params().setattr('grad_req', 'write')
        self._init_trainer()
        return self._resume_fit(train_data, val_data)

    def _resume_fit(self, train_data, val_data):
        if max(self._cfg.train.start_epoch, self.epoch) >= self._cfg.train.epochs:
            return {'time', self._time_elapsed}
        if not self.classes or not self.num_class:
            raise ValueError('Unable to determine classes of dataset')

        # dataset
        train_dataset = train_data.to_mxnet()
        val_dataset = val_data.to_mxnet()

        # dataloader
        self.batch_size = self._cfg.train.batch_size // min(1, self.num_gpus) \
            if self._cfg.horovod else self._cfg.train.batch_size
        if not self._cfg.horovod:
            if self._cfg.train.batch_size == 1 and self.num_gpus > 1:
                self.batch_size *= self.num_gpus
        train_loader, val_loader, train_eval_loader = _get_dataloader(
            self.net, train_dataset, val_dataset, FasterRCNNDefaultTrainTransform,
            FasterRCNNDefaultValTransform, self.batch_size, len(self.ctx), self._cfg)

        return self._train_loop(train_loader, val_loader, train_eval_loader)

    def _train_loop(self, train_data, val_data, train_eval_data):
        # loss and metric
        rpn_cls_loss = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
        rpn_box_loss = gluon.loss.HuberLoss(rho=self._cfg.train.rpn_smoothl1_rho)  # == smoothl1
        rcnn_cls_loss = gluon.loss.SoftmaxCrossEntropyLoss()
        rcnn_box_loss = gluon.loss.HuberLoss(rho=self._cfg.train.rcnn_smoothl1_rho)  # == smoothl1
        metrics = [mx.metric.Loss('RPN_Conf'),
                   mx.metric.Loss('RPN_SmoothL1'),
                   mx.metric.Loss('RCNN_CrossEntropy'),
                   mx.metric.Loss('RCNN_SmoothL1')]

        rpn_acc_metric = RPNAccMetric()
        rpn_bbox_metric = RPNL1LossMetric()
        rcnn_acc_metric = RCNNAccMetric()
        rcnn_bbox_metric = RCNNL1LossMetric()
        metrics2 = [rpn_acc_metric, rpn_bbox_metric, rcnn_acc_metric, rcnn_bbox_metric]

        # lr decay policy
        lr_decay = float(self._cfg.train.lr_decay)
        lr_steps = sorted(
            [float(ls) for ls in self._cfg.train.lr_decay_epoch])
        lr_warmup = float(self._cfg.train.lr_warmup)  # avoid int division

        if self._cfg.train.verbose:
            self._logger.info('Trainable parameters:')
            self._logger.info(self.net.collect_train_params().keys())
        self._logger.info('Start training from [Epoch %d]', max(self._cfg.train.start_epoch, self.epoch))

        for self.epoch in range(max(self._cfg.train.start_epoch, self.epoch), self._cfg.train.epochs):
            epoch = self.epoch
            btic = time.time()
            rcnn_task = ForwardBackwardTask(self.net, self.trainer, rpn_cls_loss, rpn_box_loss,
                                            rcnn_cls_loss, rcnn_box_loss, mix_ratio=1.0,
                                            amp_enabled=self._cfg.faster_rcnn.amp)
            executor = Parallel(self._cfg.train.executor_threads,
                                rcnn_task) if not self._cfg.horovod else None
            mix_ratio = 1.0
            if not self._cfg.disable_hybridization:
                self.net.hybridize(static_alloc=self._cfg.faster_rcnn.static_alloc)
            if self._cfg.train.mixup:
                # TODO(zhreshold) only support evenly mixup now, target generator needs to be
                #  modified otherwise
                train_data._dataset._data.set_mixup(np.random.uniform, 0.5, 0.5)
                mix_ratio = 0.5
                if epoch >= self._cfg.train.epochs - self._cfg.train.no_mixup_epochs:
                    train_data._dataset._data.set_mixup(None)
                    mix_ratio = 1.0
            while lr_steps and epoch >= lr_steps[0]:
                new_lr = self.trainer.learning_rate * lr_decay
                lr_steps.pop(0)
                self.trainer.set_learning_rate(new_lr)
                self._logger.info("[Epoch %d] Set learning rate to %f", epoch, new_lr)
            for metric in metrics:
                metric.reset()
            tic = time.time()
            base_lr = self.trainer.learning_rate
            rcnn_task.mix_ratio = mix_ratio
            for i, batch in enumerate(train_data):
                if epoch == 0 and i <= lr_warmup:
                    # adjust based on real percentage
                    new_lr = base_lr * _get_lr_at_iter(i / lr_warmup,
                                                       self._cfg.train.lr_warmup_factor)
                    if new_lr != self.trainer.learning_rate:
                        if i % self._cfg.train.log_interval == 0:
                            self._logger.info(
                                '[Epoch 0 Iteration %d] Set learning rate to %f', i, new_lr)
                        self.trainer.set_learning_rate(new_lr)
                batch = _split_and_load(batch, ctx_list=self.ctx)
                metric_losses = [[] for _ in metrics]
                add_losses = [[] for _ in metrics2]
                if executor is not None:
                    for data in zip(*batch):
                        executor.put(data)
                for _ in range(len(self.ctx)):
                    if executor is not None:
                        result = executor.get()
                    else:
                        result = rcnn_task.forward_backward(list(zip(*batch))[0])
                    if (not self._cfg.horovod) or hvd.rank() == 0:
                        for k, metric_loss in enumerate(metric_losses):
                            metric_loss.append(result[k])
                        for k, add_loss in enumerate(add_losses):
                            add_loss.append(result[len(metric_losses) + k])
                for metric, record in zip(metrics, metric_losses):
                    metric.update(0, record)
                for metric, records in zip(metrics2, add_losses):
                    for pred in records:
                        metric.update(pred[0], pred[1])
                self.trainer.step(self.batch_size)

                # update metrics
                if (not self._cfg.horovod or hvd.rank() == 0) and self._cfg.train.log_interval \
                        and not (i + 1) % self._cfg.train.log_interval:
                    msg = ','.join(
                        ['{}={:.3f}'.format(*metric.get()) for metric in
                         metrics + metrics2])
                    self._logger.info('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}'.format(
                        epoch, i,
                        self._cfg.train.log_interval * self._cfg.train.batch_size / (
                            time.time() - btic), msg))
                    btic = time.time()

            if not self._cfg.horovod or hvd.rank() == 0:
                msg = ','.join(['{}={:.3f}'.format(*metric.get()) for metric in metrics])
                # pylint: disable=logging-format-interpolation
                self._logger.info('[Epoch {}] Training cost: {:.3f}, {}'.format(
                    epoch, (time.time() - tic), msg))
                if not (epoch + 1) % self._cfg.valid.val_interval:
                    # consider reduce the frequency of validation to save time
                    map_name, mean_ap = self._evaluate(val_data)
                    val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
                    self._logger.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
                    current_map = float(mean_ap[-1])
                    if current_map > self._best_map:
                        cp_name = os.path.join(self._logdir, 'best_checkpoint.pkl')
                        self._logger.info('[Epoch %d] Current best map: %f vs previous %f, saved to %s',
                                          self.epoch, current_map, self._best_map, cp_name)
                        self.save(cp_name)
                        self._best_map = current_map
                if self._reporter:
                    self._reporter(epoch=epoch, map_reward=current_map)
            self._time_elapsed += time.time() - btic
        # map on train data
        map_name, mean_ap = self._evaluate(train_eval_data)
        return {'train_map': float(mean_ap[-1]), 'valid_map': self._best_map, 'time': self._time_elapsed}

    def _evaluate(self, val_data):
        """Evaluate on validation dataset."""
        clipper = BBoxClipToImage()
        if not isinstance(val_data, gluon.data.DataLoader):
            if hasattr(val_data, 'to_mxnet'):
                val_data = val_data.to_mxnet()
            val_bfn = Tuple(*[Append() for _ in range(3)])
            short = self.net.short[-1] if isinstance(self.net.short, (tuple, list)) else self.net.short
            # validation use 1 sample per device
            val_data = gluon.data.DataLoader(
                val_data.transform(FasterRCNNDefaultValTransform(short, self.net.max_size)),
                len(self.ctx), False, batchify_fn=val_bfn, last_batch='keep',
                num_workers=self._cfg.num_workers)
        if self._cfg.valid.metric == 'voc07':
            eval_metric = VOC07MApMetric(iou_thresh=self._cfg.valid.iou_thresh, class_names=self.classes)
        elif self._cfg.valid.metric == 'voc':
            eval_metric = VOCMApMetric(iou_thresh=self._cfg.valid.iou_thresh, class_names=self.classes)
        else:
            raise ValueError(f'Invalid metric type: {self._cfg.valid.metric}')

        if not self._cfg.disable_hybridization:
            # input format is differnet than training, thus rehybridization is needed.
            self.net.hybridize(static_alloc=self._cfg.faster_rcnn.static_alloc)
        for batch in val_data:
            batch = _split_and_load(batch, ctx_list=self.ctx)
            det_bboxes = []
            det_ids = []
            det_scores = []
            gt_bboxes = []
            gt_ids = []
            gt_difficults = []
            for x, y, im_scale in zip(*batch):
                # get prediction results
                ids, scores, bboxes = self.net(x)
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
                gt_difficults.append(
                    y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)
            # update metric
            for det_bbox, det_id, det_score, gt_bbox, gt_id, gt_diff in zip(det_bboxes, det_ids,
                                                                            det_scores, gt_bboxes,
                                                                            gt_ids, gt_difficults):
                eval_metric.update(det_bbox, det_id, det_score, gt_bbox, gt_id, gt_diff)
        return eval_metric.get()

    def _predict(self, x):
        """Predict an individual example."""
        short_size = self.net.short[-1] if isinstance(self.net.short, (tuple, list)) else self.net.short
        if isinstance(x, str):
            x = load_test(x, short=short_size, max_size=1024)[0]
        elif isinstance(x, mx.nd.NDArray):
            x = transform_test(x, short=short_size, max_size=1024)[0]
        elif isinstance(x, pd.DataFrame):
            assert 'image' in x.columns, "Expect column `image` for input images"
            def _predict_merge(x):
                y = self._predict(x)
                y['image'] = x
                return y
            return pd.concat([_predict_merge(xx) for xx in x['image']]).reset_index(drop=True)
        else:
            raise ValueError('Input is not supported: {}'.format(type(x)))
        height, width = x.shape[2:4]
        x = x.as_in_context(self.ctx[0])
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
        kwargs = {}
        module_list = []
        if self._cfg.faster_rcnn.use_fpn:
            module_list.append('fpn')
        if self._cfg.faster_rcnn.norm_layer is not None:
            module_list.append(self._cfg.faster_rcnn.norm_layer)
            if self._cfg.faster_rcnn.norm_layer == 'syncbn':
                kwargs['num_devices'] = len(self.ctx)

        self.num_gpus = hvd.size() if self._cfg.horovod else len(self.ctx)

        if self._cfg.faster_rcnn.transfer is not None:
            assert isinstance(self._cfg.faster_rcnn.transfer, str)
            self._logger.info(
                f'Using transfer learning from {self._cfg.faster_rcnn.transfer}, ' +
                'the other network parameters are ignored.')
            self._cfg.faster_rcnn.use_fpn = 'fpn' in self._cfg.faster_rcnn.transfer
            self.net = get_model(self._cfg.faster_rcnn.transfer, pretrained=True,
                                 per_device_batch_size=self._cfg.train.batch_size // self.num_gpus,
                                 **kwargs)
            self.net.reset_class(self.classes,
                                 reuse_weights=[cname for cname in self.classes if cname in self.net.classes])
        else:
            self._cfg.faster_rcnn.use_fpn = True
            if self._cfg.faster_rcnn.norm_layer == 'syncbn':
                norm_layer = gluon.contrib.nn.SyncBatchNorm
                norm_kwargs = {'num_devices': len(self.ctx)}
                sym_norm_layer = mx.sym.contrib.SyncBatchNorm
                sym_norm_kwargs = {'ndev': len(self.ctx)}
            elif self._cfg.faster_rcnn.norm_layer == 'gn':
                norm_layer = gluon.nn.GroupNorm
                norm_kwargs = {'groups': 8}
                sym_norm_layer = mx.sym.GroupNorm
                sym_norm_kwargs = {'groups': 8}
            else:
                norm_layer = gluon.nn.BatchNorm
                norm_kwargs = None
                sym_norm_layer = None
                sym_norm_kwargs = None
            self.net = get_model('custom_faster_rcnn_fpn', classes=self.classes, transfer=None,
                                 dataset=self._cfg.dataset,
                                 pretrained_base=self._cfg.train.pretrained_base,
                                 base_network_name=self._cfg.faster_rcnn.backbone,
                                 norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                                 sym_norm_layer=sym_norm_layer, sym_norm_kwargs=sym_norm_kwargs,
                                 num_fpn_filters=self._cfg.faster_rcnn.num_fpn_filters,
                                 num_box_head_conv=self._cfg.faster_rcnn.num_box_head_conv,
                                 num_box_head_conv_filters=
                                 self._cfg.faster_rcnn.num_box_head_conv_filters,
                                 num_box_head_dense_filters=
                                 self._cfg.faster_rcnn.num_box_head_dense_filters,
                                 short=self._cfg.faster_rcnn.image_short,
                                 max_size=self._cfg.faster_rcnn.image_max_size,
                                 min_stage=2, max_stage=6,
                                 nms_thresh=self._cfg.faster_rcnn.nms_thresh,
                                 nms_topk=self._cfg.faster_rcnn.nms_topk,
                                 roi_mode=self._cfg.faster_rcnn.roi_mode,
                                 roi_size=self._cfg.faster_rcnn.roi_size,
                                 strides=self._cfg.faster_rcnn.strides,
                                 clip=self._cfg.faster_rcnn.clip,
                                 rpn_channel=self._cfg.faster_rcnn.rpn_channel,
                                 base_size=self._cfg.faster_rcnn.anchor_base_size,
                                 scales=self._cfg.faster_rcnn.anchor_scales,
                                 ratios=self._cfg.faster_rcnn.anchor_aspect_ratio,
                                 alloc_size=self._cfg.faster_rcnn.anchor_alloc_size,
                                 rpn_nms_thresh=self._cfg.faster_rcnn.rpn_nms_thresh,
                                 rpn_train_pre_nms=self._cfg.train.rpn_train_pre_nms,
                                 rpn_train_post_nms=self._cfg.train.rpn_train_post_nms,
                                 rpn_test_pre_nms=self._cfg.valid.rpn_test_pre_nms,
                                 rpn_test_post_nms=self._cfg.valid.rpn_test_post_nms,
                                 rpn_min_size=self._cfg.train.rpn_min_size,
                                 per_device_batch_size=self._cfg.train.batch_size // self.num_gpus,
                                 num_sample=self._cfg.train.rcnn_num_samples,
                                 pos_iou_thresh=self._cfg.train.rcnn_pos_iou_thresh,
                                 pos_ratio=self._cfg.train.rcnn_pos_ratio,
                                 max_num_gt=self._cfg.faster_rcnn.max_num_gt)

        if self._cfg.resume.strip():
            self.net.load_parameters(self._cfg.resume.strip())
        else:
            for param in self.net.collect_params().values():
                if param._data is not None:
                    continue
                param.initialize()
        self.net.collect_params().reset_ctx(self.ctx)
        if self._cfg.faster_rcnn.amp:
            # Cast both weights and gradients to 'float16'
            self.net.cast('float16')
            # These layers don't support type 'float16'
            self.net.collect_params('.*batchnorm.*').setattr('dtype', 'float32')
            self.net.collect_params('.*normalizedperclassboxcenterencoder.*').setattr('dtype', 'float32')
        if self._cfg.resume.strip():
            self.net.load_parameters(self._cfg.resume.strip())
        else:
            for param in self.net.collect_params().values():
                if param._data is not None:
                    continue
                param.initialize()
        self.net.collect_params().reset_ctx(self.ctx)

    def _init_trainer(self):
        kv_store_type = 'device' if (self._cfg.faster_rcnn.amp and 'nccl' in self._cfg.kv_store) \
            else self._cfg.kv_store
        kv = mx.kvstore.create(kv_store_type)
        optimizer_params = {'learning_rate': self._cfg.train.lr, 'wd': self._cfg.train.wd,
                            'momentum': self._cfg.train.momentum}
        if self._cfg.faster_rcnn.amp:
            optimizer_params['multi_precision'] = True
        if self._cfg.horovod:
            hvd.broadcast_parameters(self.net.collect_params(), root_rank=0)
            self.trainer = hvd.DistributedTrainer(
                self.net.collect_train_params(),  # fix batchnorm, fix first stage, etc...
                'sgd',
                optimizer_params)
        else:
            self.trainer = gluon.Trainer(
                self.net.collect_train_params(),  # fix batchnorm, fix first stage, etc...
                'sgd',
                optimizer_params,
                update_on_kvstore=(False if self._cfg.faster_rcnn.amp else None), kvstore=kv)

        if self._cfg.faster_rcnn.amp:
            self._cfg.init_trainer(self.trainer)
