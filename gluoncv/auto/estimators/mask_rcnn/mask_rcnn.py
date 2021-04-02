"""Mask RCNN Estimator."""
# pylint: disable=consider-using-enumerate,abstract-method,arguments-differ
import os
import math
import time
import logging
from multiprocessing import Process

import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.contrib import amp

from .... import data as gdata
from .... import utils as gutils
from ....data import COCODetection, VOCDetection
from ....data.transforms import presets
from ....data.transforms.presets.rcnn import MaskRCNNDefaultTrainTransform, MaskRCNNDefaultValTransform
from ....model_zoo import get_model
from ....model_zoo.rcnn.mask_rcnn.data_parallel import ForwardBackwardTask
from ....nn.bbox import BBoxClipToImage
from ....utils.parallel import Parallel
from ....utils.metrics.rcnn import RPNAccMetric, RPNL1LossMetric, RCNNAccMetric, RCNNL1LossMetric, \
    MaskAccMetric, MaskFGAccMetric
from ..base_estimator import BaseEstimator, set_default
from .utils import _get_dataset, _get_dataloader, _save_params, _split_and_load, _get_lr_at_iter

try:
    import horovod.mxnet as hvd
except ImportError:
    hvd = None

try:
    from mpi4py import MPI
except ImportError:
    logging.info('mpi4py is not installed. Use "pip install --no-cache mpi4py" to install')
    MPI = None

from .default import MaskRCNNCfg

__all__ = ['MaskRCNNEstimator']


@set_default(MaskRCNNCfg())
class MaskRCNNEstimator(BaseEstimator):
    """Estimator implementation for Mask-RCNN.

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
        super(MaskRCNNEstimator, self).__init__(config, logger, reporter)

        # fix seed for mxnet, numpy and python builtin random generator.
        gutils.random.seed(self._cfg.train.seed)

        if self._cfg.mask_rcnn.amp:
            amp.init()

        # training contexts
        if self._cfg.horovod:
            self.ctx = [mx.gpu(hvd.local_rank())]
        else:
            ctx = [mx.gpu(int(i)) for i in self._cfg.gpus]
            self.ctx = ctx if ctx else [mx.cpu()]

        # network
        kwargs = {}
        module_list = []
        if self._cfg.mask_rcnn.use_fpn:
            module_list.append('fpn')
        if self._cfg.mask_rcnn.norm_layer is not None:
            module_list.append(self._cfg.mask_rcnn.norm_layer)
            if self._cfg.mask_rcnn.norm_layer == 'bn':
                kwargs['num_devices'] = len(self.ctx)
        self.num_gpus = hvd.size() if self._cfg.horovod else len(self.ctx)
        net_name = '_'.join(('mask_rcnn', *module_list, self._cfg.mask_rcnn.backbone, self._cfg.dataset))
        if self._cfg.mask_rcnn.custom_model:
            self._cfg.mask_rcnn.use_fpn = True
            net_name = '_'.join(('mask_rcnn_fpn', self._cfg.mask_rcnn.backbone, self._cfg.dataset))
            if self._cfg.mask_rcnn.norm_layer == 'bn':
                norm_layer = gluon.contrib.nn.SyncBatchNorm
                norm_kwargs = {'num_devices': len(self.ctx)}
                # sym_norm_layer = mx.sym.contrib.SyncBatchNorm
                sym_norm_kwargs = {'ndev': len(self.ctx)}
            elif self._cfg.mask_rcnn.norm_layer == 'gn':
                norm_layer = gluon.nn.GroupNorm
                norm_kwargs = {'groups': 8}
                # sym_norm_layer = mx.sym.GroupNorm
                sym_norm_kwargs = {'groups': 8}
            else:
                norm_layer = gluon.nn.BatchNorm
                norm_kwargs = None
                # sym_norm_layer = None
                sym_norm_kwargs = None
            if self._cfg.dataset == 'coco':
                classes = COCODetection.CLASSES
            else:
                # default to VOC
                classes = VOCDetection.CLASSES
            self.net = get_model('custom_mask_rcnn_fpn', classes=classes, transfer=None,
                                 dataset=self._cfg.dataset, pretrained_base=self._cfg.train.pretrained_base,
                                 base_network_name=self._cfg.mask_rcnn.backbone, norm_layer=norm_layer,
                                 norm_kwargs=norm_kwargs, sym_norm_kwargs=sym_norm_kwargs,
                                 num_fpn_filters=self._cfg.mask_rcnn.num_fpn_filters,
                                 num_box_head_conv=self._cfg.mask_rcnn.num_box_head_conv,
                                 num_box_head_conv_filters=self._cfg.mask_rcnn.num_box_head_conv_filters,
                                 num_box_head_dense_filters=self._cfg.mask_rcnn.num_box_head_dense_filters,
                                 short=self._cfg.mask_rcnn.image_short, max_size=self._cfg.mask_rcnn.image_max_size,
                                 min_stage=2, max_stage=6, nms_thresh=self._cfg.mask_rcnn.nms_thresh,
                                 nms_topk=self._cfg.mask_rcnn.nms_topk, post_nms=self._cfg.mask_rcnn.post_nms,
                                 roi_mode=self._cfg.mask_rcnn.roi_mode, roi_size=self._cfg.mask_rcnn.roi_size,
                                 strides=self._cfg.mask_rcnn.strides, clip=self._cfg.mask_rcnn.clip,
                                 rpn_channel=self._cfg.mask_rcnn.rpn_channel,
                                 base_size=self._cfg.mask_rcnn.anchor_base_size,
                                 scales=self._cfg.mask_rcnn.anchor_scales,
                                 ratios=self._cfg.mask_rcnn.anchor_aspect_ratio,
                                 alloc_size=self._cfg.mask_rcnn.anchor_alloc_size,
                                 rpn_nms_thresh=self._cfg.mask_rcnn.rpn_nms_thresh,
                                 rpn_train_pre_nms=self._cfg.train.rpn_train_pre_nms,
                                 rpn_train_post_nms=self._cfg.train.rpn_train_post_nms,
                                 rpn_test_pre_nms=self._cfg.valid.rpn_test_pre_nms,
                                 rpn_test_post_nms=self._cfg.valid.rpn_test_post_nms,
                                 rpn_min_size=self._cfg.train.rpn_min_size,
                                 per_device_batch_size=self._cfg.train.batch_size // self.num_gpus,
                                 num_sample=self._cfg.train.rcnn_num_samples,
                                 pos_iou_thresh=self._cfg.train.rcnn_pos_iou_thresh,
                                 pos_ratio=self._cfg.train.rcnn_pos_ratio,
                                 max_num_gt=self._cfg.mask_rcnn.max_num_gt,
                                 target_roi_scale=self._cfg.mask_rcnn.target_roi_scale,
                                 num_fcn_convs=self._cfg.mask_rcnn.num_mask_head_convs)
        else:
            self.net = get_model(net_name, pretrained_base=True,
                                 per_device_batch_size=self._cfg.train.batch_size // self.num_gpus, **kwargs)
        self._cfg.save_prefix += net_name
        if self._cfg.resume.strip():
            self.net.load_parameters(self._cfg.resume.strip())
        else:
            for param in self.net.collect_params().values():
                if param._data is not None:
                    continue
                param.initialize()
        self.net.collect_params().reset_ctx(self.ctx)

        if self._cfg.mask_rcnn.amp:
            # Cast both weights and gradients to 'float16'
            self.net.cast('float16')
            # This layers doesn't support type 'float16'
            self.net.collect_params('.*batchnorm.*').setattr('dtype', 'float32')
            self.net.collect_params('.*normalizedperclassboxcenterencoder.*').setattr('dtype', 'float32')

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
        if MPI is None and self._cfg.horovod:
            self._logger.warning('mpi4py is not installed, validation result may be incorrect.')
        self._logger.info(self._cfg)

        self.rpn_cls_loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
        self.rpn_box_loss = mx.gluon.loss.HuberLoss(rho=self._cfg.train.rpn_smoothl1_rho)  # == smoothl1
        self.rcnn_cls_loss = mx.gluon.loss.SoftmaxCrossEntropyLoss()
        self.rcnn_box_loss = mx.gluon.loss.HuberLoss(rho=self._cfg.train.rcnn_smoothl1_rho)  # == smoothl1
        self.rcnn_mask_loss = mx.gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=False)
        self.metrics = [mx.metric.Loss('RPN_Conf'),
                        mx.metric.Loss('RPN_SmoothL1'),
                        mx.metric.Loss('RCNN_CrossEntropy'),
                        mx.metric.Loss('RCNN_SmoothL1'),
                        mx.metric.Loss('RCNN_Mask')]

        self.rpn_acc_metric = RPNAccMetric()
        self.rpn_bbox_metric = RPNL1LossMetric()
        self.rcnn_acc_metric = RCNNAccMetric()
        self.rcnn_bbox_metric = RCNNL1LossMetric()
        self.rcnn_mask_metric = MaskAccMetric()
        self.rcnn_fgmask_metric = MaskFGAccMetric()
        self.metrics2 = [self.rpn_acc_metric, self.rpn_bbox_metric,
                         self.rcnn_acc_metric, self.rcnn_bbox_metric,
                         self.rcnn_mask_metric, self.rcnn_fgmask_metric]

        self.async_eval_processes = []
        self.best_map = [0]
        self.epoch = 0

        # training data
        self.train_dataset, self.val_dataset, self.eval_metric = _get_dataset(self._cfg.dataset, self._cfg)
        self.batch_size = self._cfg.train.batch_size // self.num_gpus \
            if self._cfg.horovod else self._cfg.train.batch_size
        self._train_data, self._val_data = _get_dataloader(
            self.net, self.train_dataset, self.val_dataset, MaskRCNNDefaultTrainTransform, MaskRCNNDefaultValTransform,
            self.batch_size, len(self.ctx), self._cfg)

    def _validate(self, val_data, async_eval_processes, ctx, eval_metric, logger, epoch, best_map):
        """Test on validation dataset."""
        clipper = BBoxClipToImage()
        eval_metric.reset()
        if not self._cfg.disable_hybridization:
            self.net.hybridize(static_alloc=self._cfg.mask_rcnn.static_alloc)
        tic = time.time()
        for _, batch in enumerate(val_data):
            batch = _split_and_load(batch, ctx_list=ctx)
            det_bboxes = []
            det_ids = []
            det_scores = []
            det_masks = []
            det_infos = []
            for x, im_info in zip(*batch):
                # get prediction results
                ids, scores, bboxes, masks = self.net(x)
                det_bboxes.append(clipper(bboxes, x))
                det_ids.append(ids)
                det_scores.append(scores)
                det_masks.append(masks)
                det_infos.append(im_info)
            # update metric
            for det_bbox, det_id, det_score, det_mask, det_info in zip(det_bboxes, det_ids, det_scores,
                                                                       det_masks, det_infos):
                for i in range(det_info.shape[0]):
                    # numpy everything
                    det_bbox = det_bbox[i].asnumpy()
                    det_id = det_id[i].asnumpy()
                    det_score = det_score[i].asnumpy()
                    det_mask = det_mask[i].asnumpy()
                    det_info = det_info[i].asnumpy()
                    # filter by conf threshold
                    im_height, im_width, im_scale = det_info
                    valid = np.where(((det_id >= 0) & (det_score >= 0.001)))[0]
                    det_id = det_id[valid]
                    det_score = det_score[valid]
                    det_bbox = det_bbox[valid] / im_scale
                    det_mask = det_mask[valid]
                    # fill full mask
                    im_height, im_width = int(round(im_height / im_scale)), int(
                        round(im_width / im_scale))
                    full_masks = gdata.transforms.mask.fill(det_mask, det_bbox, (im_width, im_height))
                    eval_metric.update(det_bbox, det_id, det_score, full_masks)
        if self._cfg.horovod and MPI is not None:
            comm = MPI.COMM_WORLD
            res = comm.gather(eval_metric.get_result_buffer(), root=0)
            if hvd.rank() == 0:
                logger.info('[Epoch {}] Validation Inference cost: {:.3f}'
                            .format(epoch, (time.time() - tic)))
                rank0_res = eval_metric.get_result_buffer()
                if len(rank0_res) == 2:
                    res = res[1:]
                    rank0_res[0].extend([item for res_tuple in res for item in res_tuple[0]])
                    rank0_res[1].extend([item for res_tuple in res for item in res_tuple[1]])
                else:
                    rank0_res.extend([item for r in res for item in r])

        def coco_eval_save_task(eval_metric, logger):
            map_name, mean_ap = eval_metric.get()
            if map_name and mean_ap is not None:
                val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
                logger.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
                current_map = float(mean_ap[-1])
                _save_params(self.net, logger, best_map, current_map, epoch, self._cfg.save_interval,
                             os.path.join(self._logdir, self._cfg.save_prefix))

        if not self._cfg.horovod or hvd.rank() == 0:
            p = Process(target=coco_eval_save_task, args=(eval_metric, self._logger))
            async_eval_processes.append(p)
            p.start()

    def _fit(self, train_data, val_data, time_limit=math.inf):
        """
        Fit Mask R-CNN models.
        """
        # TODO(zhreshold): remove 'dataset' in config, use train_data/val_data instead
        self._cfg.kv_store = 'device' \
            if (self._cfg.mask_rcnn.amp and 'nccl' in self._cfg.kv_store) else self._cfg.kv_store
        kv = mx.kvstore.create(self._cfg.kv_store)
        self.net.collect_params().setattr('grad_req', 'null')
        self.net.collect_train_params().setattr('grad_req', 'write')
        for k, v in self.net.collect_params('.*bias').items():
            v.wd_mult = 0.0
        optimizer_params = {'learning_rate': self._cfg.train.lr, 'wd': self._cfg.train.wd,
                            'momentum': self._cfg.train.momentum, }
        if self._cfg.train.clip_gradient > 0.0:
            optimizer_params['clip_gradient'] = self._cfg.train.clip_gradient
        if self._cfg.mask_rcnn.amp:
            optimizer_params['multi_precision'] = True
        if self._cfg.horovod:
            hvd.broadcast_parameters(self.net.collect_params(), root_rank=0)
            trainer = hvd.DistributedTrainer(
                self.net.collect_train_params(),  # fix batchnorm, fix first stage, etc...
                'sgd',
                optimizer_params
            )
        else:
            trainer = gluon.Trainer(
                self.net.collect_train_params(),  # fix batchnorm, fix first stage, etc...
                'sgd',
                optimizer_params,
                update_on_kvstore=(False if self._cfg.mask_rcnn.amp else None),
                kvstore=kv)

        if self._cfg.mask_rcnn.amp:
            amp.init_trainer(trainer)

        # lr decay policy
        lr_decay = float(self._cfg.train.lr_decay)
        lr_steps = sorted([float(ls) for ls in self._cfg.train.lr_decay_epoch])
        lr_warmup = float(self._cfg.train.lr_warmup)  # avoid int division

        if self._cfg.train.verbose:
            self._logger.info('Trainable parameters:')
            self._logger.info(self.net.collect_train_params().keys())
        self._logger.info('Start training from [Epoch %d]', self._cfg.train.start_epoch)

        base_lr = trainer.learning_rate
        for epoch in range(self._cfg.train.start_epoch, self._cfg.train.epochs):
            self.epoch = epoch
            rcnn_task = ForwardBackwardTask(self.net, trainer, self.rpn_cls_loss, self.rpn_box_loss, self.rcnn_cls_loss,
                                            self.rcnn_box_loss, self.rcnn_mask_loss,
                                            amp_enabled=self._cfg.mask_rcnn.amp)
            executor = Parallel(self._cfg.train.executor_threads, rcnn_task) if not self._cfg.horovod else None
            if not self._cfg.disable_hybridization:
                self.net.hybridize(static_alloc=self._cfg.mask_rcnn.static_alloc)
            while lr_steps and epoch >= lr_steps[0]:
                new_lr = trainer.learning_rate * lr_decay
                lr_steps.pop(0)
                trainer.set_learning_rate(new_lr)
                self._logger.info("[Epoch %d] Set learning rate to %f", epoch, new_lr)
            for metric in self.metrics:
                metric.reset()
            tic = time.time()
            btic = time.time()
            train_data_iter = iter(self._train_data)
            next_data_batch = next(train_data_iter)
            next_data_batch = _split_and_load(next_data_batch, ctx_list=self.ctx)
            for i in range(len(self._train_data)):
                batch = next_data_batch
                if i + epoch * len(self._train_data) <= lr_warmup:
                    # adjust based on real percentage
                    new_lr = base_lr * _get_lr_at_iter((i + epoch * len(self._train_data)) / lr_warmup,
                                                       self._cfg.train.lr_warmup_factor)
                    if new_lr != trainer.learning_rate:
                        if i % self._cfg.train.log_interval == 0:
                            self._logger.info('[Epoch %d Iteration %d] Set learning rate to %f', epoch, i, new_lr)
                        trainer.set_learning_rate(new_lr)
                metric_losses = [[] for _ in self.metrics]
                add_losses = [[] for _ in self.metrics2]
                if executor is not None:
                    for data in zip(*batch):
                        executor.put(data)
                for _ in range(len(self.ctx)):
                    if executor is not None:
                        result = executor.get()
                    else:
                        result = rcnn_task.forward_backward(list(zip(*batch))[0])
                    if (not self._cfg.horovod) or hvd.rank() == 0:
                        for k in range(len(metric_losses)):
                            metric_losses[k].append(result[k])
                        for k in range(len(add_losses)):
                            add_losses[k].append(result[len(metric_losses) + k])
                try:
                    # prefetch next batch
                    next_data_batch = next(train_data_iter)
                    next_data_batch = _split_and_load(next_data_batch, ctx_list=self.ctx)
                except StopIteration:
                    pass

                for metric, record in zip(self.metrics, metric_losses):
                    metric.update(0, record)
                for metric, records in zip(self.metrics2, add_losses):
                    for pred in records:
                        metric.update(pred[0], pred[1])
                trainer.step(self.batch_size)
                if (not self._cfg.horovod or hvd.rank() == 0) and self._cfg.train.log_interval \
                        and not (i + 1) % self._cfg.train.log_interval:
                    msg = ','.join(['{}={:.3f}'.format(*metric.get()) for metric in self.metrics + self.metrics2])
                    self._logger.info('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}'.format(
                        epoch, i,
                        self._cfg.train.log_interval * self._cfg.train.batch_size / (time.time() - btic), msg))
                    btic = time.time()
            # validate and save params
            if (not self._cfg.horovod) or hvd.rank() == 0:
                msg = ','.join(['{}={:.3f}'.format(*metric.get()) for metric in self.metrics])
                self._logger.info('[Epoch {}] Training cost: {:.3f}, {}'.format(
                    epoch, (time.time() - tic), msg))
            if not (epoch + 1) % self._cfg.valid.val_interval:
                # consider reduce the frequency of validation to save time
                self._validate(self._val_data, self.async_eval_processes, self.ctx, self.eval_metric,
                               self._logger, epoch, self.best_map)
            elif (not self._cfg.horovod) or hvd.rank() == 0:
                current_map = 0.
                _save_params(self.net, self._logger, self.best_map, current_map, epoch, self._cfg.save_interval,
                             os.path.join(self._logdir, self._cfg.save_prefix))
            if self._reporter:
                self._reporter(epoch=epoch, map_reward=current_map)
        for thread in self.async_eval_processes:
            thread.join()

    def _evaluate(self, val_data):
        """Evaluate the current model on dataset.
        """
        # TODO(zhreshold): remove self._val_data, use passed in val_data at runtime
        return self._validate(self._val_data, self.async_eval_processes, self.ctx, self.eval_metric,
                              self._logger, self.epoch, self.best_map)

    def predict(self, x):
        """Predict an individual example.

        Parameters
        ----------
        x : file
            An image.
        """
        x, _ = presets.rcnn.transform_test(x, short=self.net.short, max_size=self.net.max_size)
        x = x.as_in_context(self.ctx[0])
        ids, scores, bboxes, masks = [xx[0].asnumpy() for xx in self.net(x)]
        return ids, scores, bboxes, masks
