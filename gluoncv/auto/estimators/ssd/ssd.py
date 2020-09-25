"""SSD Estimator."""
# pylint: disable=logging-format-interpolation
import os
import logging
import warnings
import time
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet import autograd
from mxnet.contrib import amp

from .... import utils as gutils
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
        super(SSDEstimator, self).__init__(config, logger, reporter)

        if self._cfg.ssd.amp:
            amp.init()

        if self._cfg.horovod:
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
        devices = [int(i) for i in self._cfg.gpus]
        if self._cfg.train.dali:
            if not dali_found:
                raise SystemExit("DALI not found, please check if you installed it correctly.")
            self.train_dataset, self.val_dataset, self.eval_metric = _get_dali_dataset(self._cfg.dataset, devices,
                                                                                       self._cfg)
        else:
            self.train_dataset, self.val_dataset, self.eval_metric = _get_dataset(self._cfg.dataset, self._cfg)

        # network
        net_name = '_'.join(('ssd', str(self._cfg.ssd.data_shape), self._cfg.ssd.backbone, self._cfg.dataset))
        self._cfg.save_prefix += net_name

        if self._cfg.ssd.custom_model:
            classes = self.train_dataset.CLASSES
            if self._cfg.ssd.syncbn and len(self.ctx) > 1:
                self.net = custom_ssd(base_network_name=self._cfg.ssd.backbone,
                                      base_size=self._cfg.ssd.data_shape,
                                      filters=self._cfg.ssd.filters,
                                      sizes=self._cfg.ssd.sizes,
                                      ratios=self._cfg.ssd.ratios,
                                      steps=self._cfg.ssd.steps,
                                      classes=classes,
                                      dataset=self._cfg.dataset,
                                      pretrained_base=True,
                                      norm_layer=gluon.contrib.nn.SyncBatchNorm,
                                      norm_kwargs={'num_devices': len(self.ctx)})
                self.async_net = custom_ssd(base_network_name=self._cfg.ssd.backbone,
                                            base_size=self._cfg.ssd.data_shape,
                                            filters=self._cfg.ssd.filters,
                                            sizes=self._cfg.ssd.sizes,
                                            ratios=self._cfg.ssd.ratios,
                                            steps=self._cfg.ssd.steps,
                                            classes=classes,
                                            dataset=self._cfg.dataset,
                                            pretrained_base=False)
            else:
                self.net = custom_ssd(base_network_name=self._cfg.ssd.backbone,
                                      base_size=self._cfg.ssd.data_shape,
                                      filters=self._cfg.ssd.filters,
                                      sizes=self._cfg.ssd.sizes,
                                      ratios=self._cfg.ssd.ratios,
                                      steps=self._cfg.ssd.steps,
                                      classes=classes,
                                      dataset=self._cfg.dataset,
                                      pretrained_base=True,
                                      norm_layer=gluon.nn.BatchNorm)
                self.async_net = self.net
        else:
            if self._cfg.ssd.syncbn and len(self.ctx) > 1:
                self.net = get_model(net_name, pretrained_base=True, norm_layer=gluon.contrib.nn.SyncBatchNorm,
                                     norm_kwargs={'num_devices': len(self.ctx)})
                self.async_net = get_model(net_name, pretrained_base=False)  # used by cpu worker
            else:
                self.net = get_model(net_name, pretrained_base=True, norm_layer=gluon.nn.BatchNorm)
                self.async_net = self.net

        if self._cfg.resume.strip():
            self.net.load_parameters(self._cfg.resume.strip())
            self.async_net.load_parameters(self._cfg.resume.strip())
        else:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                self.net.initialize()
                self.async_net.initialize()
                # needed for net to be first gpu when using AMP
                self.net.collect_params().reset_ctx(self.ctx[0])

        # training dataloader
        if self._cfg.train.dali:
            if not dali_found:
                raise SystemExit("DALI not found, please check if you installed it correctly.")
            self._train_data, self._val_data = _get_dali_dataloader(
                self.async_net, self.train_dataset, self.val_dataset, self._cfg.ssd.data_shape,
                self._cfg.train.batch_size, self._cfg.num_workers,
                devices, self.ctx[0], self._cfg.horovod)
        else:
            self.batch_size = self._cfg.train.batch_size // hvd.size() \
                if self._cfg.horovod else self._cfg.train.batch_size
            self._train_data, self._val_data = _get_dataloader(
                self.async_net, self.train_dataset, self.val_dataset, self._cfg.ssd.data_shape,
                self.batch_size, self._cfg.num_workers, self.ctx[0])

        self.mbox_loss = SSDMultiBoxLoss()
        self.ce_metric = mx.metric.Loss('CrossEntropy')
        self.smoothl1_metric = mx.metric.Loss('SmoothL1')

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
        self.net.hybridize(static_alloc=True, static_shape=True)
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
        """
        Fit SSD models.
        """
        self.net.collect_params().reset_ctx(self.ctx)

        if self._cfg.horovod:
            hvd.broadcast_parameters(self.net.collect_params(), root_rank=0)
            trainer = hvd.DistributedTrainer(
                self.net.collect_params(), 'sgd',
                {'learning_rate': self._cfg.train.lr, 'wd': self._cfg.train.wd,
                 'momentum': self._cfg.train.momentum})
        else:
            trainer = gluon.Trainer(
                self.net.collect_params(), 'sgd',
                {'learning_rate': self._cfg.train.lr, 'wd': self._cfg.train.wd,
                 'momentum': self._cfg.train.momentum},
                update_on_kvstore=(False if self._cfg.ssd.amp else None))

        if self._cfg.ssd.amp:
            amp.init_trainer(trainer)

        # lr decay policy
        lr_decay = float(self._cfg.train.lr_decay)
        lr_steps = sorted([float(ls) for ls in self._cfg.train.lr_decay_epoch])

        self._logger.info('Start training from [Epoch {}]'.format(self._cfg.train.start_epoch))
        best_map = [0]

        for epoch in range(self._cfg.train.start_epoch, self._cfg.train.epochs):
            while lr_steps and epoch >= lr_steps[0]:
                new_lr = trainer.learning_rate * lr_decay
                lr_steps.pop(0)
                trainer.set_learning_rate(new_lr)
                self._logger.info("[Epoch {}] Set learning rate to {}".format(epoch, new_lr))
            self.ce_metric.reset()
            self.smoothl1_metric.reset()
            tic = time.time()
            btic = time.time()
            self.net.hybridize(static_alloc=True, static_shape=True)

            for i, batch in enumerate(self._train_data):
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
                    sum_loss, cls_loss, box_loss = self.mbox_loss(
                        cls_preds, box_preds, cls_targets, box_targets)
                    if self._cfg.ssd.amp:
                        with amp.scale_loss(sum_loss, trainer) as scaled_loss:
                            autograd.backward(scaled_loss)
                    else:
                        autograd.backward(sum_loss)
                # since we have already normalized the loss, we don't want to normalize
                # by batch-size anymore
                trainer.step(1)

                if (not self._cfg.horovod or hvd.rank() == 0):
                    local_batch_size = int(self._cfg.train.batch_size // (hvd.size() if self._cfg.horovod else 1))
                    self.ce_metric.update(0, [l * local_batch_size for l in cls_loss])
                    self.smoothl1_metric.update(0, [l * local_batch_size for l in box_loss])
                    if self._cfg.train.log_interval and not (i + 1) % self._cfg.train.log_interval:
                        name1, loss1 = self.ce_metric.get()
                        name2, loss2 = self.smoothl1_metric.get()
                        self._logger.info(
                            '[Epoch %d][Batch %d], Speed: %f samples/sec, %s=%f, %s=%f',
                            epoch, i, self._cfg.train.batch_size/(time.time()-btic), name1, loss1, name2, loss2)
                    btic = time.time()

            if (not self._cfg.horovod or hvd.rank() == 0):
                name1, loss1 = self.ce_metric.get()
                name2, loss2 = self.smoothl1_metric.get()
                self._logger.info('[Epoch %d] Training cost: %f, %s=%f, %s=%f',
                                  epoch, (time.time()-tic), name1, loss1, name2, loss2)
                if (epoch % self._cfg.validation.val_interval == 0) or \
                    (self._cfg.save_interval and epoch % self._cfg.save_interval == 0):
                    # consider reduce the frequency of validation to save time
                    map_name, mean_ap = self._validate(self._val_data, self.ctx, self.eval_metric)
                    val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
                    self._logger.info('[Epoch %d] Validation: \n%s', epoch, str(val_msg))
                    current_map = float(mean_ap[-1])
                else:
                    current_map = 0.
                _save_params(self.net, best_map, current_map, epoch, self._cfg.save_interval,
                             os.path.join(self._logdir, self._cfg.save_prefix))
                if self._reporter:
                    self._reporter(epoch=epoch, map_reward=current_map)

    def _evaluate(self):
        """Evaluate the current model on dataset.
        """
        self.net.collect_params().reset_ctx(self.ctx)
        return self._validate(self._val_data, self.ctx, self.eval_metric)

    def predict(self, x):
        """Predict an individual example.

        Parameters
        ----------
        x : file
            An image.
        """
        x, _ = presets.ssd.transform_test(x, short=512)
        x = x.as_in_context(self.ctx[0])
        ids, scores, bboxes = [xx[0].asnumpy() for xx in self.net(x)]
        return ids, scores, bboxes

    def put_parameters(self, parameters, multi_precision=False):
        """Load saved parameters into the model"""
        param_dict = self.net._collect_params_with_prefix()
        kwargs = {'ctx': None} if mx.__version__[:3] == '1.4' else {'cast_dtype': multi_precision,
                                                                    'ctx': None}
        for k, _ in param_dict.items():
            param_dict[k]._load_init(parameters[k], **kwargs)

    def get_parameters(self):
        """Return model parameters"""
        return self.net._collect_params_with_prefix()

    def save(self, filename):
        # TODO(): remove this part once dataloader is no longer attributes
        if getattr(self, '_train_data'):
            train_data = self._train_data
            self._train_data = None
        if getattr(self, '_val_data'):
            val_data = self._val_data
            self._val_data = None
        self.net.collect_params().reset_ctx(mx.cpu())
        super().save(filename)
        self._train_data = train_data
        self._val_data = val_data
