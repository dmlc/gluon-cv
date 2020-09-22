"""Classification Estimator"""
# pylint: disable=unused-variable,bad-whitespace, missing-function-docstring
import time
import logging
import os
import math

import numpy as np
import mxnet as mx
from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon.data.vision import transforms
from sacred import Experiment, Ingredient

from ...data import imagenet
from ...model_zoo import get_model
from ...utils import makedirs, LRSequential, LRScheduler
from ... import nn
from ... import loss
from .base_estimator import BaseEstimator, set_default

__all__ = ['ClassificationEstimator']


cls_net = Ingredient('cls_net')
train = Ingredient('train')
validation = Ingredient('validation')

@cls_net.config
def cls_net_default():
    model = 'resnet50_v1'
    use_pretrained = False
    use_gn = False
    batch_norm = False
    use_se = False
    last_gamma = False

@train.config
def train_config():
    gpus = (0,)
    pretrained_base = True  # whether load the imagenet pre-trained base
    batch_size = 32
    num_epochs = 3
    lr = 0.1  # learning rate
    lr_decay = 0.1  # decay rate of learning rate.
    lr_decay_period = 0
    lr_decay_epoch = '40, 60'  # epochs at which learning rate decays
    lr_mode = 'step'  # learning rate scheduler mode. options are step, poly and cosine
    warmup_lr = 0.0  # starting warmup learning rate.
    warmup_epochs = 0  # number of warmup epochs
    classes = 1000
    num_training_samples = 1281167
    num_workers = 4
    wd = 0.0001
    momentum = 0.9
    resume_params = ''
    teacher = None
    hard_weight = 0.5
    dtype = 'float32'
    input_size = 224
    crop_ratio = 0.875
    use_rec = False
    rec_train = '~/.mxnet/datasets/imagenet/rec/train.rec'
    rec_train_idx = '~/.mxnet/datasets/imagenet/rec/train.idx'
    rec_val = '~/.mxnet/datasets/imagenet/rec/val.rec'
    rec_val_idx = '~/.mxnet/datasets/imagenet/rec/val.idx'
    data_dir = '~/.mxnet/datasets/imagenet'
    mixup = False
    save_frequency = 10
    save_dir = 'params'
    logging_file = 'train_imagenet.log'
    no_wd = False
    resume_states = ''
    label_smoothing = False
    temperature = 20
    hard_weight = 0.5
    resume_epoch = 0
    mixup_alpha = 0.2
    mixup_off_epoch = 0
    log_interval = 50
    mode = ''


@validation.config
def valid_config():
    test = 1

ex = Experiment('cls_net_default',
                ingredients=[train, validation, cls_net])




@set_default(ex)
class ClassificationEstimator(BaseEstimator):
    """Estimator implementation for Image Classification.

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
    def __init__(self, config, logger=None):
        super(ClassificationEstimator, self).__init__(config, logger)

        filehandler = logging.FileHandler(self._cfg.train.logging_file)
        streamhandler = logging.StreamHandler()

        self.logger = logging.getLogger('')
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(filehandler)
        self.logger.addHandler(streamhandler)


        batch_size = self._cfg.train.batch_size
        classes = self._cfg.train.classes
        num_training_samples = self._cfg.train.num_training_samples


        num_gpus = len(self._cfg.train.gpus)
        batch_size *= max(1, num_gpus)
        self.ctx = [mx.gpu(int(i)) for i in self._cfg.train.gpus]
        self.ctx = self.ctx if self.ctx else [mx.cpu()]
        num_workers = self._cfg.train.num_workers

        lr_decay = self._cfg.train.lr_decay
        lr_decay_period = self._cfg.train.lr_decay_period
        if self._cfg.train.lr_decay_period > 0:
            lr_decay_epoch = list(range(lr_decay_period, self._cfg.train.num_epochs, lr_decay_period))
        else:
            lr_decay_epoch = [int(i) for i in self._cfg.train.lr_decay_epoch.split(',')]
        lr_decay_epoch = [e - self._cfg.train.warmup_epochs for e in lr_decay_epoch]
        num_batches = num_training_samples // batch_size

        lr_scheduler = LRSequential([
            LRScheduler('linear', base_lr=0, target_lr=self._cfg.train.lr,
                        nepochs=self._cfg.train.warmup_epochs, iters_per_epoch=num_batches),
            LRScheduler(self._cfg.train.lr_mode, base_lr=self._cfg.train.lr, target_lr=0,
                        nepochs=self._cfg.train.num_epochs - self._cfg.train.warmup_epochs,
                        iters_per_epoch=num_batches,
                        step_epoch=lr_decay_epoch,
                        step_factor=lr_decay, power=2)
        ])

        self.model_name = self._cfg.cls_net.model

        kwargs = {'ctx': self.ctx, 'pretrained': self._cfg.cls_net.use_pretrained, 'classes': classes}
        if self._cfg.cls_net.use_gn:
            kwargs['norm_layer'] = nn.GroupNorm
        if self.model_name.startswith('vgg'):
            kwargs['batch_norm'] = self._cfg.cls_net.batch_norm
        elif self.model_name.startswith('resnext'):
            kwargs['use_se'] = self._cfg.cls_net.use_se

        if self._cfg.cls_net.last_gamma:
            kwargs['last_gamma'] = True


        self.optimizer = 'nag'
        self.optimizer_params = {'wd': self._cfg.train.wd,
                                 'momentum': self._cfg.train.momentum,
                                 'lr_scheduler': lr_scheduler}
        if self._cfg.train.dtype != 'float32':
            optimizer_params['multi_precision'] = True

        self.net = get_model(self.model_name, **kwargs)
        self.net.cast(self._cfg.train.dtype)
        if self._cfg.train.resume_params != '':
            net.load_parameters(self._cfg.train.resume_params, ctx=self.ctx)

        # teacher model for distillation training
        if self._cfg.train.teacher is not None and self._cfg.train.hard_weight < 1.0:
            teacher_name = self._cfg.train.teacher
            self.teacher = get_model(teacher_name, pretrained=True, classes=classes, ctx=self.ctx)
            self.teacher.cast(self._cfg.train.dtype)
            self.distillation = True
        else:
            self.distillation = False




        def get_data_rec(rec_train, rec_train_idx, rec_val, rec_val_idx, batch_size, num_workers):
            rec_train = os.path.expanduser(rec_train)
            rec_train_idx = os.path.expanduser(rec_train_idx)
            rec_val = os.path.expanduser(rec_val)
            rec_val_idx = os.path.expanduser(rec_val_idx)
            jitter_param = 0.4
            lighting_param = 0.1
            input_size = self._cfg.train.input_size
            crop_ratio = self._cfg.train.crop_ratio if self._cfg.train.crop_ratio > 0 else 0.875
            resize = int(math.ceil(input_size / crop_ratio))
            mean_rgb = [123.68, 116.779, 103.939]
            std_rgb = [58.393, 57.12, 57.375]

            def batch_fn(batch, ctx):
                data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
                label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
                return data, label

            train_data = mx.io.ImageRecordIter(
                path_imgrec         = rec_train,
                path_imgidx         = rec_train_idx,
                preprocess_threads  = num_workers,
                shuffle             = True,
                batch_size          = batch_size,

                data_shape          = (3, input_size, input_size),
                mean_r              = mean_rgb[0],
                mean_g              = mean_rgb[1],
                mean_b              = mean_rgb[2],
                std_r               = std_rgb[0],
                std_g               = std_rgb[1],
                std_b               = std_rgb[2],
                rand_mirror         = True,
                random_resized_crop = True,
                max_aspect_ratio    = 4. / 3.,
                min_aspect_ratio    = 3. / 4.,
                max_random_area     = 1,
                min_random_area     = 0.08,
                brightness          = jitter_param,
                saturation          = jitter_param,
                contrast            = jitter_param,
                pca_noise           = lighting_param,
            )
            val_data = mx.io.ImageRecordIter(
                path_imgrec         = rec_val,
                path_imgidx         = rec_val_idx,
                preprocess_threads  = num_workers,
                shuffle             = False,
                batch_size          = batch_size,

                resize              = resize,
                data_shape          = (3, input_size, input_size),
                mean_r              = mean_rgb[0],
                mean_g              = mean_rgb[1],
                mean_b              = mean_rgb[2],
                std_r               = std_rgb[0],
                std_g               = std_rgb[1],
                std_b               = std_rgb[2],
            )
            return train_data, val_data, batch_fn

        def get_data_loader(data_dir, batch_size, num_workers):
            normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            jitter_param = 0.4
            lighting_param = 0.1
            input_size = self._cfg.train.input_size
            crop_ratio = self._cfg.train.crop_ratio if self._cfg.train.crop_ratio > 0 else 0.875
            resize = int(math.ceil(input_size / crop_ratio))

            def batch_fn(batch, ctx):
                data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
                label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
                return data, label

            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(input_size),
                transforms.RandomFlipLeftRight(),
                transforms.RandomColorJitter(brightness=jitter_param, contrast=jitter_param,
                                             saturation=jitter_param),
                transforms.RandomLighting(lighting_param),
                transforms.ToTensor(),
                normalize
            ])
            transform_test = transforms.Compose([
                transforms.Resize(resize, keep_ratio=True),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize
            ])

            train_data = gluon.data.DataLoader(
                imagenet.classification.ImageNet(data_dir, train=True).transform_first(transform_train),
                batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=num_workers)
            val_data = gluon.data.DataLoader(
                imagenet.classification.ImageNet(data_dir, train=False).transform_first(transform_test),
                batch_size=batch_size, shuffle=False, num_workers=num_workers)

            return train_data, val_data, batch_fn

        if self._cfg.train.use_rec:
            self.train_data, self.val_data, self.batch_fn = get_data_rec(self._cfg.train.rec_train,
                                                                         self._cfg.train.rec_train_idx,
                                                                         self._cfg.train.rec_val,
                                                                         self._cfg.train.rec_val_idx,
                                                                         batch_size, num_workers)
        else:
            self.train_data, self.val_data, self.batch_fn = get_data_loader(self._cfg.train.data_dir,
                                                                            batch_size, num_workers)

        if self._cfg.train.mixup:
            self.train_metric = mx.metric.RMSE()
        else:
            self.train_metric = mx.metric.Accuracy()
        self.acc_top1 = mx.metric.Accuracy()
        self.acc_top5 = mx.metric.TopKAccuracy(5)

        save_frequency = self._cfg.train.save_frequency
        if self._cfg.train.save_dir and save_frequency:
            save_dir = self._cfg.train.save_dir
            makedirs(save_dir)
        else:
            save_dir = ''
            save_frequency = 0



        if self._cfg.train.mode == 'hybrid':
            self.net.hybridize(static_alloc=True, static_shape=True)
            if self.distillation:
                self.teacher.hybridize(static_alloc=True, static_shape=True)

        self.batch_size = batch_size
        self.save_dir = save_dir

    def _fit(self):
        ctx = self.ctx
        if isinstance(ctx, mx.Context):
            ctx = [ctx]
        if self._cfg.train.resume_params == '':
            self.net.initialize(mx.init.MSRAPrelu(), ctx=ctx)

        if self._cfg.train.no_wd:
            for k, v in self.net.collect_params('.*beta|.*gamma|.*bias').items():
                v.wd_mult = 0.0

        trainer = gluon.Trainer(self.net.collect_params(), self.optimizer, self.optimizer_params)
        if self._cfg.train.resume_states != '':
            trainer.load_states(self._cfg.train.resume_states)

        if self._cfg.train.label_smoothing or self._cfg.train.mixup:
            sparse_label_loss = False
        else:
            sparse_label_loss = True
        if self.distillation:
            L = loss.DistillationSoftmaxCrossEntropyLoss(temperature=self._cfg.train.temperature,
                                                         hard_weight=self._cfg.train.hard_weight,
                                                         sparse_label=sparse_label_loss)
        else:
            L = gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=sparse_label_loss)

        best_val_score = 1

        def mixup_transform(label, classes, lam=1, eta=0.0):
            if isinstance(label, nd.NDArray):
                label = [label]
            res = []
            for l in label:
                y1 = l.one_hot(classes, on_value=1 - eta + eta/classes, off_value=eta/classes)
                y2 = l[::-1].one_hot(classes, on_value=1 - eta + eta/classes, off_value=eta/classes)
                res.append(lam*y1 + (1-lam)*y2)
            return res

        def smooth(label, classes, eta=0.1):
            if isinstance(label, nd.NDArray):
                label = [label]
            smoothed = []
            for l in label:
                res = l.one_hot(classes, on_value=1 - eta + eta/classes, off_value=eta/classes)
                smoothed.append(res)
            return smoothed

        for epoch in range(self._cfg.train.resume_epoch, self._cfg.train.num_epochs):
            tic = time.time()
            if self._cfg.train.use_rec:
                self.train_data.reset()
            self.train_metric.reset()
            btic = time.time()

            # pylint: disable=undefined-loop-variable
            for i, batch in enumerate(self.train_data):
                data, label = self.batch_fn(batch, ctx)

                if self._cfg.train.mixup:
                    lam = np.random.beta(self._cfg.train.mixup_alpha,
                                         self._cfg.train.mixup_alpha)
                    if epoch >= self._cfg.train.num_epochs - self._cfg.train.mixup_off_epoch:
                        lam = 1
                    data = [lam*X + (1-lam)*X[::-1] for X in data]

                    if self._cfg.train.label_smoothing:
                        eta = 0.1
                    else:
                        eta = 0.0
                    label = mixup_transform(label, classes, lam, eta)

                elif self._cfg.train.label_smoothing:
                    hard_label = label
                    label = smooth(label, classes)

                if self.distillation:
                    teacher_prob = [nd.softmax(self.teacher(X.astype(self._cfg.train.dtype, copy=False)) \
                                    / self._cfg.train.temperature) for X in data]

                with ag.record():
                    outputs = [self.net(X.astype(self._cfg.train.dtype, copy=False)) for X in data]
                    if self.distillation:
                        losses = [L(yhat.astype('float32', copy=False),
                                    y.astype('float32', copy=False),
                                    p.astype('float32', copy=False)) \
                                        for yhat, y, p in zip(outputs, label, teacher_prob)]
                    else:
                        losses = [L(yhat,
                                    y.astype(self._cfg.train.dtype, copy=False)) for yhat, y in zip(outputs, label)]
                for l in losses:
                    l.backward()
                trainer.step(self.batch_size)

                if self._cfg.train.mixup:
                    output_softmax = [nd.SoftmaxActivation(out.astype('float32', copy=False)) \
                                    for out in outputs]
                    self.train_metric.update(label, output_softmax)
                else:
                    if self._cfg.train.label_smoothing:
                        self.train_metric.update(hard_label, outputs)
                    else:
                        self.train_metric.update(label, outputs)

                if self._cfg.train.log_interval and not (i+1)%self._cfg.train.log_interval:
                    train_metric_name, train_metric_score = self.train_metric.get()
                    self.logger.info('Epoch[%d] Batch [%d]\tSpeed: %f samples/sec\t%s=%f\tlr=%f',
                                     epoch, i,
                                     self._cfg.train.batch_size*self._cfg.train.log_interval/(time.time()-btic),
                                     train_metric_name, train_metric_score, trainer.learning_rate)
                    btic = time.time()

            train_metric_name, train_metric_score = self.train_metric.get()
            throughput = int(self.batch_size * i /(time.time() - tic))

            err_top1_val, err_top5_val = self._evaluate()

            self.logger.info('[Epoch %d] training: %s=%f', epoch, train_metric_name, train_metric_score)
            self.logger.info('[Epoch %d] speed: %d samples/sec\ttime cost: %f', epoch, throughput, time.time()-tic)
            self.logger.info('[Epoch %d] validation: err-top1=%f err-top5=%f', epoch, err_top1_val, err_top5_val)

            if err_top1_val < best_val_score:
                best_val_score = err_top1_val
                self.net.save_parameters(
                    '%s/%.4f-imagenet-%s-%d-best.params'%(self.save_dir, best_val_score, model_name, epoch))
                trainer.save_states(
                    '%s/%.4f-imagenet-%s-%d-best.states'%(self.save_dir, best_val_score, model_name, epoch))

            if save_frequency and save_dir and (epoch + 1) % save_frequency == 0:
                net.save_parameters('%s/imagenet-%s-%d.params'%(self.save_dir, self.model_name, epoch))
                trainer.save_states('%s/imagenet-%s-%d.states'%(self.save_dir, self.model_name, epoch))

        if save_frequency and save_dir:
            net.save_parameters('%s/imagenet-%s-%d.params'%(self.save_dir, self.model_name, opt.num_epochs-1))
            trainer.save_states('%s/imagenet-%s-%d.states'%(self.save_dir, self.model_name, opt.num_epochs-1))

    def _evaluate(self):
        """Test on validation dataset."""
        if self._cfg.train.use_rec:
            self.val_data.reset()
        self.acc_top1.reset()
        self.acc_top5.reset()
        for _, batch in enumerate(self.val_data):
            data, label = self.batch_fn(batch, self.ctx)
            outputs = [self.net(X.astype(self._cfg.train.dtype, copy=False)) for X in data]
            self.acc_top1.update(label, outputs)
            self.acc_top5.update(label, outputs)

        _, top1 = self.acc_top1.get()
        _, top5 = self.acc_top5.get()
        return (1-top1, 1-top5)


@ex.automain
def main(_config, _log):
    # main is the commandline entry for user w/o coding
    c = ClassificationEstimator(_config, _log)
    c.fit()
