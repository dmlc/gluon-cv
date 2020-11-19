"""Classification Estimator"""
# pylint: disable=unused-variable,bad-whitespace, missing-function-docstring,logging-format-interpolation
import time
import os
import math
import copy

import pandas as pd
import numpy as np
import mxnet as mx
from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon.data.vision import transforms
from ....data.transforms.presets.imagenet import transform_eval
from ....model_zoo import get_model
from ....utils import LRSequential, LRScheduler
from .... import nn
from .... import loss
from ..base_estimator import BaseEstimator, set_default
from .utils import get_data_loader, get_data_rec
from .default import ImageClassificationCfg

__all__ = ['ImageClassificationEstimator']


@set_default(ImageClassificationCfg())
class ImageClassificationEstimator(BaseEstimator):
    """Estimator implementation for Image Classification.

    Parameters
    ----------
    config : dict
        Config in nested dict.
    logger : logging.Logger
        Optional logger for this estimator, can be `None` when default setting is used.
    reporter : callable
        The reporter for metric checkpointing.
    """
    def __init__(self, config, logger=None, reporter=None):
        super(ImageClassificationEstimator, self).__init__(config, logger, reporter=reporter, name=None)
        self.last_train = None
        self.input_size = self._cfg.train.input_size
        self._feature_net = None

    def _fit(self, train_data, val_data):
        self._best_acc = 0
        self.epoch = 0
        self._time_elapsed = 0
        if max(self._cfg.train.start_epoch, self.epoch) >= self._cfg.train.epochs:
            return {'time', self._time_elapsed}
        if not isinstance(train_data, pd.DataFrame):
            self.last_train = len(train_data)
        else:
            self.last_train = train_data
        self._init_trainer()
        return self._resume_fit(train_data, val_data)

    def _resume_fit(self, train_data, val_data):
        if max(self._cfg.train.start_epoch, self.epoch) >= self._cfg.train.epochs:
            return {'time', self._time_elapsed}
        if not self.classes or not self.num_class:
            raise ValueError('Unable to determine classes of dataset')

        num_workers = self._cfg.train.num_workers
        if self._cfg.train.use_rec:
            self._logger.info(f'Loading data from rec files: {self._cfg.train.rec_train}/{self._cfg.train.rec_val}')
            train_loader, val_loader, self.batch_fn = get_data_rec(self._cfg.train.rec_train,
                                                                   self._cfg.train.rec_train_idx,
                                                                   self._cfg.train.rec_val,
                                                                   self._cfg.train.rec_val_idx,
                                                                   self.batch_size, num_workers,
                                                                   self.input_size,
                                                                   self._cfg.train.crop_ratio)
        else:
            train_dataset = train_data.to_mxnet()
            val_dataset = val_data.to_mxnet()
            train_loader, val_loader, self.batch_fn = get_data_loader(self._cfg.train.data_dir,
                                                                      self.batch_size, num_workers,
                                                                      self.input_size,
                                                                      self._cfg.train.crop_ratio,
                                                                      train_dataset=train_dataset,
                                                                      val_dataset=val_dataset)
        return self._train_loop(train_loader, val_loader)

    def _train_loop(self, train_data, val_data):
        if self._cfg.train.no_wd:
            for k, v in self.net.collect_params('.*beta|.*gamma|.*bias').items():
                v.wd_mult = 0.0
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

        if self._cfg.train.mixup:
            train_metric = mx.metric.RMSE()
        else:
            train_metric = mx.metric.Accuracy()
        if self._cfg.train.mode == 'hybrid':
            self.net.hybridize(static_alloc=True, static_shape=True)
            if self.distillation:
                self.teacher.hybridize(static_alloc=True, static_shape=True)

        self._logger.info('Start training from [Epoch %d]', max(self._cfg.train.start_epoch, self.epoch))
        for self.epoch in range(max(self._cfg.train.start_epoch, self.epoch), self._cfg.train.epochs):
            epoch = self.epoch
            tic = time.time()
            btic = time.time()
            if self._cfg.train.use_rec:
                train_data.reset()
            train_metric.reset()

            # pylint: disable=undefined-loop-variable
            for i, batch in enumerate(train_data):
                data, label = self.batch_fn(batch, self.ctx)

                if self._cfg.train.mixup:
                    lam = np.random.beta(self._cfg.train.mixup_alpha,
                                         self._cfg.train.mixup_alpha)
                    if epoch >= self._cfg.train.epochs - self._cfg.train.mixup_off_epoch:
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
                self.trainer.step(self.batch_size)

                if self._cfg.train.mixup:
                    output_softmax = [nd.SoftmaxActivation(out.astype('float32', copy=False)) \
                                    for out in outputs]
                    train_metric.update(label, output_softmax)
                else:
                    if self._cfg.train.label_smoothing:
                        train_metric.update(hard_label, outputs)
                    else:
                        train_metric.update(label, outputs)

                if self._cfg.train.log_interval and not (i+1)%self._cfg.train.log_interval:
                    train_metric_name, train_metric_score = train_metric.get()
                    self._logger.info('Epoch[%d] Batch [%d]\tSpeed: %f samples/sec\t%s=%f\tlr=%f',
                                      epoch, i,
                                      self._cfg.train.batch_size*self._cfg.train.log_interval/(time.time()-btic),
                                      train_metric_name, train_metric_score, self.trainer.learning_rate)
                    btic = time.time()

            train_metric_name, train_metric_score = train_metric.get()
            throughput = int(self.batch_size * i /(time.time() - tic))

            top1_val, top5_val = self._evaluate(val_data)

            self._logger.info('[Epoch %d] training: %s=%f', epoch, train_metric_name, train_metric_score)
            self._logger.info('[Epoch %d] speed: %d samples/sec\ttime cost: %f', epoch, throughput, time.time()-tic)
            self._logger.info('[Epoch %d] validation: top1=%f top5=%f', epoch, top1_val, top5_val)

            if top1_val > self._best_acc:
                cp_name = os.path.join(self._logdir, 'best_checkpoint.pkl')
                self._logger.info('[Epoch %d] Current best top-1: %f vs previous %f, saved to %s',
                                  self.epoch, top1_val, self._best_acc, cp_name)
                self.save(cp_name)
                self._best_acc = top1_val
            if self._reporter:
                self._reporter(epoch=epoch, acc_reward=top1_val)
            self._time_elapsed += time.time() - btic
        return {'train_acc': train_metric_score, 'valid_acc': self._best_acc, 'time': self._time_elapsed}

    def _init_network(self):
        if not self.num_class:
            raise ValueError('Unable to create network when `num_class` is unknown. \
                It should be inferred from dataset or resumed from saved states.')
        assert len(self.classes) == self.num_class
        # ctx
        self.ctx = [mx.gpu(int(i)) for i in self._cfg.gpus]
        self.ctx = self.ctx if self.ctx else [mx.cpu()]

        # network
        if isinstance(self._cfg.img_cls.model, str):
            model_name = self._cfg.img_cls.model.lower()
            input_size = self.input_size
            if 'inception' in model_name or 'googlenet' in model_name:
                self.input_size = 299
            elif 'resnest101' in model_name:
                self.input_size = 256
            elif 'resnest200' in model_name:
                self.input_size = 320
            elif 'resnest269' in model_name:
                self.input_size = 416
            elif 'cifar' in model_name:
                self.input_size = 28
        elif isinstance(self._cfg.img_cls.model, gluon.Block):
            self.net = self._cfg.img_cls.model
            model_name = ''
            self.input_size = self._cfg.train.input_size
        else:
            raise ValueError('Expected `model_name` to be (str, gluon.Block), given {}'.format(
                type(self._cfg.img_cls.model)))


        if input_size != self.input_size:
            self._logger.info(f'Change input size to {self.input_size}, given model type: {model_name}')

        if self._cfg.img_cls.use_pretrained:
            kwargs = {'ctx': self.ctx, 'pretrained': True, 'classes': 1000 if 'cifar' not in model_name else 10}
        else:
            kwargs = {'ctx': self.ctx, 'pretrained': False, 'classes': self.num_class}
        if self._cfg.img_cls.use_gn:
            kwargs['norm_layer'] = nn.GroupNorm
        if model_name.startswith('vgg'):
            kwargs['batch_norm'] = self._cfg.img_cls.batch_norm
        elif model_name.startswith('resnext'):
            kwargs['use_se'] = self._cfg.img_cls.use_se

        if self._cfg.img_cls.last_gamma:
            kwargs['last_gamma'] = True

        if self.net is None:
            self.net = get_model(model_name, **kwargs)
        if model_name and self._cfg.img_cls.use_pretrained:
            # reset last fully connected layer
            fc_layer_found = False
            for fc_name in ('output', 'fc'):
                fc_layer = getattr(self.net, fc_name, None)
                if fc_layer is not None:
                    fc_layer_found = True
                    break
            if fc_layer_found:
                in_channels = list(fc_layer.collect_params().values())[0].shape[1]
                if isinstance(fc_layer, gluon.nn.Dense):
                    new_fc_layer = gluon.nn.Dense(self.num_class, in_units=in_channels)
                elif isinstance(fc_layer, gluon.nn.Conv2D):
                    new_fc_layer = gluon.nn.Conv2D(self.num_class, in_channels=in_channels, kernel_size=1)
                else:
                    raise TypeError(f'Invalid FC layer type {type(fc_layer)} found, expected (Conv2D, Dense)...')
                new_fc_layer.initialize(mx.init.MSRAPrelu(), ctx=self.ctx)
                self.net.collect_params().setattr('lr_mult', self._cfg.train.transfer_lr_mult)
                new_fc_layer.collect_params().setattr('lr_mult', self._cfg.train.output_lr_mult)
                self._logger.debug(f'Reduce network lr multiplier to {self._cfg.train.transfer_lr_mult}, while keep ' +
                                   f'last FC layer lr_mult to {self._cfg.train.output_lr_mult}')
                setattr(self.net, fc_name, new_fc_layer)
            else:
                raise RuntimeError('Unable to modify the last fc layer in network, (output, fc) expected...')
        else:
            self.net.initialize(mx.init.MSRAPrelu(), ctx=self.ctx)
        self.net.cast(self._cfg.train.dtype)

        # teacher model for distillation training
        if self._cfg.train.teacher is not None and self._cfg.train.hard_weight < 1.0 and self.num_class == 1000:
            teacher_name = self._cfg.train.teacher
            self.teacher = get_model(teacher_name, pretrained=True, classes=self.num_class, ctx=self.ctx)
            self.teacher.cast(self._cfg.train.dtype)
            self.teacher.collect_params().reset_ctx(self.ctx)
            self.distillation = True
        else:
            self.distillation = False
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


        num_gpus = len(self.ctx)
        batch_size = self._cfg.train.batch_size
        self.batch_size = batch_size
        lr_decay = self._cfg.train.lr_decay
        lr_decay_period = self._cfg.train.lr_decay_period
        if self._cfg.train.lr_decay_period > 0:
            lr_decay_epoch = list(range(lr_decay_period, self._cfg.train.epochs, lr_decay_period))
        else:
            lr_decay_epoch = [int(i) for i in self._cfg.train.lr_decay_epoch.split(',')]
        lr_decay_epoch = [e - self._cfg.train.warmup_epochs for e in lr_decay_epoch]
        num_batches = train_size // batch_size

        lr_scheduler = LRSequential([
            LRScheduler('linear', base_lr=0, target_lr=self._cfg.train.lr,
                        nepochs=self._cfg.train.warmup_epochs, iters_per_epoch=num_batches),
            LRScheduler(self._cfg.train.lr_mode, base_lr=self._cfg.train.lr, target_lr=0,
                        nepochs=self._cfg.train.epochs - self._cfg.train.warmup_epochs,
                        iters_per_epoch=num_batches,
                        step_epoch=lr_decay_epoch,
                        step_factor=lr_decay, power=2)
        ])

        optimizer = 'nag'
        optimizer_params = {'wd': self._cfg.train.wd,
                            'momentum': self._cfg.train.momentum,
                            'lr_scheduler': lr_scheduler}
        if self._cfg.train.dtype != 'float32':
            optimizer_params['multi_precision'] = True
        self.trainer = gluon.Trainer(self.net.collect_params(), optimizer, optimizer_params)

    def _evaluate(self, val_data):
        """Test on validation dataset."""
        acc_top1 = mx.metric.Accuracy()
        acc_top5 = mx.metric.TopKAccuracy(min(5, self.num_class))

        if not isinstance(val_data, (gluon.data.DataLoader, mx.io.MXDataIter)):
            if hasattr(val_data, 'to_mxnet'):
                val_data = val_data.to_mxnet()
            resize = int(math.ceil(self.input_size / self._cfg.train.crop_ratio))
            normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            transform_test = transforms.Compose([
                transforms.Resize(resize, keep_ratio=True),
                transforms.CenterCrop(self.input_size),
                transforms.ToTensor(),
                normalize
            ])
            val_data = gluon.data.DataLoader(
                val_data.transform_first(transform_test),
                batch_size=self._cfg.valid.batch_size, shuffle=False, last_batch='keep',
                num_workers=self._cfg.valid.num_workers)
        for _, batch in enumerate(val_data):
            data, label = self.batch_fn(batch, self.ctx)
            outputs = [self.net(X.astype(self._cfg.train.dtype, copy=False)) for X in data]
            acc_top1.update(label, outputs)
            acc_top5.update(label, outputs)

        _, top1 = acc_top1.get()
        _, top5 = acc_top5.get()
        return top1, top5

    def _predict(self, x):
        resize = int(math.ceil(self.input_size / self._cfg.train.crop_ratio))
        if isinstance(x, str):
            x = transform_eval(mx.image.imread(x), resize_short=resize, crop_size=self.input_size)
        elif isinstance(x, mx.nd.NDArray):
            x = transform_eval(x, resize_short=resize, crop_size=self.input_size)
        elif isinstance(x, pd.DataFrame):
            assert 'image' in x.columns, "Expect column `image` for input images"
            def _predict_merge(x):
                y = self._predict(x)
                y['image'] = x
                return y
            return pd.concat([_predict_merge(xx) for xx in x['image']]).reset_index(drop=True)
        elif isinstance(x, (list, tuple)):
            return pd.concat([self._predict(xx) for xx in x]).reset_index(drop=True)
        else:
            raise ValueError('Input is not supported: {}'.format(type(x)))
        x = x.as_in_context(self.ctx[0])
        pred = self.net(x)
        topK = min(5, self.num_class)
        ind = nd.topk(pred, k=topK)[0].astype('int').asnumpy().flatten()
        probs = mx.nd.softmax(pred)[0].asnumpy().flatten()
        df = pd.DataFrame([{'class': self.classes[ind[i]], 'score': probs[ind[i]], 'id': ind[i]} for i in range(topK)])
        return df

    def _get_feature_net(self):
        """Get the network slice for feature extraction only"""
        if hasattr(self, '_feature_net') and self._feature_net is not None:
            return self._feature_net
        self._feature_net = copy.copy(self.net)
        fc_layer_found = False
        for fc_name in ('output', 'fc'):
            fc_layer = getattr(self._feature_net, fc_name, None)
            if fc_layer is not None:
                fc_layer_found = True
                break
        if fc_layer_found:
            self._feature_net.register_child(nn.Identity(), fc_name)
            super(gluon.Block, self._feature_net).__setattr__(fc_name, nn.Identity())
        else:
            raise RuntimeError('Unable to modify the last fc layer in network, (output, fc) expected...')
        return self._feature_net

    def _predict_feature(self, x):
        resize = int(math.ceil(self.input_size / self._cfg.train.crop_ratio))
        if isinstance(x, str):
            x = transform_eval(mx.image.imread(x), resize_short=resize, crop_size=self.input_size)
        elif isinstance(x, mx.nd.NDArray):
            x = transform_eval(x, resize_short=resize, crop_size=self.input_size)
        elif isinstance(x, pd.DataFrame):
            assert 'image' in x.columns, "Expect column `image` for input images"
            def _predict_merge(x):
                y = self._predict_feature(x)
                y['image'] = x
                return y
            return pd.concat([_predict_merge(xx) for xx in x['image']]).reset_index(drop=True)
        elif isinstance(x, (list, tuple)):
            return pd.concat([self._predict_feature(xx) for xx in x]).reset_index(drop=True)
        else:
            raise ValueError('Input is not supported: {}'.format(type(x)))
        x = x.as_in_context(self.ctx[0])
        feat_net = self._get_feature_net()
        feat = feat_net(x)[0].asnumpy().flatten()
        df = pd.DataFrame({'image_feature': [feat]})
        return df
