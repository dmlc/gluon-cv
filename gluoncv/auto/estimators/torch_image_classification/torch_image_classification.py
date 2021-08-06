"""Torch Classification Estimator"""
# pylint: disable=unused-variable,bad-whitespace,missing-function-docstring,logging-format-interpolation,arguments-differ,logging-not-lazy, not-callable
import math
import copy
import os
import logging
import time
import warnings
import pickle
from contextlib import suppress
from PIL import Image

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils
from torch.optim.optimizer import Optimizer

from timm.data import create_loader, Mixup, FastCollateMixup, AugMixDataset
from timm.models import create_model, safe_model_name, convert_splitbn_model, model_parameters
from timm.utils import random_seed, dispatch_clip_grad, accuracy, unwrap_model, get_state_dict
from timm.optim import create_optimizer_v2
from timm.utils import ApexScaler, NativeScaler, ModelEmaV2, AverageMeter
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from .default import TorchImageClassificationCfg
from .utils import resolve_data_config, update_cfg, optimizer_kwargs, \
                   create_scheduler, rmse, create_optimizer_v2a
from ..utils import EarlyStopperOnPlateau
from ..conf import _BEST_CHECKPOINT_FILE
from ..base_estimator import BaseEstimator, set_default
from ....utils.filesystem import try_import
problem_type_constants = try_import(package='autogluon.core.constants',
                                    fromlist=['MULTICLASS', 'BINARY', 'REGRESSION'],
                                    message='Failed to import problem type constants from autogluon.core.')
MULTICLASS = problem_type_constants.MULTICLASS
BINARY = problem_type_constants.BINARY
REGRESSION = problem_type_constants.REGRESSION

warnings.filterwarnings('ignore', message='.*Argument interpolation should be of type InterpolationMode instead of int.*')
logging.getLogger('timm.models').setLevel(logging.WARNING)

try:
    from apex import amp
    from apex.parallel import convert_syncbn_model
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass


@set_default(TorchImageClassificationCfg())
class TorchImageClassificationEstimator(BaseEstimator):
    """Torch Estimator implementation for Image Classification.

    Parameters
    ----------
    config : dict
        Config in nested dict.
    logger : logging.Logger
        Optional logger for this estimator, can be `None` when default setting is used.
    reporter : callable
        The reporter for metric checkpointing.
    net : torch.nn.Module
        The custom network. If defined, the model name in config will be ignored so your
        custom network will be used for training rather than pulling it from model zoo.
    """
    def __init__(self, config, logger=None, reporter=None, net=None, optimizer=None, problem_type=None):
        super().__init__(config, logger=logger, reporter=reporter, name=None)
        if problem_type is None:
            problem_type = MULTICLASS
        self._problem_type = problem_type
        self._feature_net = None
        self._custom_net = False

        self._img_cls_cfg = self._cfg.img_cls
        self._data_cfg = self._cfg.data
        self._optimizer_cfg = self._cfg.optimizer
        self._train_cfg = self._cfg.train
        self._augmentation_cfg = self._cfg.augmentation
        self._model_ema_cfg = self._cfg.model_ema
        self._misc_cfg = self._cfg.misc

        # resolve AMP arguments based on PyTorch / Apex availability
        self.use_amp = None
        if self._misc_cfg.amp:
            # `amp` chooses native amp before apex (APEX ver not actively maintained)
            if self._misc_cfg.native_amp and has_native_amp:
                self.use_amp = 'native'
            elif self._misc_cfg.apex_amp and has_apex:
                self.use_amp = 'apex'
            elif self._misc_cfg.apex_amp or self._misc_cfg.native_amp:
                self._logger.warning(f'Neither APEX or native Torch AMP is available, using float32. \
                                       Install NVIDA apex or upgrade to PyTorch 1.6')
        # FIXME: will provided model conflict with config provided?
        if net is not None:
            assert isinstance(net, nn.Module), f"given custom network {type(net)}, `torch.nn` expected"
            try:
                net.to('cpu')
                self._custom_net = True
            except ValueError:
                pass
        self.net = net
        if optimizer is not None:
            self._logger.warning('Custom optimizer object not supported. Will follow the config instead.')
        self._optimizer = None

    def _fit(self, train_data, val_data, time_limit=math.inf):
        tic = time.time()
        self._cp_name = ''
        self._best_acc = -float('inf')
        self.epochs = self._train_cfg.epochs
        self.epoch = 0
        self.start_epoch = self._train_cfg.start_epoch
        self._time_elapsed = 0
        if max(self.start_epoch, self.epoch) >= self.epochs:
            return {'time', self._time_elapsed}
        self._init_trainer()
        self._init_model_ema()
        self._time_elapsed += time.time() - tic
        return self._resume_fit(train_data, val_data, time_limit=time_limit)

    def _resume_fit(self, train_data, val_data, time_limit=math.inf):
        tic = time.time()
        # TODO: regression not implemented
        if self._problem_type != REGRESSION and (not self.classes or not self.num_class):
            raise ValueError('This is a classification problem and we are not able to determine classes of dataset')

        if max(self.start_epoch, self.epoch) >= self.epochs:
            return {'time': self._time_elapsed}

        # wrap DP if possible
        if self.found_gpu:
            self.net = torch.nn.DataParallel(self.net, device_ids=[int(i) for i in self.valid_gpus])
        self.net = self.net.to(self.ctx[0])

        # prepare dataset
        train_dataset = train_data.to_torch()
        val_dataset = val_data.to_torch()

        # setup mixup / cutmix
        self._collate_fn = None
        self._mixup_fn = None
        self.mixup_active = self._augmentation_cfg.mixup > 0 or self._augmentation_cfg.cutmix > 0. or self._augmentation_cfg.cutmix_minmax is not None
        if self.mixup_active:
            mixup_args = dict(
                mixup_alpha=self._augmentation_cfg.mixup, cutmix_alpha=self._augmentation_cfg.cutmix,
                cutmix_minmax=self._augmentation_cfg.cutmix_minmax, prob=self._augmentation_cfg.mixup_prob,
                switch_prob=self._augmentation_cfg.mixup_switch_prob, mode=self._augmentation_cfg.mixup_mode,
                label_smoothing=self._augmentation_cfg.smoothing, num_classes=self.num_class)
            if self._misc_cfg.prefetcher:
                self._collate_fn = FastCollateMixup(**mixup_args)
            else:
                self._mixup_fn = Mixup(**mixup_args)

        # create data loaders w/ augmentation pipeiine
        train_interpolation = self._augmentation_cfg.train_interpolation
        if self._augmentation_cfg.no_aug or not train_interpolation:
            train_interpolation = self._data_cfg.interpolation
        train_loader = create_loader(
            train_dataset,
            input_size=self._data_cfg.input_size,
            batch_size=self._train_cfg.batch_size,
            is_training=True,
            use_prefetcher=self._misc_cfg.prefetcher,
            no_aug=self._augmentation_cfg.no_aug,
            scale=self._augmentation_cfg.scale,
            ratio=self._augmentation_cfg.ratio,
            hflip=self._augmentation_cfg.hflip,
            vflip=self._augmentation_cfg.vflip,
            color_jitter=self._augmentation_cfg.color_jitter,
            auto_augment=self._augmentation_cfg.auto_augment,
            interpolation=train_interpolation,
            mean=self._data_cfg.mean,
            std=self._data_cfg.std,
            num_workers=self._misc_cfg.num_workers,
            distributed=False,
            collate_fn=self._collate_fn,
            pin_memory=self._misc_cfg.pin_mem,
            use_multi_epochs_loader=self._misc_cfg.use_multi_epochs_loader
        )

        val_loader = create_loader(
            val_dataset,
            input_size=self._data_cfg.input_size,
            batch_size=self._data_cfg.validation_batch_size_multiplier * self._train_cfg.batch_size,
            is_training=False,
            use_prefetcher=self._misc_cfg.prefetcher,
            interpolation=self._data_cfg.interpolation,
            mean=self._data_cfg.mean,
            std=self._data_cfg.std,
            num_workers=self._misc_cfg.num_workers,
            distributed=False,
            crop_pct=self._data_cfg.crop_pct,
            pin_memory=self._misc_cfg.pin_mem,
        )

        self._time_elapsed += time.time() - tic
        return self._train_loop(train_loader, val_loader, time_limit=time_limit)

    def _train_loop(self, train_loader, val_loader, time_limit=math.inf):
        start_tic = time.time()
        # setup loss function
        if self.mixup_active:
            # smoothing is handled with mixup target transform
            train_loss_fn = SoftTargetCrossEntropy()
        elif self._augmentation_cfg.smoothing:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=self._augmentation_cfg.smoothing)
        else:
            train_loss_fn = nn.CrossEntropyLoss()
        validate_loss_fn = nn.CrossEntropyLoss()
        train_loss_fn = train_loss_fn.to(self.ctx[0])
        validate_loss_fn = validate_loss_fn.to(self.ctx[0])
        eval_metric = self._misc_cfg.eval_metric
        if self._problem_type == REGRESSION:
            train_loss_fn = nn.MSELoss()
            validate_loss_fn = nn.MSELoss()
            eval_metric = 'rmse'
        early_stopper = EarlyStopperOnPlateau(
            patience=self._train_cfg.early_stop_patience,
            min_delta=self._train_cfg.early_stop_min_delta,
            baseline_value=self._train_cfg.early_stop_baseline,
            max_value=self._train_cfg.early_stop_max_value)

        self._logger.info('Start training from [Epoch %d]', max(self._train_cfg.start_epoch, self.epoch))

        self._time_elapsed += time.time() - start_tic
        for self.epoch in range(max(self.start_epoch, self.epoch), self.epochs):
            epoch = self.epoch
            if self._best_acc >= 1.0:
                self._logger.info('[Epoch {}] Early stopping as acc is reaching 1.0'.format(epoch))
                break
            should_stop, stop_message = early_stopper.get_early_stop_advice()
            if should_stop:
                self._logger.info('[Epoch {}] '.format(epoch) + stop_message)
                break
            train_metrics = self.train_one_epoch(
                epoch, self.net, train_loader, self._optimizer, train_loss_fn,
                lr_scheduler=self._lr_scheduler, output_dir=self._logdir,
                amp_autocast=self._amp_autocast, loss_scaler=self._loss_scaler, model_ema=self._model_ema, mixup_fn=self._mixup_fn, time_limit=time_limit)
            # reaching time limit, exit early
            if train_metrics['time_limit']:
                self._logger.warning(f'`time_limit={time_limit}` reached, exit early...')
                return {'train_acc': train_metrics['train_acc'], 'valid_acc': self._best_acc,
                        'time': self._time_elapsed, 'checkpoint': self._cp_name}
            post_tic = time.time()

            eval_metrics = self.validate(self.net, val_loader, validate_loss_fn, amp_autocast=self._amp_autocast)

            if self._model_ema is not None and not self._model_ema_cfg.model_ema_force_cpu:
                ema_eval_metrics = self.validate(
                    self._model_ema.module, val_loader, validate_loss_fn, amp_autocast=self._amp_autocast)
                eval_metrics = ema_eval_metrics

            if self._problem_type == REGRESSION:
                val_acc = eval_metrics['rmse']
                if self._reporter:
                    self._reporter(epoch=epoch, acc_reward=-val_acc)
                early_stopper.update(-val_acc)

                if -val_acc > self._best_acc:
                    self._cp_name = os.path.join(self._logdir, _BEST_CHECKPOINT_FILE)
                    self._logger.info('[Epoch %d] Current best rmse: %f vs previous %f, saved to %s',
                                      self.epoch, val_acc, -self._best_acc, self._cp_name)
                    self.save(self._cp_name)
                    self._best_acc = -val_acc
            else:
                val_acc = eval_metrics['top1']
                if self._reporter:
                    self._reporter(epoch=epoch, acc_reward=val_acc)
                early_stopper.update(val_acc)

                if val_acc > self._best_acc:
                    self._cp_name = os.path.join(self._logdir, _BEST_CHECKPOINT_FILE)
                    self._logger.info('[Epoch %d] Current best top-1: %f vs previous %f, saved to %s',
                                      self.epoch, val_acc, self._best_acc, self._cp_name)
                    self.save(self._cp_name)
                    self._best_acc = val_acc

            if self._lr_scheduler is not None:
                # step LR for next epoch
                self._lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            self._time_elapsed += time.time() - post_tic

        if 'accuracy' in train_metrics:
            return {'train_acc': train_metrics['accuracy'], 'valid_acc': self._best_acc,
                    'time': self._time_elapsed, 'checkpoint': self._cp_name}
        # rmse
        else:
            if self._problem_type == REGRESSION:
                return {'train_score': train_metrics['rmse'], 'valid_score': -self._best_acc,
                        'time': self._time_elapsed, 'checkpoint': self._cp_name}
            # mixup
            else:
                return {'train_score': train_metrics['rmse'], 'valid_acc': self._best_acc,
                        'time': self._time_elapsed, 'checkpoint': self._cp_name}

    def train_one_epoch(
            self, epoch, net, loader, optimizer, loss_fn,
            lr_scheduler=None, output_dir=None, amp_autocast=suppress,
            loss_scaler=None, model_ema=None, mixup_fn=None, time_limit=math.inf):
        start_tic = time.time()
        if self._augmentation_cfg.mixup_off_epoch and epoch >= self._augmentation_cfg.mixup_off_epoch:
            if self._misc_cfg.prefetcher and loader.mixup_enabled:
                loader.mixup_enabled = False
            elif mixup_fn is not None:
                mixup_fn.mixup_enabled = False

        second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        losses_m = AverageMeter()
        train_metric_score_m = AverageMeter()

        net.train()

        num_updates = epoch * len(loader)
        self._time_elapsed += time.time() - start_tic
        tic = time.time()
        last_tic = time.time()
        train_metric_name = 'accuracy'
        batch_idx = 0
        for batch_idx, (input, target) in enumerate(loader):
            b_tic = time.time()
            if self._time_elapsed > time_limit:
                return {'train_acc': train_metric_score_m.avg, 'train_loss': losses_m.avg, 'time_limit': True}
            if self._problem_type == REGRESSION:
                target = target.to(torch.float32)
            if not self._misc_cfg.prefetcher:
                # prefetcher would move data to cuda by default
                input, target = input.to(self.ctx[0]), target.to(self.ctx[0])
                if mixup_fn is not None:
                    input, target = mixup_fn(input, target)

            with amp_autocast():
                output = net(input)
                if self._problem_type == REGRESSION:
                    output = output.flatten()
                loss = loss_fn(output, target)
            if self._problem_type == REGRESSION:
                train_metric_name = 'rmse'
                train_metric_score = rmse(output, target)
            else:
                if output.shape == target.shape:
                    train_metric_name = 'rmse'
                    train_metric_score = rmse(output, target)
                else:
                    train_metric_score = accuracy(output, target)[0] / 100

            losses_m.update(loss.item(), input.size(0))
            train_metric_score_m.update(train_metric_score.item(), output.size(0))

            optimizer.zero_grad()
            if loss_scaler is not None:
                loss_scaler(
                    loss, optimizer,
                    clip_grad=self._optimizer_cfg.clip_grad, clip_mode=self._optimizer_cfg.clip_mode,
                    parameters=model_parameters(net, exclude_head='agc' in self._optimizer_cfg.clip_mode),
                    create_graph=second_order)
            else:
                loss.backward(create_graph=second_order)
                if self._optimizer_cfg.clip_grad is not None:
                    dispatch_clip_grad(
                        model_parameters(net, exclude_head='agc' in self._optimizer_cfg.clip_mode),
                        value=self._optimizer_cfg.clip_grad, mode=self._optimizer_cfg.clip_mode)
                optimizer.step()

            if model_ema is not None:
                model_ema.update(net)

            if self.found_gpu:
                torch.cuda.synchronize()

            num_updates += 1
            if (batch_idx+1) % self._misc_cfg.log_interval == 0:
                lrl = [param_group['lr'] for param_group in optimizer.param_groups]
                lr = sum(lrl) / len(lrl)
                self._logger.info('Epoch[%d] Batch [%d]\tSpeed: %f samples/sec\t%s=%f\tlr=%f',
                                  epoch, batch_idx,
                                  self._train_cfg.batch_size*self._misc_cfg.log_interval/(time.time()-last_tic),
                                  train_metric_name, train_metric_score_m.avg, lr)
                last_tic = time.time()

                if self._misc_cfg.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True)

            if lr_scheduler is not None:
                lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

            self._time_elapsed += time.time() - b_tic

        throughput = int(self._train_cfg.batch_size * batch_idx / (time.time() - tic))
        self._logger.info('[Epoch %d] training: %s=%f', epoch, train_metric_name, train_metric_score_m.avg)
        self._logger.info('[Epoch %d] speed: %d samples/sec\ttime cost: %f', epoch, throughput, time.time()-tic)

        end_time = time.time()
        if hasattr(optimizer, 'sync_lookahead'):
            optimizer.sync_lookahead()

        self._time_elapsed += time.time() - end_time

        return {train_metric_name: train_metric_score_m.avg, 'train_loss': losses_m.avg, 'time_limit': False}

    def validate(self, net, loader, loss_fn, amp_autocast=suppress, metric_name=None):
        losses_m = AverageMeter()
        top1_m = AverageMeter()
        top5_m = AverageMeter()
        rmse_m = AverageMeter()

        net.eval()

        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(loader):
                if not self._misc_cfg.prefetcher:
                    input = input.to(self.ctx[0])
                    target = target.to(self.ctx[0])

                with amp_autocast():
                    output = net(input)
                    if self._problem_type == REGRESSION:
                        output = output.flatten()
                if isinstance(output, (tuple, list)):
                    output = output[0]

                if self._problem_type == REGRESSION:
                    if metric_name:
                        assert metric_name == 'rmse', f'{metric_name} metric not supported for regression.'
                    val_metric_score = rmse(output, target)
                else:
                    val_metric_score = accuracy(output, target, topk=(1, min(5, self.num_class)))

                # augmentation reduction
                reduce_factor = self._misc_cfg.tta
                if self._problem_type != REGRESSION and reduce_factor > 1:
                    output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                    target = target[0:target.size(0):reduce_factor]

                loss = loss_fn(output, target)
                reduced_loss = loss.data

                if self.found_gpu:
                    torch.cuda.synchronize()

                losses_m.update(reduced_loss.item(), input.size(0))
                if self._problem_type == REGRESSION:
                    rmse_score = val_metric_score
                    rmse_m.update(rmse_score.item(), output.size(0))
                else:
                    acc1, acc5 = val_metric_score
                    acc1 /= 100
                    acc5 /= 100
                    top1_m.update(acc1.item(), output.size(0))
                    top5_m.update(acc5.item(), output.size(0))

        if self._problem_type == REGRESSION:
            self._logger.info('[Epoch %d] validation: rmse=%f', self.epoch, rmse_m.avg)
            return {'loss': losses_m.avg, 'rmse': rmse_m.avg}
        else:
            self._logger.info('[Epoch %d] validation: top1=%f top5=%f', self.epoch, top1_m.avg, top5_m.avg)
            return {'loss': losses_m.avg, 'top1': top1_m.avg, 'top5': top5_m.avg}

    def _init_network(self, **kwargs):
        load_only = kwargs.get('load_only', False)
        if not self.num_class and self._problem_type != REGRESSION:
            raise ValueError('This is a classification problem and we are not able to create network when `num_class` is unknown. \
                It should be inferred from dataset or resumed from saved states.')
        assert len(self.classes) == self.num_class

        # Disable syncBatchNorm as it's only supported on DDP
        if self._train_cfg.sync_bn:
            self._logger.info(
                'Disable Sync batch norm as it is not supported for now.')
            update_cfg(self._cfg, {'train': {'sync_bn': False}})

        # ctx
        self.found_gpu = False
        valid_gpus = []
        if self._cfg.gpus:
            valid_gpus = self._torch_validate_gpus(self._cfg.gpus)
            self.found_gpu = True
            if not valid_gpus:
                self.found_gpu = False
                self._logger.warning(
                    'No gpu detected, fallback to cpu. You can ignore this warning if this is intended.')
            elif len(valid_gpus) != len(self._cfg.gpus):
                self._logger.warning(
                    f'Loaded on gpu({valid_gpus}), different from gpu({self._cfg.gpus}).')
        self.ctx = [torch.device(f'cuda:{gid}') for gid in valid_gpus] if self.found_gpu else [torch.device('cpu')]
        self.valid_gpus = valid_gpus

        if not self.found_gpu and self.use_amp:
            self.use_amp = None
            self._logger.warning('Training on cpu. AMP disabled.')
            update_cfg(self._cfg, {'misc': {'amp': False, 'apex_amp': False, 'native_amp': False}})

        if not self.found_gpu and self._misc_cfg.prefetcher:
            self._logger.warning(
                'Training on cpu. Prefetcher disabled.')
            update_cfg(self._cfg, {'misc': {'prefetcher': False}})
            self._logger.warning(
                'Training on cpu. SyncBatchNorm disabled.')
            update_cfg(self._cfg, {'train': {'sync_bn': False}})

        random_seed(self._misc_cfg.seed)

        if not self.net:
            self.net = create_model(
                self._img_cls_cfg.model,
                pretrained=self._img_cls_cfg.pretrained and not load_only,
                num_classes=max(self.num_class, 1),
                global_pool=self._img_cls_cfg.global_pool_type,
                drop_rate=self._augmentation_cfg.drop,
                drop_path_rate=self._augmentation_cfg.drop_path,
                drop_block_rate=self._augmentation_cfg.drop_block,
                bn_momentum=self._train_cfg.bn_momentum,
                bn_eps=self._train_cfg.bn_eps,
                scriptable=self._misc_cfg.torchscript
            )

            self._logger.info(f'Model {safe_model_name(self._img_cls_cfg.model)} created, param count: \
                                        {sum([m.numel() for m in self.net.parameters()])}')
        else:
            self._logger.info(f'Use user provided model. Neglect model in config.')
            out_features = list(self.net.children())[-1].out_features
            if self._problem_type != REGRESSION:
                assert out_features == self.num_class, f'Custom model out_feature {out_features} != num_class {self.num_class}.'
            else:
                assert out_features == 1, f'Regression problem expects num_out_feature == 1, got {out_features} instead.'

        resolve_data_config(self._cfg, model=self.net)

        self.net = self.net.to(self.ctx[0])

        # setup synchronized BatchNorm
        if self._train_cfg.sync_bn:
            if has_apex and self.use_amp != 'native':
                # Apex SyncBN preferred unless native amp is activated
                self.net = convert_syncbn_model(self.net)
            else:
                self.net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.net)
            self._logger.info(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

        if self._misc_cfg.torchscript:
            assert not self.use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
            assert not self._train_cfg.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
            self.net = torch.jit.script(self.net)

    def _init_trainer(self):
        if self._optimizer is None:
            if self._img_cls_cfg.pretrained and not self._custom_net \
                and (self._train_cfg.transfer_lr_mult != 1 or self._train_cfg.output_lr_mult != 1):
                # adjust feature/last_fc learning rate multiplier in optimizer
                self._logger.debug(f'Reduce network lr multiplier to {self._train_cfg.transfer_lr_mult}, while keep ' +
                                   f'last FC layer lr_mult to {self._train_cfg.output_lr_mult}')
                optim_kwargs = optimizer_kwargs(cfg=self._cfg)
                optim_kwargs['feature_lr_mult'] = self._cfg.train.transfer_lr_mult
                optim_kwargs['fc_lr_mult'] = self._cfg.train.output_lr_mult
                self._optimizer = create_optimizer_v2a(self.net, **optim_kwargs)
            else:
                self._optimizer = create_optimizer_v2(self.net, **optimizer_kwargs(cfg=self._cfg))
        self._init_loss_scaler()
        self._lr_scheduler, self.epochs = create_scheduler(self._cfg, self._optimizer)
        self._lr_scheduler.step(self.start_epoch, self.epoch)

    def _init_loss_scaler(self):
        # setup automatic mixed-precision (AMP) loss scaling and op casting
        self._amp_autocast = suppress  # do nothing
        self._loss_scaler = None
        if self.use_amp == 'apex':
            self.net, self._optimizer = amp.initialize(self.net, self._optimizer, opt_level='O1')
            self._loss_scaler = ApexScaler()
            self._logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
        elif self.use_amp == 'native':
            self._amp_autocast = torch.cuda.amp.autocast
            self._loss_scaler = NativeScaler()
            self._logger.info('Using native Torch AMP. Training in mixed precision.')
        else:
            self._logger.info('AMP not enabled. Training in float32.')

    def _init_model_ema(self):
        # Disable for now
        if self._model_ema_cfg.model_ema:
            self._logger.info('Disable EMA as it is not supported for now.')
            update_cfg(self._cfg, {'model_ema': {'model_ema': False}})
        # setup exponential moving average of model weights, SWA could be used here too
        self._model_ema = None
        if self._model_ema_cfg.model_ema:
            # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
            self._model_ema = ModelEmaV2(
                self.net, decay=self._model_ema_cfg.model_ema_decay, device='cpu' if self._model_ema_cfg.model_ema_force_cpu else None)


    def evaluate(self, val_data, metric_name=None):
        return self._evaluate(val_data, metric_name)

    def _evaluate(self, val_data, metric_name=None):
        if self._problem_type == REGRESSION:
            validate_loss_fn = nn.MSELoss()
        else:
            validate_loss_fn = nn.CrossEntropyLoss()
        validate_loss_fn = validate_loss_fn.to(self.ctx[0])
        val_data = val_data.to_torch()
        val_loader = create_loader(
            val_data,
            input_size=self._data_cfg.input_size,
            batch_size=self._data_cfg.validation_batch_size_multiplier * self._train_cfg.batch_size,
            is_training=False,
            use_prefetcher=self._misc_cfg.prefetcher,
            interpolation=self._data_cfg.interpolation,
            mean=self._data_cfg.mean,
            std=self._data_cfg.std,
            num_workers=self._misc_cfg.num_workers,
            distributed=False,
            crop_pct=self._data_cfg.crop_pct,
            pin_memory=self._misc_cfg.pin_mem,
        )
        return self.validate(self.net, val_loader, validate_loss_fn, amp_autocast=self._amp_autocast, metric_name=metric_name)

    def _predict(self, x, **kwargs):
        with_proba = kwargs.get('with_proba', False)
        if with_proba and self._problem_type not in [MULTICLASS, BINARY]:
            raise AssertionError('with_proba is only supported for classification problems. Please use predict instead.')
        if isinstance(x, str):
            return self._predict((x,), **kwargs).drop(columns=['image'], errors='ignore')
        elif isinstance(x, pd.DataFrame):
            assert 'image' in x.columns, "Expect column `image` for input images"
            df = self._predict(tuple(x['image']), **kwargs)
            return df.reset_index(drop=True)
        elif isinstance(x, (list, tuple)):
            loader = create_loader(
                ImageListDataset(x),
                input_size=self._data_cfg.input_size,
                batch_size=self._train_cfg.batch_size,
                use_prefetcher=self._misc_cfg.prefetcher,
                interpolation=self._data_cfg.interpolation,
                mean=self._data_cfg.mean,
                std=self._data_cfg.std,
                num_workers=self._misc_cfg.num_workers,
                crop_pct=self._data_cfg.crop_pct
            )

            self.net.eval()

            topk = min(5, self.num_class)
            results = []
            idx = 0
            with torch.no_grad():
                for input, _ in loader:
                    input = input.to(self.ctx[0])
                    labels = self.net(input)
                    for l in labels:
                        if self._problem_type in [MULTICLASS, BINARY]:
                            probs = nn.functional.softmax(l, dim=0).cpu().numpy().flatten()
                            if with_proba:
                                results.append({'image_proba': probs.tolist(), 'image': x[idx]})
                            else:
                                topk_inds = l.topk(topk)[1].cpu().numpy().flatten()
                                results.extend([{'class': self.classes[topk_inds[k]],
                                                 'score': probs[topk_inds[k]],
                                                 'id': topk_inds[k],
                                                 'image': x[idx]}
                                                for k in range(topk)])
                        else:
                            results.append({'prediction': l.cpu().numpy().flatten(), 'image': x[idx]})
                        idx += 1
            return pd.DataFrame(results)
        elif not isinstance(x, torch.Tensor):
            raise ValueError('Input is not supported: {}'.format(type(x)))
        with torch.no_grad():
            input = x.to(self.ctx[0])
            label = self.net(input)
            if self._problem_type in [MULTICLASS, BINARY]:
                topk = min(5, self.num_class)
                probs = nn.functional.softmax(label, dim=0).cpu().numpy().flatten()
                topk_inds = label.topk(topk)[1].cpu().numpy().flatten()
                if with_proba:
                    df = pd.DataFrame([{'image_proba': probs.tolist()}])
                else:
                    df = pd.DataFrame([{'class': self.classes[topk_inds[k]],
                                        'score': probs[topk_inds[k]],
                                        'id': topk_inds[k]}
                                       for k in range(topk)])
            else:
                df = pd.DataFrame([{'prediction': label.cpu().numpy().flatten()}])
        return df


    def _predict_feature(self, x, **kwargs):
        if isinstance(x, str):
            return self._predict_feature((x,))
        elif isinstance(x, pd.DataFrame):
            assert 'image' in x.columns, "Expect column `image` for input images"
            df = self._predict_feature(tuple(x['image']))
            df = df.set_index(x.index)
            df['image'] = x['image']
            return df
        elif isinstance(x, (list, tuple)):
            assert isinstance(x[0], str), "expect image paths in list/tuple input"
            loader = create_loader(
                ImageListDataset(x),
                input_size=self._data_cfg.input_size,
                batch_size=self._train_cfg.batch_size,
                use_prefetcher=self._misc_cfg.prefetcher,
                interpolation=self._data_cfg.interpolation,
                mean=self._data_cfg.mean,
                std=self._data_cfg.std,
                num_workers=self._misc_cfg.num_workers,
                crop_pct=self._data_cfg.crop_pct
            )

            self.net.eval()

            results = []
            with torch.no_grad():
                for input, _ in loader:
                    input = input.to(self.ctx[0])
                    try:
                        features = self.net.forward_features(input)
                    except AttributeError:
                        features = self.net.module.forward_features(input)
                    for f in features:
                        f = f.cpu().numpy().flatten()
                        results.append({'image_feature': f})
            df = pd.DataFrame(results)
            df['image'] = x
            return df
        elif not isinstance(x, torch.Tensor):
            raise ValueError('Input is not supported: {}'.format(type(x)))
        with torch.no_grad():
            input = x.to(self.ctx[0])
            feature = self.net.forward_features(input)
            result = [{'image_feature': feature}]
        df = pd.DataFrame(result)
        return df

    def _reconstruct_state_dict(self, state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module') else k
            new_state_dict[name] = v
        return new_state_dict

    # pylint: disable=redefined-outer-name, reimported
    def __getstate__(self):
        d = self.__dict__.copy()
        try:
            import torch
            net = d.pop('net', None)
            model_ema = d.pop('_model_ema', None)
            optimizer = d.pop('_optimizer', None)
            loss_scaler = d.pop('_loss_scaler', None)
            save_state = {}
            if net is not None:
                if not self._custom_net:
                    if isinstance(net, torch.nn.DataParallel):
                        save_state['state_dict'] = get_state_dict(net.module, unwrap_model)
                    else:
                        save_state['state_dict'] = get_state_dict(net, unwrap_model)
                else:
                    net_pickle = pickle.dumps(net)
                    save_state['net_pickle'] = net_pickle
            if optimizer is not None:
                save_state['optimizer'] = optimizer.state_dict()
            if loss_scaler is not None:
                save_state[loss_scaler.state_dict_key] = loss_scaler.state_dict()
            if model_ema is not None:
                save_state['state_dict_ema'] = get_state_dict(model_ema, unwrap_model)
        except ImportError:
            pass
        d['save_state'] = save_state
        d['_logger'] = None
        d['_reporter'] = None
        return d

    def __setstate__(self, state):
        save_state = state.pop('save_state', None)
        self.__dict__.update(state)
        # logger
        self._logger = logging.getLogger(state.get('_name', self.__class__.__name__))
        self._logger.setLevel(logging.ERROR)
        try:
            fh = logging.FileHandler(self._log_file)
            self._logger.addHandler(fh)
        #pylint: disable=bare-except
        except:
            pass
        if not save_state:
            self.net = None
            self._optimizer = None
            self._logger.setLevel(logging.INFO)
            return
        try:
            import torch
            self.net = None
            self._optimizer = None
            if self._custom_net:
                if save_state.get('net_pickle', None):
                    self.net = pickle.loads(save_state['net_pickle'])
            else:
                if save_state.get('state_dict', None):
                    self._init_network(load_only=True)
                    net_state_dict = self._reconstruct_state_dict(save_state['state_dict'])
                    if isinstance(self.net, torch.nn.DataParallel):
                        self.net.module.load_state_dict(net_state_dict)
                    else:
                        self.net.load_state_dict(net_state_dict)
            if save_state.get('optimizer', None):
                self._init_trainer()
                self._optimizer.load_state_dict(save_state['optimizer'])
            if hasattr(self, '_loss_scaler') and self._loss_scaler and self._loss_scaler.state_dict_key in save_state:
                loss_scaler_dict = save_state[self._loss_scaler.state_dict_key]
                self._loss_scaler.load_state_dict(loss_scaler_dict)
            if save_state.get('state_dict_ema', None):
                self._init_model_ema()
                model_ema_dict = save_state.get('state_dict_ema')
                model_ema_dict = self._reconstruct_state_dict(model_ema_dict)
                if isinstance(self.net, torch.nn.DataParallel):
                    self._model_ema.module.module.load_state_dict(model_ema_dict)
                else:
                    self._model_ema.module.load_state_dict(model_ema_dict)
        except ImportError:
            pass
        self._logger.setLevel(logging.INFO)

class ImageListDataset(torch.utils.data.Dataset):
    """An internal image list dataset for batch predict"""
    def __init__(self, imlist):
        self._imlist = imlist
        self.transform = None

    def __getitem__(self, idx):
        img = Image.open(self._imlist[idx]).convert('RGB')
        label = None
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.tensor(-1, dtype=torch.long)

    def __len__(self):
        return len(self._imlist)
