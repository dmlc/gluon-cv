import math
import os
import time
from contextlib import suppress
from collections import OrderedDict

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils
from torch.optim.optimizer import Optimizer

from timm.data import create_loader, Mixup, FastCollateMixup, AugMixDataset
from timm.models import create_model, safe_model_name, convert_splitbn_model, load_checkpoint, model_parameters
from timm.utils import random_seed, dispatch_clip_grad, accuracy, ModelEmaV2, AverageMeter
from timm.optim import create_optimizer_v2
from timm.utils import ApexScaler, NativeScaler
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy, JsdCrossEntropy

from .default import TorchImageClassificationCfg
from .utils import *
from ..base_estimator import BaseEstimator, set_default
from ...data.dataset import ImageClassificationDataset
from ....utils.filesystem import try_import
problem_type_constants = try_import(package='autogluon.core.constants',
                                    fromlist=['MULTICLASS', 'BINARY', 'REGRESSION'],
                                    message='Failed to import problem type constants from autogluon.core.')
MULTICLASS = problem_type_constants.MULTICLASS
BINARY = problem_type_constants.BINARY
REGRESSION = problem_type_constants.REGRESSION

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
    def __init__(self, config, logger=None, reporter=None, model=None, optimizer=None, problem_type=None):
        super().__init__(config, logger=logger, reporter=reporter, name=None)
        if problem_type is None:
            problem_type = MULTICLASS
        self._problem_type = problem_type
        self.last_train = None
        self._feature_net = None

        self._model_cfg = self._cfg.model
        self._dataset_cfg = self._cfg.dataset
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

        if model is not None:
            assert isinstance(model, nn), f"given custom network {type(model)}, `torch.nn` expected"
            try:
                self.model.to('cpu')
            except ValueError:
                pass
        self._custom_model = model
        if optimizer is not None:
            if isinstance(optimizer, str):
                pass
            else:
                assert isinstance(optimizer, Optimizer)
        self._optimizer = optimizer

    def _fit(self, train_data, val_data, time_limit=math.inf):
        tic = time.time()
        self._best_acc = -float('inf')
        self.epoch = 0
        self.start_epoch = self._train_cfg.start_epoch
        self._time_elapsed = 0
        if max(self.start_epoch, self.epoch) >= self._train_cfg.epochs:
            return {'time', self._time_elapsed}
        if not isinstance(train_data, pd.DataFrame):
            self.last_train = len(train_data)
        else:
            self.last_train = train_data
        self._init_trainer()
        self._time_elapsed += time.time() - tic
        return self._resume_fit(train_data, val_data, time_limit=time_limit)

    def _resume_fit(self, train_data, val_data, time_limit=math.inf):
        tic = time.time()
        # TODO: regression not implemented
        if self._problem_type != REGRESSION and (not self.classes or not self.num_class):
            raise ValueError('This is a classification problem and we are not able to determine classes of dataset')

        # setup automatic mixed-precision (AMP) loss scaling and op casting
        self._amp_autocast = suppress  # do nothing
        self._loss_scaler = None
        if self.use_amp == 'apex':
            self.model, self._optimizer = amp.initialize(self.model, self._optimizer, opt_level='O1')
            self._loss_scaler = ApexScaler()
            self._logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
        elif self.use_amp == 'native':
            self._amp_autocast = torch.cuda.amp.autocast
            self._loss_scaler = NativeScaler()
            self._logger.info('Using native Torch AMP. Training in mixed precision.')
        else:
            self._logger.info('AMP not enabled. Training in float32.')

        # TODO: move resume_checkpoint and save_checkpoint to baseestimator save & load
        resume_epoch = None
        if self._model_cfg.resume:
            resume_epoch = resume_checkpoint(
                self.model, self._model_cfg.resume,
                _optimizer=None if self._model_cfg.no_resume_opt else self._optimizer,
                loss_scaler=None if self._model_cfg.no_resume_opt else self._loss_scaler,
                logger=self._logger, log_info=True)
        if resume_epoch is not None:
            self.start_epoch = resume_epoch

        if max(self.start_epoch, self.epoch) >= self._train_cfg.epochs:
            return {'time', self._time_elapsed}

        # setup exponential moving average of model weights, SWA could be used here too
        self._model_ema = None
        if self._model_ema_cfg.model_ema:
            # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
            self._model_ema = ModelEmaV2(
                self.model, decay=self._model_ema_cfg.model_ema_decay, device='cpu' if self._model_ema_cfg.model_ema_force_cpu else None)
            if self._model_cfg.resume:
                load_checkpoint(self._model_ema.module, self._cfg.modelresume, use_ema=True)

        # prepare dataset
        train_dataset = train_data.to_torch()
        val_dataset = val_data.to_torch()

        # setup mixup / cutmix
        self._collate_fn = None
        self._mixup_fn = None
        self.mixup_active = self._augmentation_cfg.mixup > 0 or self._augmentation_cfg.cut_mix > 0. or self._augmentation_cfg.cutmix_minmax is not None
        if self.mixup_active:
            mixup_args = dict(
                mixup_alpha=self._augmentation_cfg.mixup, cutmix_alpha=self._augmentation_cfg.cutmix, 
                cutmix_minmax=self._augmentation_cfg.cutmix_minmax, prob=self._augmentation_cfg.mixup_prob, 
                switch_prob=self._augmentation_cfg.mixup_switch_prob, mode=self._augmentation_cfg.mixup_mode,
                label_smoothing=self._augmentation_cfg.smoothing, num_classes=self.num_class)
            if self._misc_cfg.prefetcher:
                assert not self._augmentation_cfg.aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
                self._collate_fn = FastCollateMixup(**mixup_args)
            else:
                self._mixup_fn = Mixup(**mixup_args)

        # wrap dataset in AugMix helper
        if self._augmentation_cfg.aug_splits > 1:
            train_dataset = AugMixDataset(train_dataset, num_splits=self._augmentation_cfg.aug_splits)

        # create data loaders w/ augmentation pipeiine
        train_interpolation = self._augmentation_cfg.train_interpolation
        if self._augmentation_cfg.no_aug or not train_interpolation:
            train_interpolation = self._dataset_cfg.interpolation
        train_loader = create_loader(
            train_dataset,
            input_size=self._dataset_cfg.input_size,
            batch_size=self._train_cfg.batch_size,
            is_training=True,
            use_prefetcher=self._misc_cfg.prefetcher,
            no_aug=self._augmentation_cfg.no_aug,
            re_prob=self._augmentation_cfg.reprob,
            re_mode=self._augmentation_cfg.remode,
            re_count=self._augmentation_cfg.recount,
            re_split=self._augmentation_cfg.resplit,
            scale=self._augmentation_cfg.scale,
            ratio=self._augmentation_cfg.ratio,
            hflip=self._augmentation_cfg.hflip,
            vflip=self._augmentation_cfg.vflip,
            color_jitter=self._augmentation_cfg.color_jitter,
            auto_augment=self._augmentation_cfg.auto_augment,
            num_aug_splits=self._augmentation_cfg.aug_splits,
            interpolation=train_interpolation,
            mean=self._dataset_cfg.mean,
            std=self._dataset_cfg.std,
            num_workers=self._misc_cfg.num_workers,
            distributed=False,
            collate_fn=self._collate_fn,
            pin_memory=self._misc_cfg.pin_mem,
            use_multi_epochs_loader=self._misc_cfg.use_multi_epochs_loader
        )

        val_loader = create_loader(
            val_dataset,
            input_size=self._dataset_cfg.input_size,
            batch_size=self._dataset_cfg.validation_batch_size_multiplier * self._train_cfg.batch_size,
            is_training=False,
            use_prefetcher=self._misc_cfg.prefetcher,
            interpolation=self._dataset_cfg.interpolation,
            mean=self._dataset_cfg.mean,
            std=self._dataset_cfg.std,
            num_workers=self._misc_cfg.num_workers,
            distributed=False,
            crop_pct=self._dataset_cfg.crop_pct,
            pin_memory=self._misc_cfg.pin_mem,
        )

        self._time_elapsed += time.time() - tic
        return self._train_loop(train_loader, val_loader, time_limit=time_limit)

    def _train_loop(self, train_loader, val_loader, time_limit=math.inf):
        start_tic = time.time()
        # setup loss function
        if self._augmentation_cfg.jsd:
            assert self._augmentation_cfg.aug_splits > 1  # JSD only valid with aug splits set
            train_loss_fn = JsdCrossEntropy(num_splits=self._augmentation_cfg.aug_splits, smoothing=self._augmentation_cfg.smoothing)
        elif self.mixup_active:
            # smoothing is handled with mixup target transform
            train_loss_fn = SoftTargetCrossEntropy()
        elif self._augmentation_cfg.smoothing:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=self._augmentation_cfg.smoothing)
        else:
            train_loss_fn = nn.CrossEntropyLoss()
        validate_loss_fn = nn.CrossEntropyLoss()
        train_loss_fn = train_loss_fn.to(self.ctx)
        validate_loss_fn = validate_loss_fn.to(self.ctx)
        # TODO: output matrix
        # setup checkpoint saver and eval metric tracking
        eval_metric = self._misc_cfg.eval_metric
        best_metric = None
        best_epoch = None
        saver = None
        decreasing = True if eval_metric == 'loss' else False
        # TODO: custom saver
        # saver = CheckpointSaver(
        #     model=model, optimizer=self._optimizer, args=args, model_ema=self._model_ema, amp_scaler=self._loss_scaler,
        #     checkpoint_dir=self._logdir, recovery_dir=self._logdir, decreasing=decreasing, max_history=self._misc_cfg.checkpoint_hist)
        # TODO: early stoper
        self._time_elapsed += time.time() - start_tic
        for epoch in range(max(self.start_epoch, self.epoch), self._train_cfg.epochs):
            train_metrics = self.train_one_epoch(
                epoch, self.model, train_loader, self._optimizer, train_loss_fn,
                lr_scheduler=self._lr_scheduler, saver=saver, output_dir=self._logdir,
                amp_autocast=self._amp_autocast, loss_scaler=self._loss_scaler, model_ema=self._model_ema, mixup_fn=self._mixup_fn)
            post_tic = time.time()

            # TODO: evaluation function takes different parameters than mxnet one
            eval_metrics = self.validate(self.model, val_loader, validate_loss_fn, amp_autocast=self._amp_autocast)

            if self._model_ema is not None and not self._model_ema_cfg.model_ema_force_cpu:
                ema_eval_metrics = self.validate(
                    self._model_ema.module, val_loader, validate_loss_fn, amp_autocast=self._amp_autocast, log_suffix=' (EMA)')
                eval_metrics = ema_eval_metrics

            if self._lr_scheduler is not None:
                # step LR for next epoch
                self._lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)

            if best_metric is not None:
                self._logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))

            self._time_elapsed += time.time() - post_tic
        # TODO: return score, time and checkpoint
        return {'train_loss': train_metrics['loss']}

    def train_one_epoch(
        self, epoch, model, loader, optimizer, loss_fn,
        lr_scheduler=None, saver=None, output_dir=None, amp_autocast=suppress,
        loss_scaler=None, model_ema=None, mixup_fn=None):
        start_tic = time.time()
        if self._augmentation_cfg.mixup_off_epoch and epoch >= self._augmentation_cfg.mixup_off_epoch:
            if self._misc_cfg.prefetcher and loader.mixup_enabled:
                loader.mixup_enabled = False
            elif mixup_fn is not None:
                mixup_fn.mixup_enabled = False

        second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        batch_time_m = AverageMeter()
        losses_m = AverageMeter()

        self.model.train()

        last_idx = len(loader) - 1
        num_updates = epoch * len(loader)
        self._time_elapsed += time.time() - start_tic
        for batch_idx, (input, target) in enumerate(loader):
            b_tic = time.time()
            last_batch = batch_idx == last_idx
            if not self._misc_cfg.prefetcher:
                # FIXME: prefetcher only work on cpu?
                input, target = input.to(self.ctx), target.to(self.ctx)
                if mixup_fn is not None:
                    input, target = mixup_fn(input, target)
            
            with amp_autocast():
                output = self.model(input)
                loss = loss_fn(output, target)

            losses_m.update(loss.item(), input.size(0))

            optimizer.zero_grad()
            if loss_scaler is not None:
                loss_scaler(
                    loss, optimizer,
                    clip_grad=self._optimizer_cfg.clip_grad, clip_mode=self._optimizer_cfg.clip_mode,
                    parameters=model_parameters(model, exclude_head='agc' in self._optimizer_cfg.clip_mode),
                    create_graph=second_order)
            else:
                loss.backward(create_graph=second_order)
                if self._optimizer_cfg.clip_grad is not None:
                    dispatch_clip_grad(
                        model_parameters(model, exclude_head='agc' in self._optimizer_cfg.clip_mode),
                        value=self._optimizer_cfg.clip_grad, mode=self._optimizer_cfg.clip_mode)
                optimizer.step()

            if model_ema is not None:
                model_ema.update(model)

            if self.found_gpu:
                torch.cuda.synchronize()

            num_updates += 1
            batch_time_m.update(time.time() - b_tic)
            if last_batch or batch_idx % self._misc_cfg.log_interval == 0:
                lrl = [param_group['lr'] for param_group in optimizer.param_groups]
                lr = sum(lrl) / len(lrl)

                self._logger.info(
                    'Train: {} [{:>4d}/{} ({:>3.0f}%)]  '
                    'Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  '
                    'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                    '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'LR: {lr:.3e}  '.format(
                        epoch,
                        batch_idx, len(loader),
                        100. * batch_idx / last_idx,
                        loss=losses_m,
                        batch_time=batch_time_m,
                        rate=input.size(0) * 1 / batch_time_m.val,
                        rate_avg=input.size(0) * 1 / batch_time_m.avg,
                        lr=lr))

                if self._misc_cfg.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True)

            if saver is not None and self._misc_cfg.recovery_interval and (
                    last_batch or (batch_idx + 1) % self._misc_cfg.recovery_interval == 0):
                saver.save_recovery(epoch, batch_idx=batch_idx)

            if lr_scheduler is not None:
                lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

            self._time_elapsed += time.time() - b_tic
        
        end_time = time.time()
        if hasattr(optimizer, 'sync_lookahead'):
            optimizer.sync_lookahead()
        self._time_elapsed += time.time() - end_time

        return OrderedDict([('loss', losses_m.avg)])

    def validate(self, model, loader, loss_fn, amp_autocast=suppress, log_suffix=''):
        batch_time_m = AverageMeter()
        losses_m = AverageMeter()
        top1_m = AverageMeter()
        top5_m = AverageMeter()

        model.eval()

        last_idx = len(loader) - 1
        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(loader):
                b_tic = time.time()
                last_batch = batch_idx == last_idx
                if not self._misc_cfg.prefetcher:
                    input = input.to(self.ctx)
                    target = target.to(self.ctx)

                with amp_autocast():
                    output = model(input)
                if isinstance(output, (tuple, list)):
                    output = output[0]

                # augmentation reduction
                reduce_factor = self._misc_cfg.tta
                if reduce_factor > 1:
                    output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                    target = target[0:target.size(0):reduce_factor]

                loss = loss_fn(output, target)
                acc1, acc5 = accuracy(output, target, topk=(1, min(5, self.num_class)))

                reduced_loss = loss.data

                if self.found_gpu:
                    torch.cuda.synchronize()

                losses_m.update(reduced_loss.item(), input.size(0))
                top1_m.update(acc1.item(), output.size(0))
                top5_m.update(acc5.item(), output.size(0))

                batch_time_m.update(time.time() - b_tic)
                if last_batch or batch_idx % self._misc_cfg.log_interval == 0:
                    log_name = 'Test' + log_suffix
                    self._logger.info(
                        '{0}: [{1:>4d}/{2}]  '
                        'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                        'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                        'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                        'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                            log_name, batch_idx, last_idx, batch_time=batch_time_m,
                            loss=losses_m, top1=top1_m, top5=top5_m))

        metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

        return metrics

    def _init_network(self, **kwargs):
        if self._problem_type == REGRESSION:
            raise NotImplementedError
        assert len(self.classes) == self.num_class

        # ctx
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
            valid_gpus = ','.join(valid_gpus)
        self.ctx = torch.device(f'cuda:{valid_gpus}' if self.found_gpu else 'cpu')

        self.model = create_model(
            self._model_cfg.model,
            pretrained=self._model_cfg.pretrained,
            num_classes=self.num_class,
            global_pool=self._model_cfg.global_pool_type,
            checkpoint_path=self._model_cfg.initial_checkpoint,
            drop_rate=self._augmentation_cfg.drop,
            drop_path_rate=self._augmentation_cfg.drop_path,
            drop_block_rate=self._augmentation_cfg.drop_block,
            bn_momentum=self._train_cfg.bn_momentum,
            bn_eps=self._train_cfg.bn_eps,
            scriptable=self._misc_cfg.torchscript
        )

        self._logger.info(f'Model {safe_model_name(self._model_cfg.model)} created, param count: \
                                    {sum([m.numel() for m in self.model.parameters()])}')

        resolve_data_config(self._cfg, model=self.model)

        # setup augmentation batch splits for contrastive loss or split bn
        # TODO: disable for now
        if self._augmentation_cfg.aug_splits > 0:
            assert self._augmentation_cfg.aug_splits > 1, 'A split of 1 makes no sense'

        # enable split bn (separate bn stats per batch-portion)
        if self._train_cfg.split_bn:
            assert self._augmentation_cfg.aug_splits > 1 or self._augmentation_cfg.resplit
            self.model = convert_splitbn_model(self.model, max(self._augmentation_cfg.aug_splits, 2))

        # move model to correct ctx
        self.model = self.model.to(self.ctx)

        # setup synchronized BatchNorm
        if self._train_cfg.sync_bn:
            assert not self._train_cfg.split_bn
            if has_apex and self.use_amp != 'native':
                # Apex SyncBN preferred unless native amp is activated
                self.model = convert_syncbn_model(self.model)
            else:
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self._logger.info(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')
        
        if self._misc_cfg.torchscript:
            assert not self.use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
            assert not self._train_cfg.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
            self.model = torch.jit.script(self.model)

    def _init_trainer(self):
        if self._optimizer is None:
            self._optimizer = create_optimizer_v2(self.model, **optimizer_kwargs(cfg=self._cfg))
        self._lr_scheduler, self.num_epochs = create_scheduler(self._cfg, self._optimizer)

    def evaluate(self, val_data):
        return self._evaluate(val_data)
    
    def _evaluate(self, val_data):
        validate_loss_fn = nn.CrossEntropyLoss()
        validate_loss_fn = validate_loss_fn.to(self.ctx)
        return self.validate(self.model, val_data, validate_loss_fn, amp_autocast=self._amp_autocast)

    def _predict(self, x, **kwargs):
        pass

    def _predict_feature(self, x, **kwargs):
        pass
