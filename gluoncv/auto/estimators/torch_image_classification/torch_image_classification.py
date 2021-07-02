import math
import os
import time
from contextlib import suppress

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from torch.optim.optimizer import Optimizer

from timm.models import create_model, safe_model_name, convert_splitbn_model, load_checkpoint
from timm.utils import random_seed, ModelEmaV2
from timm.optim import create_optimizer_v2
from timm.utils import ApexScaler, NativeScaler

from .default import TorchImageClassificationCfg
from .utils import *
from ..base_estimator import BaseEstimator, set_default
from ....utils.filesystem import try_import
problem_type_constants = try_import(package='autogluon.core.constants',
                                    fromlist=['MULTICLASS', 'BINARY', 'REGRESSION'],
                                    message='Failed to import problem type constants from autogluon.core.')
MULTICLASS = problem_type_constants.MULTICLASS
BINARY = problem_type_constants.BINARY
REGRESSION = problem_type_constants.REGRESSION

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
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
        self.input_size = self._cfg.train.input_size
        self._feature_net = None

        # resolve AMP arguments based on PyTorch / Apex availability
        self.use_amp = None
        if _cfg.misc.amp:
            # `amp` chooses native amp before apex (APEX ver not actively maintained)
            if _cfg.misc.native_amp and has_native_amp:
                self.use_amp = 'native'
            elif _cfg.misc.apex_amp and has_apex:
                self.use_amp = 'apex'
            elif _cfg.misc.apex_amp or _cfg.misc.native_amp:
                self._logger.warning(f'Neither APEX or native Torch AMP is available, using float32. \
                                       Install NVIDA apex or upgrade to PyTorch 1.6')

        if model is not None:
            assert isinstance(model, nn), f"given custom network {type(model)}, `torch.nn` expected"
            try:
                net.to('cpu')
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
        self._time_elapsed = 0
        if max(self._cfg.train.start_epoch, self.epoch) >= self._cfg.train.epochs:
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

        # setup automatic mixed-precision (AMP) loss scaling and op casting
        self.amp_autocast = suppress  # do nothing
        self.loss_scaler = None
        if self.use_amp == 'apex':
            self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O1')
            self.loss_scaler = ApexScaler()
            if self._cfg.misc.local_rank == 0:
                self._logger.info('Using NVIDIA APEX AMP. Training in mixed precision.')
        elif self.use_amp == 'native':
            self.amp_autocast = torch.cuda.amp.autocast
            self.loss_scaler = NativeScaler()
            if self._cfg.misc.local_rank == 0:
                self._logger.info('Using native Torch AMP. Training in mixed precision.')
        else:
            if self._cfg.misc.local_rank == 0:
                self._logger.info('AMP not enabled. Training in float32.')

        resume_epoch = None
        if self._cfg.model.resume:
            resume_epoch = resume_checkpoint(
                self.model, self._cfg.model.resume,
                optimizer=None if self._cfg.model.no_resume_opt else self.optimizer,
                loss_scaler=None if self._cfg.model.no_resume_opt else self.loss_scaler,
                logger=self._logger, log_info=self.local_rank == 0)
        start_epoch = 0
        if self._cfg.train.start_epoch is not None:
            # a specified start_epoch will always override the resume epoch
            start_epoch = self._cfg.train.start_epoch
        elif resume_epoch is not None:
            start_epoch = resume_epoch

        if max(start_epoch, self.epoch) >= self._cfg.train.epochs:
            return {'time', self._time_elapsed}
        if self._problem_type != REGRESSION and (not self.classes or not self.num_class):
            raise ValueError('This is a classification problem and we are not able to determine classes of dataset')

        # setup exponential moving average of model weights, SWA could be used here too
        model_ema = None
        if self._cfg.model_ema.model_ema is not None:
            # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
            model_ema = ModelEmaV2(
                self.model, decay=self._cfg.model_ema.model_ema_decay, device='cpu' if self._cfg.model_ema.model_ema_force_cpu else None)
            if self._cfg.model.resume:
                load_checkpoint(model_ema.module, self._cfg.modelresume, use_ema=True)

        # setup distributed training
        if self.distributed:
            if has_apex and self.use_amp != 'native':
                # Apex DDP preferred unless native amp is activated
                if self.local_rank == 0:
                    self._logger.info("Using NVIDIA APEX DistributedDataParallel.")
                self.model = ApexDDP(self.model, delay_allreduce=True)
            else:
                if self.local_rank == 0:
                    self._logger.info("Using native Torch DistributedDataParallel.")
                self.model = NativeDDP(self.model, device_ids=[self.local_rank])  # can use device str in Torch >= 1.1
            # NOTE: EMA model does not need to be wrapped by DDP

        # TODO: read in dataset and load in dataloader

        # TODO: train

    def _init_network(self, **kwargs):
        if not self.num_class and self._problem_type != REGRESSION:
            raise ValueError('This is a classification problem and we are not able to create network when `num_class` is unknown. \
                It should be inferred from dataset or resumed from saved states.')
        assert len(self.classes) == self.num_class

        # ctx
        valid_gpus = []
        if self._cfg.gpus:
            valid_gpus = self._torch_validate_gpus(self._cfg.gpus)
            found_gpu = True
            if not valid_gpus:
                found_gpu = False
                self._logger.warning(
                    'No gpu detected, fallback to cpu. You can ignore this warning if this is intended.')
            elif len(valid_gpus) != len(self._cfg.gpus):
                self._logger.warning(
                    f'Loaded on gpu({valid_gpus}), different from gpu({self._cfg.gpus}).')
            valid_gpus = ','.join(valid_gpus)
        self.ctx = [torch.device(f'cuda:{i}') for i in valid_gpus]
        self.ctx = torch.device(f'cuda:{valid_gpus}' if found_gpu else 'cpu')

        # FIXME: distributed logic
        if found_gpu:
            self.distributed = False
            if 'WORLD_SIZE' in os.environ:
                self.distributed = int(os.environ['WORLD_SIZE']) > 1
            self.device = 'cuda:0'
            self.world_size = 1
            self.rank = 0
            if self.distributed:
                self.device = f'cuda:{self._cfg.misc.local_rank}'
                torch.cuda.set_device(self.device)
                torch.distributed.init_process_group(backend='nccl', init_method='env://')
                self.world_size = torch.distributed.get_world_size()
                self.rank = torch.distributed.get_rank()
                self._logger.info(f'Training in distributed mode with multiple processes, 1 GPU per process. \
                                    Process {self.rank}, total {self.world_size}')
            else:
                self._logger.info('Training with a single process on 1 GPUs.')
            assert self.rank >= 0

        self.model = create_model(
            self._cfg.model.model,
            pretrained=self._cfg.model.pretrained,
            num_classes=self.num_class,
            global_pool=self._cfg.model.global_pool_type,
            checkpoint_path=self._cfg.model.initial_checkpoint,
            drop_rate=self._cfg.augmentation.drop,
            drop_path_rate=self._cfg.augmentation.drop_path,
            drop_block_rate=self._cfg.augmentation.drop_block,
            bn_momentum=self._cfg.train.bn_momentum,
            bn_eps=self._cfg.train.bn_eps,
            scriptable=self._cfg.misc.torchscript
        )

        if self._cfg.misc.local_rank == 0:
            self._logger.info(f'Model {safe_model_name(self._cfg.model.model)} created, param count: \
                                      {sum([m.numel() for m in self.model.parameters()])}')

        resolve_data_config(self._cfg, model=self.model)

        # setup augmentation batch splits for contrastive loss or split bn
        num_aug_splits = 0
        if self._cfg.augmentation.aug_splits > 0:
            assert self._cfg.augmentation.aug_splits < 1, 'A split of 1 makes no sense'
            num_aug_splits = self._cfg.augmentation.aug_splits

        # enable split bn (separate bn stats per batch-portion)
        if self._cfg.train.split_bn:
            assert num_aug_splits > 1 or self._cfg.augmentation.resplit
            self.model = convert_splitbn_model(self.model, max(num_aug_splits, 2))

        # move model to GPU, enable channels last layout if set
        if found_gpu:
            self.model.cuda()
        if self._cfg.misc.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

        # setup synchronized BatchNorm for distributed training
        if self.distributed and self._cfg.train.sync_bn:
            assert not self._cfg.train.split_bn
            if has_apex and self.use_amp != 'native':
                # Apex SyncBN preferred unless native amp is activated
                self.model = convert_syncbn_model(self.model)
            else:
                self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            if self._cfg.misc.local_rank == 0:
                self._logger.info(
                    'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                    'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')
        
        if self._cfg.misc.torchscript:
            assert not self.use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
            assert not self._cfg.train.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
            self.model = torch.jit.script(self.model)

    def _init_trainer(self):
        if self.optimizer is None:
            self.optimizer = create_optimizer_v2(self.model, **optimizer_kwargs(cfg=self._cfg))

        # FIXME: lr_scheduler is timm's trainer counterpart
        self.lr_scheduler, self.num_epochs = create_scheduler(self._cfg, self.optimizer)
    
    def _evaluate(self, val_data):
        pass

    def _predict(self, x, **kwargs):
        pass

    def _predict_feature(self, x, **kwargs):
        pass
