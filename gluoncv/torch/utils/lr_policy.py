"""Learning rate policy
Gradually warm-up(increasing) learning rate for pytorch's optimizer.
Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
Code adapted from https://github.com/ildoonet/pytorch-gradual-warmup-lr
"""
# pylint: disable=missing-function-docstring, line-too-long, inconsistent-return-statements, redefined-builtin,missing-class-docstring
from typing import List
import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """
    def __init__(self,
                 optimizer,
                 multiplier,
                 total_epoch,
                 after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError(
                'multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [
                base_lr * (float(self.last_epoch) / self.total_epoch)
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr *
                ((self.multiplier - 1.) * self.last_epoch / self.total_epoch +
                 1.) for base_lr in self.base_lrs
            ]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        # pylint: disable=access-member-before-definition
        if epoch is None:
            epoch = self.last_epoch + 1
        # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        self.last_epoch = epoch if epoch != 0 else 1
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [
                base_lr *
                ((self.multiplier - 1.) * self.last_epoch / self.total_epoch +
                 1.) for base_lr in self.base_lrs
            ]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if not isinstance(self.after_scheduler, ReduceLROnPlateau):
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


class ReduceLROnPlateauWarmup(ReduceLROnPlateau):
    """ Reduce LR on plateau """
    def __init__(self, optimizer, warmup_epochs, **kwargs):
        self.warmup_epochs = warmup_epochs
        super().__init__(optimizer, **kwargs)
        self.last_epoch = 0
        self.base_lrs = []
        for group in optimizer.param_groups:
            self.base_lrs.append(group["lr"])
        self.step_rop(self.mode_worse, False, None)

    def step_rop(self, metrics, do_eval, epoch=None):
        assert epoch is None
        epoch = self.last_epoch + 1
        if epoch <= self.warmup_epochs:
            factor = epoch / self.warmup_epochs
            self.last_epoch = epoch
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.base_lrs[i] * factor
        elif not do_eval:
            pass
        else:
            super().step(metrics, epoch=epoch)

class WarmupLinearLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 max_iter: int = 90000,
                 warmup_factor: float = 0.001,
                 warmup_iters: int = 1000,
                 warmup_method: str = "linear",
                 last_epoch: int = -1):
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        self.max_iter = max_iter
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        return [base_lr * warmup_factor * (1 - self.last_epoch / self.max_iter)
                for base_lr in self.base_lrs]

    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()

def _get_warmup_factor_at_iter(
        method: str, iter: int, warmup_iters: int, warmup_factor: float) -> float:
    """
    Return the learning rate warmup factor at a specific iteration.
    See :paper:`ImageNet in 1h` for more details.

    Args:
        method (str): warmup method; either "constant" or "linear".
        iter (int): iteration at which to calculate the warmup factor.
        warmup_iters (int): the number of warmup iterations.
        warmup_factor (float): the base warmup factor (the meaning changes according
            to the method used).

    Returns:
        float: the effective warmup factor at the given iteration.
    """
    if iter >= warmup_iters:
        return 1.0

    if method == "constant":
        return warmup_factor
    elif method == "linear":
        alpha = iter / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    else:
        raise ValueError("Unknown warmup method: {}".format(method))


def build_lr_scheduler(cfg, optimizer):
    """
    Build a LR scheduler from config.
    """
    name = cfg.CONFIG.TRAIN.ITER_LR_SCHEDULER_NAME
    if name == "WarmupMultiStepLR":
        return WarmupMultiStepLR(
            optimizer,
            cfg.CONFIG.TRAIN.ITER_LR_STEPS,
            cfg.CONFIG.TRAIN.STEP,
            warmup_factor=cfg.CONFIG.TRAIN.ITER_BASED_WARMUP_FACTOR,
            warmup_iters=cfg.CONFIG.TRAIN.ITER_BASED_WARMUP_ITERS,
            warmup_method=cfg.CONFIG.TRAIN.ITER_BASED_WARMUP_METHOD,
        )
    elif name == "WarmupCosineLR":
        return WarmupCosineLR(
            optimizer,
            cfg.CONFIG.TRAIN.ITER_NUM,
            warmup_factor=cfg.CONFIG.TRAIN.ITER_BASED_WARMUP_FACTOR,
            warmup_iters=cfg.CONFIG.TRAIN.ITER_BASED_WARMUP_ITERS,
            warmup_method=cfg.CONFIG.TRAIN.ITER_BASED_WARMUP_METHOD,
        )
    elif name == 'WarmupLinearLR':
        return WarmupLinearLR(
            optimizer=optimizer,
            max_iter=cfg.CONFIG.TRAIN.ITER_NUM,
            warmup_factor=cfg.CONFIG.TRAIN.ITER_BASED_WARMUP_FACTOR,
            warmup_iters=cfg.CONFIG.TRAIN.ITER_BASED_WARMUP_ITERS,
            warmup_method=cfg.CONFIG.TRAIN.ITER_BASED_WARMUP_METHOD,
        )
    else:
        raise ValueError("Unknown LR scheduler: {}".format(name))
