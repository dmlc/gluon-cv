"""Popular Learning Rate Schedulers"""
# pylint: disable=missing-docstring
from __future__ import division

from math import pi, cos
from mxnet import lr_scheduler

class LRCompose(lr_scheduler.LRScheduler):
    r"""Compose Learning Rate Schedulers

    Parameters
    ----------

    schedulers: list
        list of LRScheduler objects
    """
    def __init__(self, schedulers):
        super(LRCompose, self).__init__()
        assert(len(schedulers) > 0)

        self.update_sep = [0]
        self.count = 0
        self.learning_rate = 0
        self.schedulers = []
        for lr in schedulers:
            self.add(lr)

    def add(self, scheduler):
        assert(isinstance(scheduler, LRScheduler))

        scheduler.offset = self.count
        self.count += scheduler.niters
        self.update_sep.append(self.count)
        self.schedulers.append(scheduler)

    def __call__(self, num_update):
        self.update(num_update)
        return self.learning_rate

    def update(self, num_update):
        if num_update < self.count:
            ind = len(self.schedulers) - 1
            for i, sep in enumerate(self.update_sep):
                if sep > num_update:
                    ind = i - 1
                    break
            lr = self.schedulers[ind]
            lr.update(num_update)
            self.learning_rate = lr.learning_rate

class LRScheduler(lr_scheduler.LRScheduler):
    r"""Learning Rate Scheduler

    Parameters
    ----------

    mode : str
        Modes for learning rate scheduler.
        Currently it supports 'constant', 'step', 'linear', 'poly' and 'cosine'.
    base_lr : float
        Base learning rate, i.e. the starting learning rate.
    target_lr : float
        Target learning rate, i.e. the ending learning rate.
        With constant mode target_lr is ignored.
    niters : int
        Number of iterations to be scheduled.
    offset : int
        Number of iterations before this scheduler.
    power : float
        Power parameter of poly scheduler.
    step : list
        A list of iterations to decay the learning rate.
    step_factor : float
        Learning rate decay factor.
    """
    def __init__(self, mode, base_lr=0.1, target_lr=0, niters=0, offset=0, power=2,
                 step=None, step_factor=0.1, baselr=None, targetlr=None):
        super(LRScheduler, self).__init__()
        assert(mode in ['constant', 'step', 'linear', 'poly', 'cosine'])

        self.mode = mode
        if baselr is not None:
            warnings.warn("baselr is deprecated. Please use base_lr.")
            if base_lr == 0.1:
                base_lr = baselr
        self.base_lr = base_lr
        if targetlr is not None:
            warnings.warn("targetlr is deprecated. Please use target_lr.")
            if target_lr == 0:
                target_lr = targetlr
        self.target_lr = target_lr
        if self.mode == 'constant':
            self.target_lr = self.base_lr
        self.niters = niters
        self.offset = offset
        self.power = power
        self.step = step
        self.step_factor = step_factor

    def __call__(self, num_update):
        self.update(num_update)
        return self.learning_rate

    def update(self, num_update):
        N = self.niters - 1
        T = num_update - self.offset
        T = min(max(0, T), N)

        if self.mode == 'constant':
            factor = 0
        elif self.mode == 'linear':
            factor = 1 - T / N
        elif self.mode == 'poly':
            factor = pow(1 - T / N, self.power)
        elif self.mode == 'cosine':
            factor = (1 + cos(pi * T / N)) / 2
        elif self.mode == 'step':
            count = sum([1 for s in self.step if s <= T])
            factor = pow(self.step_factor, count)
        else:
            raise NotImplementedError

        if self.mode == 'step':
            self.learning_rate = self.base_lr * factor
        else:
            self.learning_rate = self.target_lr + (self.base_lr - self.target_lr) * factor
