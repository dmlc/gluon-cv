"""Popular Learning Rate Schedulers"""
# pylint: disable=missing-docstring
from __future__ import division

from math import pi, cos
from mxnet import lr_scheduler

class Compose(lr_scheduler.LRScheduler):
    r"""Compose Learning Rate Schedulers

    Parameters
    ----------

    schedulers: list
        list of LRScheduler objects
    """
    def __init__(self, schedulers):
        super(Compose, self).__init__()
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
        self.count += scheduler.niter
        self.update_sep.append(count)
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
            self.learning_rate = self.schedulers[ind].update(num_update)

class LRScheduler(lr_scheduler.LRScheduler):
    r"""Learning Rate Scheduler

    Parameters
    ----------

    mode : str
        Modes for learning rate scheduler.
        Currently it supports 'constant', 'linear', 'poly' and 'cosine'
    baselr : float
        Base learning rate, i.e. the starting learning rate
    targetlr : float
        Target learning rate, i.e. the ending learning rate
    niter : int
        Number of iterations to be scheduled
    offset : int
        Number of iterations before this scheduler
    power : float
        Power parameter of poly scheduler
    """
    def __init__(self, mode, baselr, targetlr, niter=0, offset=0, power=2):
        super(LRScheduler, self).__init__()
        assert(mode in ['constant', 'linear', 'poly', 'cosine'])

        self.mode = mode
        self.baselr = baselr
        self.targetlr = targetlr
        if self.mode == 'constant':
            self.targetlr = self.baselr
        self.niter = niter
        self.offset = offset
        self.power = power

    def __call__(self, num_update):
        self.update(num_update)
        return self.learning_rate

    def update(self, num_update):
        N = self.niter
        T = num_update - self.offset

        if self.mode == 'constant':
            factor = 0
        elif self.mode == 'linear':
            factor = T / N
        elif self.mode == 'poly':
            factor = pow(1 - T / N, self.power)
        elif self.mode == 'cosine':
            factor = (1 + cos(pi * T / N)) / 2
        else:
            raise NotImplementedError

        self.learning_rate = self.targetlr + (self.baselr - self.target_lr) * factor
