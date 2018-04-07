"""Poly like Learning Rate Scheduler"""
from mxnet import lr_scheduler

class PolyLRScheduler(lr_scheduler.LRScheduler):
    r"""Poly like Learning Rate Scheduler
    It returns a new learning rate by::

        lr = baselr * (1 - iter/maxiter) ^ power

    Parameters
    ----------
    baselr : float
        Base learning rate.
    niters : int
        Number of iterations in each epoch.
    nepochs : int
        Number of training epochs.
    power : float
        Power of poly function.
    """
    def __init__(self, baselr, niters, nepochs, power=0.9):
        super(PolyLRScheduler, self).__init__()
        self.baselr = baselr
        self.learning_rate = self.baselr
        self.niters = niters
        self.N = nepochs * niters
        self.power = power

    def __call__(self, num_update):
        return self.learning_rate

    def update(self, i, epoch):
        T = epoch * self.niters + i
        assert(T >= 0 and T <= self.N)
        self.learning_rate = self.baselr * pow((1 - 1.0 * T / self.N), self.power)
