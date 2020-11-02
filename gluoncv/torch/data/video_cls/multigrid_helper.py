"""Multigrid support to speed up training of video models"""
from functools import reduce
import numpy as np

from torch.utils.data import Sampler
from torch._six import int_classes as _int_classes


__all__ = ['multiGridSampler', 'MultiGridBatchSampler']


sq2 = np.sqrt(2)
class multiGridSampler(object):
    """
    A Multigrid Method for Efficiently Training Video Models
    Chao-Yuan Wu, Ross Girshick, Kaiming He, Christoph Feichtenhofer, Philipp Krähenbühl
    CVPR 2020, https://arxiv.org/abs/1912.00998
    """
    def __init__(self):
        self.long_cycle = np.asarray([[1, 1, 1], [2, 1, 1], [2, sq2, sq2]])[::-1]
        self.short_cycle = np.asarray([[1, 1, 1], [1, sq2, sq2], [1, 2, 2]])[::-1]
        self.short_cycle_sp = np.asarray([[1, 1, 1], [1, sq2, sq2], [1, sq2, sq2]])[::-1]
        self.mod_long = len(self.long_cycle)
        self.mod_short = len(self.short_cycle)

    def scale(self, x):
        return int(np.around(reduce(lambda a, b: a * b, x)))

    def get_scale(self, alpha, beta):
        long_scale = self.scale(self.long_cycle[alpha])
        if alpha == 0:
            short_scale = self.scale(self.short_cycle_sp[beta])
        else:
            short_scale = self.scale(self.short_cycle[beta])
        return long_scale * short_scale

    def get_scale_alpha(self, alpha):
        long_scale = self.scale(self.long_cycle[alpha])
        return long_scale

    def get_scale_beta(self, beta):
        short_scale = self.scale(self.short_cycle[beta])
        return short_scale

    def get_resize(self, alpha, beta):
        long_item = self.long_cycle[alpha]
        short_item = self.short_cycle[beta]
        return long_item * short_item


class MultiGridBatchSampler(Sampler):
    """
    A Multigrid Method for Efficiently Training Video Models
    Chao-Yuan Wu, Ross Girshick, Kaiming He, Christoph Feichtenhofer, Philipp Krähenbühl
    CVPR 2020, https://arxiv.org/abs/1912.00998
    """
    def __init__(self, sampler, batch_size, drop_last):
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.MG_sampler = multiGridSampler()
        self.alpha = self.MG_sampler.mod_long - 1
        self.beta = 0
        self.batch_scale = self.MG_sampler.get_scale(self.alpha, self.beta)
        self.label = True

    def deactivate(self):
        self.label = False
        self.alpha = self.MG_sampler.mod_long - 1

    def activate(self):
        self.label = True
        self.alpha = 0

    def __iter__(self):
        batch = []
        if self.label:
            self.beta = 0
        else:
            self.beta = self.MG_sampler.mod_short - 1
        self.batch_scale = self.MG_sampler.get_scale(self.alpha, self.beta)
        for idx in self.sampler:
            batch.append([idx, self.alpha, self.beta])
            if len(batch) == self.batch_size*self.batch_scale:
                yield batch
                batch = []
                if self.label:
                    self.beta = (self.beta + 1)%self.MG_sampler.mod_short
                    self.batch_scale = self.MG_sampler.get_scale(self.alpha, self.beta)

        if len(batch) > 0 and not self.drop_last:
            yield batch

    def step_alpha(self):
        self.alpha = (self.alpha + 1)%self.MG_sampler.mod_long

    def compute_lr_milestone(self, lr_milestone):
        """
        long cycle milestones
        """
        self.len_long = self.MG_sampler.mod_long
        self.n_epoch_long = 0
        for x in range(self.len_long):
            self.n_epoch_long += self.MG_sampler.get_scale_alpha(x)
        lr_long_cycle = []
        for i, _ in enumerate(lr_milestone):
            if i == 0:
                pre = 0
            else:
                pre = lr_milestone[i-1]
            cycle_length = (lr_milestone[i] - pre) // self.n_epoch_long
            bonus = (lr_milestone[i] - pre)%self.n_epoch_long // self.len_long
            for j in range(self.len_long)[::-1]:
                pre = pre + cycle_length*(2**j) + bonus
                if j == 0:
                    pre = lr_milestone[i]
                lr_long_cycle.append(pre)
        lr_long_cycle.append(0)
        lr_long_cycle = sorted(lr_long_cycle)
        return lr_long_cycle

    def __len__(self):
        self.len_short = self.MG_sampler.mod_short
        self.n_epoch_short = 0
        for x in range(self.len_short):
            self.n_epoch_short += self.MG_sampler.get_scale_beta(x)
        short_batch_size = self.batch_size * self.MG_sampler.get_scale_alpha(self.alpha)
        num_short = len(self.sampler) // short_batch_size

        total = num_short // self.n_epoch_short * self.len_short
        remain = self.n_epoch_short
        for x in range(self.len_short):
            remain = remain - (2**x)
            if remain <= 0:
                break
            total = total + int(num_short%self.n_epoch_short >= remain)

        return total
