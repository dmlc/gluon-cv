"""Multigrid support to speed up training of video models"""
from functools import reduce
import numpy as np

from torch.utils.data import Sampler
from torch._six import int_classes as _int_classes


__all__ = ['multiGridHelper', 'MultiGridBatchSampler']


sq2 = np.sqrt(2)
class multiGridHelper(object):
    """
    A Multigrid Method for Efficiently Training Video Models
    Chao-Yuan Wu, Ross Girshick, Kaiming He, Christoph Feichtenhofer, Philipp Krähenbühl
    CVPR 2020, https://arxiv.org/abs/1912.00998
    """
    def __init__(self):
        # Scale: [T, H, W]
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
    def __init__(self, sampler, batch_size, drop_last, use_long=False, use_short=False):
        '''
        :param sampler: torch.utils.data.Sample
        :param batch_size: int
        :param drop_last: bool
        :param use_long: bool
        :param use_short: bool
        Apply batch collecting function based on multiGridHelper definition
        '''
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
        if not isinstance(use_long, bool):
            raise ValueError("use_long should be a boolean value, but got "
                             "use_long={}".format(use_long))
        if not isinstance(use_short, bool):
            raise ValueError("use_short should be a boolean value, but got "
                             "use_short={}".format(use_short))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.mg_helper = multiGridHelper()
        # single grid setting
        self.alpha = self.mg_helper.mod_long - 1
        self.beta = self.mg_helper.mod_short - 1
        self.short_cycle_label = False
        if use_long:
            self.activate_long_cycle()
        if use_short:
            self.activate_short_cycle()
        self.batch_scale = self.mg_helper.get_scale(self.alpha, self.beta)

    def activate_short_cycle(self):
        self.short_cycle_label = True
        self.beta = 0

    def activate_long_cycle(self):
        self.alpha = 0

    def deactivate(self):
        self.alpha = self.mg_helper.mod_long - 1
        self.beta = self.mg_helper.mod_short - 1
        self.short_cycle_label = False

    def __iter__(self):
        batch = []
        if self.short_cycle_label:
            self.beta = 0
        else:
            self.beta = self.mg_helper.mod_short - 1
        self.batch_scale = self.mg_helper.get_scale(self.alpha, self.beta)
        for idx in self.sampler:
            batch.append([idx, self.alpha, self.beta])
            if len(batch) == self.batch_size*self.batch_scale:
                yield batch
                batch = []
                if self.short_cycle_label:
                    self.beta = (self.beta + 1) % self.mg_helper.mod_short
                    self.batch_scale = self.mg_helper.get_scale(self.alpha, self.beta)

        if len(batch) > 0 and not self.drop_last:
            yield batch

    def step_long_cycle(self):
        self.alpha = (self.alpha + 1) % self.mg_helper.mod_long

    # def compute_lr_milestone(self, lr_milestone):
    #     """
    #     long cycle milestones, deprecated. Define long cycle in config files
    #     """
    #     self.len_long = self.mg_helper.mod_long
    #     self.n_epoch_long = 0
    #     for x in range(self.len_long):
    #         self.n_epoch_long += self.mg_helper.get_scale_alpha(x)
    #     lr_long_cycle = []
    #     for i, _ in enumerate(lr_milestone):
    #         if i == 0:
    #             pre = 0
    #         else:
    #             pre = lr_milestone[i-1]
    #         cycle_length = (lr_milestone[i] - pre) // self.n_epoch_long
    #         bonus = (lr_milestone[i] - pre)%self.n_epoch_long // self.len_long
    #         for j in range(self.len_long)[::-1]:
    #             pre = pre + cycle_length*(2**j) + bonus
    #             if j == 0:
    #                 pre = lr_milestone[i]
    #             lr_long_cycle.append(pre)
    #     lr_long_cycle.append(0)
    #     lr_long_cycle = sorted(lr_long_cycle)
    #     return lr_long_cycle

    def __len__(self):
        scale_per_short_cycle = 0
        for x in range(self.mg_helper.mod_short):
            scale_per_short_cycle += self.mg_helper.get_scale(self.alpha, x)
        num_full_short_cycle = len(self.sampler) // (self.batch_size * scale_per_short_cycle)

        total = num_full_short_cycle * self.mg_helper.mod_short
        remain = len(self.sampler) % (self.batch_size * scale_per_short_cycle)
        for x in range(self.mg_helper.mod_short):
            remain = remain - self.mg_helper.get_scale(self.alpha, x)*self.batch_size
            if remain >= 0 or (remain < 0 and self.drop_last is False):
                total += 1
            if remain <= 0:
                break
        assert remain <= 0
        return total
