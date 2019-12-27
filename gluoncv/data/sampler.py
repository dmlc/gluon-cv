# pylint: disable=line-too-long,too-many-lines,missing-docstring
import random
from mxnet import gluon
import numpy as np

__all__ = ['SplitSampler', 'ShuffleSplitSampler']

class SplitSampler(gluon.data.sampler.Sampler):
    """ Split the dataset into `num_parts` parts and sample from the part with index `part_index`

    Parameters
    ----------
    length: int
      Number of examples in the dataset
    num_parts: int
      Partition the data into multiple parts
    part_index: int
      The index of the part to read from
    """
    def __init__(self, length, num_parts=1, part_index=0):
        # Compute the length of each partition
        self.part_len = length // num_parts
        # Compute the start index for this partition
        self.start = self.part_len * part_index
        # Compute the end index for this partition
        self.end = self.start + self.part_len

    def __iter__(self):
        # Extract examples between `start` and `end`, shuffle and return them.
        indices = list(range(self.start, self.end))
        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.part_len

class ShuffleSplitSampler(gluon.data.sampler.Sampler):
    """Split the dataset into `num_parts` parts and randomly sample from the part
    with index `part_index`.
    The data is randomly shuffled at each iteration within each partition.

    Parameters
    ----------
    length: int
      Number of examples in the dataset
    num_parts: int
      Number of partitions which the data is split into
    part_index: int
      The index of the part to read from
    """
    def __init__(self, length, num_parts=1, part_index=0, seed=0):
        if length % num_parts != 0:
            print('Length ({}) must be a multiple of the number of partitions ({}).'.format(length, num_parts))
        self._seed = seed
        self._state = np.random.RandomState(seed)
        self._indices = list(range(length))
        # Compute the length of each partition
        part_len = length // num_parts
        # Compute the start index for this partition
        self._start = part_len * part_index
        # Compute the end index for this partition
        self._end = self._start + part_len
        if part_index == num_parts - 1:
            self._end = length

    def __iter__(self):
        self._state.shuffle(self._indices)
        # Extract examples between `start` and `end`, shuffle and return them.
        indices = list(self._indices[self._start:self._end])
        return iter(indices)

    def __len__(self):
        return self._end - self._start
