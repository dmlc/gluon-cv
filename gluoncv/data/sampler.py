# pylint: disable=line-too-long,too-many-lines,missing-docstring
import random

import numpy as np
from mxnet import gluon

__all__ = ['SplitSampler', 'ShuffleSplitSampler', 'SplitSortedBucketSampler']


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
            print('Length ({}) must be a multiple of the number of partitions ({}).'.format(length,
                                                                                            num_parts))
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


class SplitSortedBucketSampler(gluon.data.sampler.Sampler):
    r"""Batches are sampled from sorted buckets of data.
    First, partition data in buckets of size `batch_size * mult`.
    Each bucket contains `batch_size * mult` elements. The samples inside each bucket are sorted
    based on sort_key and then batched.
    Parameters
    ----------
    sort_keys : list-like object
        The keys to sort the samples.
    batch_size : int
        Batch size of the sampler.
    mult : int or float, default 100
        The multiplier to determine the bucket size. Each bucket will have size `mult * batch_size`.
    num_parts: int, default 1
      Number of partitions which the data is split into
    part_index: int, default 0
      The index of the part to read from
    shuffle : bool, default False
        Whether to shuffle the data.
    Examples
    --------
    >>> lengths = [np.random.randint(1, 1000) for _ in range(1000)]
    >>> sampler = gluoncv.data.SplitSortedBucketSampler(lengths, 16, 1000)
    >>> # The sequence lengths within the batch will be sorted
    >>> for i, indices in enumerate(sampler):
    ...     if i == 0:
    ...         print([lengths[ele] for ele in indices])
    [-etc-]
    """

    def __init__(self, sort_keys, batch_size, mult=32, num_parts=1, part_index=0, shuffle=False,
                 seed=233):
        assert len(sort_keys) > 0
        assert batch_size > 0
        assert mult >= 1, 'Bucket size multiplier must be greater or equal to 1'
        self._sort_keys = sort_keys
        length = len(sort_keys)
        self._batch_size = batch_size
        self._mult = mult
        self._shuffle = shuffle
        # Compute the length of each partition
        part_len = int(np.ceil(float(length) / num_parts))
        # Compute the start index for this partition
        self._start = part_len * part_index
        # Compute the end index for this partition
        self._end = self._start + part_len
        if part_index == num_parts - 1:
            # last part
            self._end = length
            self._start = length - part_len
        self._num_parts = num_parts
        self._seed = seed
        self._shuffled_ids = np.random.RandomState(seed=self._seed).permutation(range(length))

    def __iter__(self):
        if self._num_parts > 1:
            self._shuffled_ids = np.random.RandomState(seed=self._seed).permutation(
                self._shuffled_ids)
        if self._shuffle:
            sample_ids = np.random.permutation(self._shuffled_ids[self._start:self._end])
        else:
            sample_ids = list(range(self._start, self._end))
        bucket_size = int(self._mult * self._batch_size)
        for bucket_begin in range(0, len(sample_ids), bucket_size):
            bucket_end = min(bucket_begin + bucket_size, len(sample_ids))
            if bucket_end - bucket_begin < self._batch_size:
                bucket_begin = bucket_end - self._batch_size
            sorted_sample_ids = sorted(sample_ids[bucket_begin:bucket_end],
                                       key=lambda i: self._sort_keys[i],
                                       reverse=random.randint(0, 1))
            batch_begins = list(range(0, len(sorted_sample_ids), self._batch_size))
            if self._shuffle:
                np.random.shuffle(batch_begins)
            for batch_begin in batch_begins:
                batch_end = min(batch_begin + self._batch_size, len(sorted_sample_ids))
                if batch_end - batch_begin < self._batch_size:
                    yield sorted_sample_ids[batch_end - self._batch_size:batch_end]
                else:
                    yield sorted_sample_ids[batch_begin:batch_end]

    def __len__(self):
        length = int(np.ceil(float(self._end - self._start) / self._batch_size))
        assert length >= 0
        return length
