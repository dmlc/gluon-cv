"""DataLoader utils."""
import io
import pickle
import multiprocessing
from multiprocessing.reduction import ForkingPickler
import numpy as np
from mxnet import nd
from mxnet import context
from mxnet.gluon.data.dataloader import DataLoader, _MultiWorkerIter
from mxnet.gluon.data.dataloader import default_mp_batchify_fn, default_batchify_fn

def default_pad_batchify_fn(data):
    """Collate data into batch, labels are padded to same shape"""
    if isinstance(data[0], nd.NDArray):
        return nd.stack(*data)
    elif isinstance(data[0], tuple):
        data = zip(*data)
        return [default_pad_batchify_fn(i) for i in data]
    else:
        data = np.asarray(data)
        pad = max([l.shape[0] for l in data] + [1,])
        buf = np.full((len(data), pad, data[0].shape[-1]), -1, dtype=data[0].dtype)
        for i, l in enumerate(data):
            buf[i][:l.shape[0], :] = l
        return nd.array(buf, dtype=data[0].dtype)

def default_mp_pad_batchify_fn(data):
    """Use shared memory for collating data into batch, labels are padded to same shape"""
    if isinstance(data[0], nd.NDArray):
        out = nd.empty((len(data),) + data[0].shape, dtype=data[0].dtype,
                       ctx=context.Context('cpu_shared', 0))
        return nd.stack(*data, out=out)
    elif isinstance(data[0], tuple):
        data = zip(*data)
        return [default_mp_pad_batchify_fn(i) for i in data]
    else:
        data = np.asarray(data)
        batch_size = len(data)
        pad = max([l.shape[0] for l in data] + [1,])
        buf = np.full((batch_size, pad, data[0].shape[-1]), -1, dtype=data[0].dtype)
        for i, l in enumerate(data):
            buf[i][:l.shape[0], :] = l
        return nd.array(buf, dtype=data[0].dtype, ctx=context.Context('cpu_shared', 0))


class DetectionDataLoader(DataLoader):
    """Data loader for detection dataset.

    .. deprecated:: 0.2.0
        :py:class:`DetectionDataLoader` is deprecated,
        please use :py:class:`mxnet.gluon.data.DataLoader` with
        batchify functions listed in `gluoncv.data.batchify` directly.

    It loads data batches from a dataset and then apply data
    transformations. It's a subclass of :py:class:`mxnet.gluon.data.DataLoader`,
    and therefore has very simliar APIs.

    The main purpose of the DataLoader is to pad variable length of labels from
    each image, because they have different amount of objects.

    Parameters
    ----------
    dataset : mxnet.gluon.data.Dataset or numpy.ndarray or mxnet.ndarray.NDArray
        The source dataset.
    batch_size : int
        The size of mini-batch.
    shuffle : bool, default False
        If or not randomly shuffle the samples. Often use True for training
        dataset and False for validation/test datasets
    sampler : mxnet.gluon.data.Sampler, default None
        The sampler to use. We should either specify a sampler or enable
        shuffle, not both, because random shuffling is a sampling method.
    last_batch : {'keep', 'discard', 'rollover'}, default is keep
        How to handle the last batch if the batch size does not evenly divide by
        the number of examples in the dataset. There are three options to deal
        with the last batch if its size is smaller than the specified batch
        size.

        - keep: keep it
        - discard: throw it away
        - rollover: insert the examples to the beginning of the next batch
    batch_sampler : mxnet.gluon.data.BatchSampler
        A sampler that returns mini-batches. Do not specify batch_size,
        shuffle, sampler, and last_batch if batch_sampler is specified.
    batchify_fn : callable
        Callback function to allow users to specify how to merge samples
        into a batch.
        Defaults to :py:meth:`gluoncv.data.dataloader.default_pad_batchify_fn`::
            def default_pad_batchify_fn(data):
                if isinstance(data[0], nd.NDArray):
                    return nd.stack(*data)
                elif isinstance(data[0], tuple):
                    data = zip(*data)
                    return [pad_batchify(i) for i in data]
                else:
                    data = np.asarray(data)
                    pad = max([l.shape[0] for l in data])
                    buf = np.full((len(data), pad, data[0].shape[-1]),
                                  -1, dtype=data[0].dtype)
                    for i, l in enumerate(data):
                        buf[i][:l.shape[0], :] = l
                    return nd.array(buf, dtype=data[0].dtype)
    num_workers : int, default 0
        The number of multiprocessing workers to use for data preprocessing.
        If ``num_workers`` = 0, multiprocessing is disabled.
        Otherwise ``num_workers`` multiprocessing worker is used to process data.

    """
    def __init__(self, dataset, batch_size=None, shuffle=False, sampler=None,
                 last_batch=None, batch_sampler=None, batchify_fn=None,
                 num_workers=0):
        import warnings
        warnings.warn('DetectionDataLoader is deprecated. ' +
                      'Please use mxnet.gluon.data.DataLoader '
                      'with batchify functions directly.')
        if batchify_fn is None:
            if num_workers > 0:
                batchify_fn = default_mp_pad_batchify_fn
            else:
                batchify_fn = default_pad_batchify_fn
        super(DetectionDataLoader, self).__init__(
            dataset, batch_size, shuffle, sampler, last_batch,
            batch_sampler, batchify_fn, num_workers)

_worker_dataset = None
def _worker_initializer(dataset):
    """Initialier for processing pool."""
    # global dataset is per-process based and only available in worker processes
    # this is only necessary to handle MXIndexedRecordIO because otherwise dataset
    # can be passed as argument
    global _worker_dataset
    _worker_dataset = dataset

def _worker_fn(samples, transform_fn, batchify_fn):
    """Function for processing data in worker process."""
    # it is required that each worker process has to fork a new MXIndexedRecordIO handle
    # preserving dataset as global variable can save tons of overhead and is safe in new process
    global _worker_dataset
    t_dataset = _worker_dataset.transform(transform_fn)
    batch = batchify_fn([t_dataset[i] for i in samples])
    buf = io.BytesIO()
    ForkingPickler(buf, pickle.HIGHEST_PROTOCOL).dump(batch)
    return buf.getvalue()

class _RandomTransformMultiWorkerIter(_MultiWorkerIter):
    """Internal multi-worker iterator for DataLoader."""
    def __init__(self, transform_fns, interval, worker_pool, batchify_fn, batch_sampler,
                 pin_memory=False, worker_fn=_worker_fn, prefetch=0):
        super(_RandomTransformMultiWorkerIter, self).__init__(
            worker_pool, batchify_fn, batch_sampler, pin_memory, worker_fn, prefetch=0)
        self._transform_fns = transform_fns
        self._current_fn = np.random.choice(self._transform_fns)
        self._interval = max(int(interval), 1)
        # pre-fetch, super class was inited without prefetch
        for _ in range(prefetch):
            self._push_next()

    def _push_next(self):
        """Assign next batch workload to workers."""
        r = next(self._iter, None)
        if r is None:
            return
        if self._sent_idx % self._interval == 0:
            self._current_fn = np.random.choice(self._transform_fns)
        async_ret = self._worker_pool.apply_async(
            self._worker_fn, (r, self._current_fn, self._batchify_fn))
        self._data_buffer[self._sent_idx] = async_ret
        self._sent_idx += 1


class RandomTransformDataLoader(DataLoader):
    """DataLoader that support random transform function applied to dataset.

    Parameters
    ----------
    transform_fns : iterable of callables
        Transform functions that takes a sample as input and returns the transformed sample.
        They will be randomly selected during the dataloader iteration.
    dataset : mxnet.gluon.data.Dataset or numpy.ndarray or mxnet.ndarray.NDArray
        The source dataset. Original dataset is recommanded here since we will apply transform
        function from candidates again during the iteration.
    interval : int, default is 1
        For every `interval` batches, transform function is randomly selected from candidates.
    batch_size : int
        The size of mini-batch.
    shuffle : bool, default False
        If or not randomly shuffle the samples. Often use True for training
        dataset and False for validation/test datasets
    sampler : mxnet.gluon.data.Sampler, default None
        The sampler to use. We should either specify a sampler or enable
        shuffle, not both, because random shuffling is a sampling method.
    last_batch : {'keep', 'discard', 'rollover'}, default is keep
        How to handle the last batch if the batch size does not evenly divide by
        the number of examples in the dataset. There are three options to deal
        with the last batch if its size is smaller than the specified batch
        size.

        - keep: keep it
        - discard: throw it away
        - rollover: insert the examples to the beginning of the next batch
    batch_sampler : mxnet.gluon.data.BatchSampler
        A sampler that returns mini-batches. Do not specify batch_size,
        shuffle, sampler, and last_batch if batch_sampler is specified.
    batchify_fn : callable
        Callback function to allow users to specify how to merge samples
        into a batch.
        Defaults to :py:meth:`gluoncv.data.dataloader.default_pad_batchify_fn`::
            def default_pad_batchify_fn(data):
                if isinstance(data[0], nd.NDArray):
                    return nd.stack(*data)
                elif isinstance(data[0], tuple):
                    data = zip(*data)
                    return [pad_batchify(i) for i in data]
                else:
                    data = np.asarray(data)
                    pad = max([l.shape[0] for l in data])
                    buf = np.full((len(data), pad, data[0].shape[-1]),
                                  -1, dtype=data[0].dtype)
                    for i, l in enumerate(data):
                        buf[i][:l.shape[0], :] = l
                    return nd.array(buf, dtype=data[0].dtype)
    num_workers : int, default 0
        The number of multiprocessing workers to use for data preprocessing.
        If ``num_workers`` = 0, multiprocessing is disabled.
        Otherwise ``num_workers`` multiprocessing worker is used to process data.
    prefetch : int, default is `num_workers * 2`
        The number of prefetching batches only works if `num_workers` > 0.
        If `prefetch` > 0, it allow worker process to prefetch certain batches before
        acquiring data from iterators.
        Note that using large prefetching batch will provide smoother bootstrapping performance,
        but will consume more shared_memory. Using smaller number may forfeit the purpose of using
        multiple worker processes, try reduce `num_workers` in this case.
        By default it defaults to `num_workers * 2`.

    """
    def __init__(self, transform_fns, dataset, interval=1, batch_size=None, shuffle=False,
                 sampler=None, last_batch=None, batch_sampler=None, batchify_fn=None,
                 num_workers=0, pin_memory=False, prefetch=None):
        super(RandomTransformDataLoader, self).__init__(
            dataset, batch_size, shuffle, sampler,
            last_batch, batch_sampler, batchify_fn, 0, pin_memory)
        self._transform_fns = transform_fns
        assert len(self._transform_fns) > 0
        self._interval = max(int(interval), 1)
        # override
        self._num_workers = num_workers if num_workers >= 0 else 0
        self._worker_pool = None
        self._prefetch = max(0, int(prefetch) if prefetch is not None else 2 * self._num_workers)
        if self._num_workers > 0:
            self._worker_pool = multiprocessing.Pool(
                self._num_workers, initializer=_worker_initializer, initargs=[self._dataset])
        if batchify_fn is None:
            if num_workers > 0:
                self._batchify_fn = default_mp_batchify_fn
            else:
                self._batchify_fn = default_batchify_fn
        else:
            self._batchify_fn = batchify_fn

    def __iter__(self):
        if self._num_workers == 0:
            def same_process_iter():
                t = np.random.choice(self._transform_fns)
                for ib, batch in enumerate(self._batch_sampler):
                    if ib % self._interval == 0:
                        t = np.random.choice(self._transform_fns)
                    yield self._batchify_fn([self._dataset.transform(t)[idx] for idx in batch])
            return same_process_iter()
        else:
            return _RandomTransformMultiWorkerIter(
                self._transform_fns, self._interval, self._worker_pool, self._batchify_fn,
                self._batch_sampler, pin_memory=self._pin_memory, worker_fn=_worker_fn,
                prefetch=self._prefetch)

    def __del__(self):
        if self._worker_pool:
            # manually terminate due to a bug that pool is not automatically terminated
            assert isinstance(self._worker_pool, multiprocessing.pool.Pool)
            self._worker_pool.terminate()
