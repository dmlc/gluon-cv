"""DataLoader utils."""
import sys
import threading
import multiprocessing
import numpy as np
from mxnet import nd
from mxnet import context
from mxnet.gluon.data import DataLoader

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

def _as_in_context(data, ctx):
    """Move data into new context."""
    if isinstance(data, nd.NDArray):
        return data.as_in_context(ctx)
    elif isinstance(data, (list, tuple)):
        return [_as_in_context(d, ctx) for d in data]
    return data

def random_worker_loop(datasets, key_queue, data_queue, batchify_fn):
    """Worker loop for multiprocessing DataLoader with multiple transform functions."""
    for dataset in datasets:
        dataset._fork()
    while True:
        idx, samples, random_idx = key_queue.get()
        if idx is None:
            break
        batch = batchify_fn([datasets[random_idx][i] for i in samples])
        data_queue.put((idx, batch))

def fetcher_loop(data_queue, data_buffer, pin_memory=False):
    """Fetcher loop for fetching data from queue and put in reorder dict."""
    while True:
        idx, batch = data_queue.get()
        if idx is None:
            break
        if pin_memory:
            batch = _as_in_context(batch, context.cpu_pinned())
        else:
            batch = _as_in_context(batch, context.cpu())
        data_buffer[idx] = batch


class _RandomTransformMultiWorkerIter(object):
    """Interal multi-worker iterator for DataLoader with random transform functions."""
    def __init__(self, transform_fns, interval, num_workers, dataset, batchify_fn, batch_sampler,
                 pin_memory=False):
        assert num_workers > 0, "_MultiWorkerIter is not for {} workers".format(num_workers)
        assert isinstance(transform_fns, (list, tuple)) and len(transform_fns) > 1
        from mxnet.gluon.data.dataloader import Queue, SimpleQueue
        self._transform_fns = transform_fns
        self._fn_idx = np.random.randint(len(self._transform_fns))
        self._interval = max(int(interval), 1)
        self._num_workers = num_workers
        self._datasets = [dataset.transform(trans_fn) for trans_fn in self._transform_fns]
        self._batchify_fn = batchify_fn
        self._batch_sampler = batch_sampler
        self._key_queue = Queue()
        self._data_queue = Queue() if sys.version_info[0] <= 2 else SimpleQueue()
        self._data_buffer = {}
        self._rcvd_idx = 0
        self._sent_idx = 0
        self._iter = iter(self._batch_sampler)
        self._shutdown = False

        workers = []
        for _ in range(self._num_workers):
            worker = multiprocessing.Process(
                target=random_worker_loop,
                args=(self._datasets, self._key_queue, self._data_queue, self._batchify_fn))
            worker.daemon = True
            worker.start()
            workers.append(worker)

        self._fetcher = threading.Thread(
            target=fetcher_loop,
            args=(self._data_queue, self._data_buffer, pin_memory))
        self._fetcher.daemon = True
        self._fetcher.start()

        # pre-fetch
        for _ in range(2 * self._num_workers):
            self._push_next()

    def __len__(self):
        return len(self._batch_sampler)

    def __del__(self):
        self.shutdown()

    def _push_next(self):
        """Assign next batch workload to workers."""
        r = next(self._iter, None)
        if r is None:
            return
        if (self._sent_idx + 1) % self._interval == 0:
            self._fn_idx = np.random.randint(len(self._transform_fns))
        self._key_queue.put((self._sent_idx, r, self._fn_idx))
        self._sent_idx += 1

    def __next__(self):
        assert not self._shutdown, "call __next__ after shutdown is forbidden"
        if self._rcvd_idx == self._sent_idx:
            assert not self._data_buffer, "Data buffer should be empty at this moment"
            self.shutdown()
            raise StopIteration

        while True:
            if self._rcvd_idx in self._data_buffer:
                batch = self._data_buffer.pop(self._rcvd_idx)
                self._rcvd_idx += 1
                self._push_next()
                return batch

    def next(self):
        return self.__next__()

    def __iter__(self):
        return self

    def shutdown(self):
        """Shutdown internal workers by pushing terminate signals."""
        if not self._shutdown:
            for _ in range(self._num_workers):
                self._key_queue.put((None, None, None))
            self._data_queue.put((None, None))
            self._shutdown = True


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

    """
    def __init__(self, transform_fns, dataset, interval=1, batch_size=None, shuffle=False,
                 sampler=None, last_batch=None, batch_sampler=None, batchify_fn=None,
                 num_workers=0, pin_memory=False):
        super(RandomTransformDataLoader, self).__init__(
            dataset, batch_size, shuffle, sampler,
            last_batch, batch_sampler, batchify_fn, num_workers)
        self._transform_fns = transform_fns
        assert len(self._transform_fns) > 0
        self._interval = max(int(interval), 1)
        self._pin_memory = pin_memory

    def __iter__(self):
        if self._num_workers == 0:
            def same_process_iter():
                t = np.random.choice(self._transform_fns)
                for ib, batch in enumerate(self._batch_sampler):
                    if (ib + 1) % self._interval == 0:
                        t = np.random.choice(self._transform_fns)
                    yield self._batchify_fn([self._dataset.transform(t)[idx] for idx in batch])
            return same_process_iter()
        else:
            return _RandomTransformMultiWorkerIter(
                self._transform_fns, self._interval, self._num_workers, self._dataset,
                self._batchify_fn, self._batch_sampler, self._pin_memory)
