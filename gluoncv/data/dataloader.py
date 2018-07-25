"""DataLoader utils."""
import numpy as np
from mxnet import nd
from mxnet import context
from mxnet.gluon.data import DataLoader, BatchSampler, RandomSampler

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


class RandomTransformDataLoader(object):
    def __init__(self, dataset, batch_size, transform_fns, interval=1,
                 last_batch=None, batchify_fn=None, num_workers=0):
        assert isinstance(transform_fns, (list, tuple))
        self._num_loader = len(transform_fns)
        self._transform_fns = transform_fns
        assert len(dataset) >= batch_size * self._num_loader, ("Dataset not large enough with batch"
            " size {} and transforms {}, given len(dataset): {}".format(
            len(dataset), batch_size, self._num_loader))
        self._dataset = dataset
        self._batch_size = batch_size
        self._interval = int(interval)
        self._batchify_fn = batchify_fn
        self._last_batch = last_batch if last_batch else 'keep'
        self._num_workers = num_workers
        self._batch_sampler = BatchSampler(RandomSampler(len(dataset)), batch_size, self._last_batch)
        self._loaders = []
        self._current_loader = None
        self._prev = []

    def _reset(self):
        """Shuffle sampler for next epoch."""
        indices = np.concatenate((np.array(self._prev), np.arange(len(self._dataset))))
        np.random.shuffle(indices)
        residue = len(indices) % self._batch_size
        if self._last_batch == 'discard':
            indices = indices[:-residue]
        elif self._last_batch == 'rollover':
            indices = indices[:-residue]
            self._prev = indices[-residue:]
            print(self._prev)
        num_split = len(indices) // self._batch_size
        samplers = np.split(indices, num_split)
        samplers = np.split(np.array(np.split(indices, num_split)), self._num_loader)
        print(samplers)
        raise
        # if len(samplers[-1]) < self._batch_size and self._last_batch == 'rollover':
        #     self._prev = samplers[-1]
        #     samplers = samplers[:-1]
        samplers = [iter(x) for x in samplers]
        # print([list(xx) for xx in samplers])
        self._loaders = [iter(DataLoader(self._dataset.transform(t), batch_size=self._batch_size,
            sampler=s, last_batch='keep', batchify_fn=self._batchify_fn,
            num_workers=self._num_workers)) for s, t in zip(samplers, self._transform_fns)]
        self._loader_idx = np.random.choice(len(self._loaders))

    def __iter__(self):
        self._reset()

        for i in range(len(self)):
            if (i + 1) % self._interval == 0:
                self._loader_idx = np.random.choice(len(self._loaders))

            batch = None
            while self._loaders:
                try:
                    batch = next(self._loaders[self._loader_idx])
                    break
                except StopIteration:
                    self._loaders.pop(self._loader_idx)
                    self._loader_idx = np.random.choice(len(self._loaders))
            assert batch is not None
            yield batch

    def __len__(self):
        return len(self._batch_sampler)
