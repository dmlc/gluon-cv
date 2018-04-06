"""DataLoader utils."""
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
    """Wrapper of `mxnet.gluon.data.DataLoader` for detection.

    It loads data from a dataset and returns mini-batches of data.

    Parameters
    ----------
    dataset : Dataset
        Source dataset. Note that numpy and mxnet arrays can be directly used
        as a Dataset.
    batch_size : int
        Size of mini-batch.
    shuffle : bool
        Whether to shuffle the samples.
    sampler : Sampler
        The sampler to use. Either specify sampler or shuffle, not both.
    last_batch : {'keep', 'discard', 'rollover'}
        How to handle the last batch if batch_size does not evenly divide
        `len(dataset)`.
        keep - A batch with less samples than previous batches is returned.
        discard - The last batch is discarded if its incomplete.
        rollover - The remaining samples are rolled over to the next epoch.
    batch_sampler : Sampler
        A sampler that returns mini-batches. Do not specify batch_size,
        shuffle, sampler, and last_batch if batch_sampler is specified.
    batchify_fn : callable
        Callback function to allow users to specify how to merge samples
        into a batch. Defaults to `default_pad_batchify_fn`::
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
        `num_workers > 0` is not supported on Windows yet.
    """
    def __init__(self, dataset, batch_size=None, shuffle=False, sampler=None,
                 last_batch=None, batch_sampler=None, batchify_fn=None,
                 num_workers=0):
        if batchify_fn is None:
            if num_workers > 0:
                batchify_fn = default_mp_pad_batchify_fn
            else:
                batchify_fn = default_pad_batchify_fn
        super(DetectionDataLoader, self).__init__(
            dataset, batch_size, shuffle, sampler, last_batch,
            batch_sampler, batchify_fn, num_workers)
