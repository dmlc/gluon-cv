"""Dataloader helper functions. Synchronize slices for both data and label."""
__all__ = ['split_data', 'split_and_load']

from mxnet import ndarray

def split_data(data, num_slice, batch_axis=0, even_split=True, multiplier=1):
    """Splits an NDArray into `num_slice` slices along `batch_axis`.
    Usually used for data parallelism where each slices is sent
    to one device (i.e. GPU).

    Parameters
    ----------
    data : NDArray
        A batch of data.
    num_slice : int
        Number of desired slices.
    batch_axis : int, default 0
        The axis along which to slice.
    even_split : bool, default True
        Whether to force all slices to have the same number of elements.
        If `True`, an error will be raised when `num_slice` does not evenly
        divide `data.shape[batch_axis]`.
    multiplier : int, default 1
        The batch size has to be the multiples of multiplier

    Returns
    -------
    list of NDArray
        Return value is a list even if `num_slice` is 1.
    """
    size = data.shape[batch_axis]
    if even_split and size % num_slice != 0:
        raise ValueError(
            "data with shape %s cannot be evenly split into %d slices along axis %d. " \
            "Use a batch size that's multiple of %d or set even_split=False to allow " \
            "uneven partitioning of data."%(
                str(data.shape), num_slice, batch_axis, num_slice))

    step = (int(size / multiplier) // num_slice) * multiplier

    # If size < num_slice, make fewer slices
    if not even_split and size < num_slice:
        step = 1
        num_slice = size

    if batch_axis == 0:
        slices = [data[i*step:(i+1)*step] if i < num_slice - 1 else data[i*step:size]
                  for i in range(num_slice)]
    elif even_split:
        slices = ndarray.split(data, num_outputs=num_slice, axis=batch_axis)
    else:
        slices = [ndarray.slice_axis(data, batch_axis, i*step, (i+1)*step)
                  if i < num_slice - 1 else
                  ndarray.slice_axis(data, batch_axis, i*step, size)
                  for i in range(num_slice)]
    return slices


def split_and_load(data, ctx_list, batch_axis=0, even_split=True, multiplier=1):
    """Splits an NDArray into `len(ctx_list)` slices along `batch_axis` and loads
    each slice to one context in `ctx_list`.

    Parameters
    ----------
    data : NDArray
        A batch of data.
    ctx_list : list of Context
        A list of Contexts.
    batch_axis : int, default 0
        The axis along which to slice.
    even_split : bool, default True
        Whether to force all slices to have the same number of elements.
    multiplier : int, default 1
        The batch size has to be the multiples of channel multiplier. Need to investigate further.

    Returns
    -------
    list of NDArray
        Each corresponds to a context in `ctx_list`.
    """
    if not isinstance(data, ndarray.NDArray):
        data = ndarray.array(data, ctx=ctx_list[0])
    if len(ctx_list) == 1:
        return [data.as_in_context(ctx_list[0])]

    slices = split_data(data, len(ctx_list), batch_axis, even_split, multiplier)
    return [i.as_in_context(ctx) for i, ctx in zip(slices, ctx_list)]
