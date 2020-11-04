"""Batchify functions.
They can be used in Gluon data loader to help combine individual samples
into batches for fast processing."""

import logging

import numpy as np
import mxnet as mx

__all__ = ['Stack', 'Pad', 'Append', 'Tuple', 'FasterRCNNTrainBatchify']


def _pad_arrs_to_max_length(arrs, pad_axis, pad_val, num_shards=1, use_shared_mem=False):
    """Inner Implementation of the Pad batchify
    Parameters
    ----------
    arrs : list
    pad_axis : int or tuple
    pad_val : number
    num_shards : int, default 1
    use_shared_mem : bool, default False
    Returns
    -------
    ret : NDArray or a list of NDArrays
    original_length : NDArray or a list of NDArrays
    """
    if not isinstance(arrs[0], (mx.nd.NDArray, np.ndarray)):
        arrs = [np.asarray(ele) for ele in arrs]
    if isinstance(pad_axis, tuple):
        original_length = []
        for axis in pad_axis:
            original_length.append(np.array([ele.shape[axis] for ele in arrs]))
        original_length = np.stack(original_length).T
    else:
        original_length = np.array([ele.shape[pad_axis] for ele in arrs])
        pad_axis = [pad_axis]
    if len(original_length) % num_shards != 0:
        logging.warning(
            'Batch size cannot be evenly split. Trying to shard %d items into %d shards',
            len(original_length), num_shards)
    original_length = np.array_split(original_length, num_shards)
    max_lengths = [np.max(l, axis=0, keepdims=len(pad_axis) == 1) for l in original_length]
    # add batch dimension
    ret_shape = [[l.shape[0], ] + list(arrs[0].shape) for l in original_length]
    for i, shape in enumerate(ret_shape):
        for j, axis in enumerate(pad_axis):
            shape[1 + axis] = max_lengths[i][j]
    if use_shared_mem:
        ret = [mx.nd.full(shape=tuple(shape), val=pad_val, ctx=mx.Context('cpu_shared', 0),
                          dtype=arrs[0].dtype) for shape in ret_shape]
        original_length = [mx.nd.array(l, ctx=mx.Context('cpu_shared', 0),
                                       dtype=np.int32) for l in original_length]
    else:
        ret = [mx.nd.full(shape=tuple(shape), val=pad_val, dtype=arrs[0].dtype) for shape in
               ret_shape]
        original_length = [mx.nd.array(l, dtype=np.int32) for l in original_length]
    for i, arr in enumerate(arrs):
        if ret[i // ret[0].shape[0]].shape[1:] == arr.shape:
            ret[i // ret[0].shape[0]][i % ret[0].shape[0]] = arr
        else:
            slices = [slice(0, l) for l in arr.shape]
            ret[i // ret[0].shape[0]][i % ret[0].shape[0]][tuple(slices)] = arr
    if len(ret) == len(original_length) == 1:
        return ret[0], original_length[0]
    return ret, original_length


def _stack_arrs(arrs, use_shared_mem=False):
    """Internal imple for stacking arrays."""
    if isinstance(arrs[0], mx.nd.NDArray):
        if use_shared_mem:
            out = mx.nd.empty((len(arrs),) + arrs[0].shape, dtype=arrs[0].dtype,
                              ctx=mx.Context('cpu_shared', 0))
            return mx.nd.stack(*arrs, out=out)
        else:
            return mx.nd.stack(*arrs)
    else:
        out = np.asarray(arrs)
        if use_shared_mem:
            return mx.nd.array(out, ctx=mx.Context('cpu_shared', 0))
        else:
            return mx.nd.array(out)


def _append_arrs(arrs, use_shared_mem=False, expand=False, batch_axis=0):
    """Internal impl for returning appened arrays as list."""
    if isinstance(arrs[0], mx.nd.NDArray):
        if use_shared_mem:
            out = [x.as_in_context(mx.Context('cpu_shared', 0)) for x in arrs]
        else:
            out = arrs
    else:
        if use_shared_mem:
            out = [mx.nd.array(x, ctx=mx.Context('cpu_shared', 0)) for x in arrs]
        else:
            out = [mx.nd.array(x) for x in arrs]

    # add batch axis
    if expand:
        out = [x.expand_dims(axis=batch_axis) for x in out]
    return out


class Stack(object):
    r"""Stack the input data samples to construct the batch.
    The N input samples must have the same shape/length and will be stacked to construct a batch.
    Examples
    --------
    >>> from gluoncv.data import batchify
    >>> # Stack multiple lists
    >>> a = [1, 2, 3, 4]
    >>> b = [4, 5, 6, 8]
    >>> c = [8, 9, 1, 2]
    >>> batchify.Stack()([a, b, c])
    [[1. 2. 3. 4.]
     [4. 5. 6. 8.]
     [8. 9. 1. 2.]]
    <NDArray 3x4 @cpu(0)>
    >>> # Stack multiple numpy.ndarrays
    >>> import numpy as np
    >>> a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    >>> b = np.array([[5, 6, 7, 8], [1, 2, 3, 4]])
    >>> batchify.Stack()([a, b])
    [[[1. 2. 3. 4.]
      [5. 6. 7. 8.]]
     [[5. 6. 7. 8.]
      [1. 2. 3. 4.]]]
    <NDArray 2x2x4 @cpu(0)>
    >>> # Stack multiple NDArrays
    >>> import mxnet as mx
    >>> a = mx.nd.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    >>> b = mx.nd.array([[5, 6, 7, 8], [1, 2, 3, 4]])
    >>> batchify.Stack()([a, b])
    [[[1. 2. 3. 4.]
      [5. 6. 7. 8.]]
     [[5. 6. 7. 8.]
      [1. 2. 3. 4.]]]
    <NDArray 2x2x4 @cpu(0)>
    """

    def __call__(self, data):
        """Batchify the input data
        Parameters
        ----------
        data : list
            The input data samples
        Returns
        -------
        batch_data : NDArray
        """
        return _stack_arrs(data, True)


class Pad(object):
    """Pad the input ndarrays along the specific padding axis and stack them to get the output.
    Input of the function will be N samples. Each sample should contain a single element that
    can be 1) numpy.ndarray, 2) mxnet.nd.NDArray, 3) list of numbers.
    You can set the `axis` and `pad_val` to determine the padding axis and
    value.
    The arrays will be padded to the largest dimension at `axis` and then
    stacked to form the final output. In addition, the function will output the original dimensions
    at the `axis` if ret_length is turned on.
    Parameters
    ----------
    axis : int or tuple, default 0
        The axis to pad the arrays. The arrays will be padded to the largest dimension at
        pad_axis. For example, assume the input arrays have shape
        (10, 8, 5), (6, 8, 5), (3, 8, 5) and the pad_axis is 0. Each input will be padded into
        (10, 8, 5) and then stacked to form the final output.
    pad_val : float or int, default 0
        The padding value.
    num_shards : int, default 1
        Number of shards to create. Each shard are padded separately.
    ret_length : bool, default False
        Whether to return the valid length in the output.
    Examples
    --------
    >>> from gluoncv.data import batchify
    >>> # Inputs are multiple lists
    >>> a = [1, 2, 3, 4]
    >>> b = [4, 5, 6]
    >>> c = [8, 2]
    >>> batchify.Pad()([a, b, c])
    [[ 1  2  3  4]
     [ 4  5  6  0]
     [ 8  2  0  0]]
    <NDArray 3x4 @cpu(0)>
    >>> # Also output the lengths
    >>> a = [1, 2, 3, 4]
    >>> b = [4, 5, 6]
    >>> c = [8, 2]
    >>> batchify.Pad(ret_length=True)([a, b, c])
    (
     [[1 2 3 4]
      [4 5 6 0]
      [8 2 0 0]]
     <NDArray 3x4 @cpu(0)>,
     [4 3 2]
     <NDArray 3 @cpu(0)>)
    >>> # Inputs are multiple ndarrays
    >>> import numpy as np
    >>> a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    >>> b = np.array([[5, 8], [1, 2]])
    >>> batchify.Pad(axis=1, pad_val=-1)([a, b])
    [[[ 1  2  3  4]
      [ 5  6  7  8]]
     [[ 5  8 -1 -1]
      [ 1  2 -1 -1]]]
    <NDArray 2x2x4 @cpu(0)>
    >>> # Inputs are multiple NDArrays
    >>> import mxnet as mx
    >>> a = mx.nd.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    >>> b = mx.nd.array([[5, 8], [1, 2]])
    >>> batchify.Pad(axis=1, pad_val=-1)([a, b])
    [[[ 1.  2.  3.  4.]
      [ 5.  6.  7.  8.]]
     [[ 5.  8. -1. -1.]
      [ 1.  2. -1. -1.]]]
    <NDArray 2x2x4 @cpu(0)>
    """

    def __init__(self, axis=0, pad_val=0, num_shards=1, ret_length=False):
        self._axis = axis
        assert isinstance(axis, (int, tuple, list)), 'axis must be an integer, tuple or list! ' \
                                                     'Received axis=%s, type=%s.' % (str(axis),
                                                                                     str(type(
                                                                                         axis)))
        self._pad_val = pad_val
        self._num_shards = num_shards
        self._ret_length = ret_length

    def __call__(self, data):
        """Batchify the input data.
        Parameters
        ----------
        data : list
            A list of N samples. Each sample can be 1) ndarray or
             2) a list/tuple of ndarrays
        Returns
        -------
        batch_data: NDArray
            Data in the minibatch. Shape is (N, ...)
        valid_length: NDArray, optional
            The sequences' original lengths at the padded axis. Shape is (N,). This will only be
            returned in `ret_length` is True.
        """
        if isinstance(data[0], (mx.nd.NDArray, np.ndarray, list)):
            padded_arr, original_length = _pad_arrs_to_max_length(data, self._axis, self._pad_val,
                                                                  self._num_shards, True)
            if self._ret_length:
                return padded_arr, original_length
            else:
                return padded_arr

        else:
            raise NotImplementedError


class Append(object):
    r"""Loosely return list of the input data samples.
    There is no constraint of shape for any of the input samples, however, you will
    only be able to apply single batch operations since the output have different shapes.

    Examples
    --------
    >>> a = [1, 2, 3, 4]
    >>> b = [4, 5, 6]
    >>> c = [8, 2]
    >>> batchify.Append()([a, b, c])
    [
    [[1. 2. 3. 4.]]
    <NDArray 1x4 @cpu_shared(0)>,
    [[4. 5. 6.]]
    <NDArray 1x3 @cpu_shared(0)>,
    [[8. 2.]]
    <NDArray 1x2 @cpu_shared(0)>
    ]
    """

    def __init__(self, expand=True, batch_axis=0):
        self._expand = expand
        self._batch_axis = batch_axis

    def __call__(self, data):
        """Batchify the input data.

        Parameters
        ----------
        data : list
            The input data samples
        Returns
        -------
        batch_data : NDArray
        """
        return _append_arrs(data, use_shared_mem=True,
                            expand=self._expand, batch_axis=self._batch_axis)


class Tuple(object):
    """Wrap multiple batchify functions to form a function apply each input function on each
    input fields respectively.
    Each data sample should be a list or tuple containing multiple attributes. The `i`th batchify
    function stored in `Tuple` will be applied on the `i`th attribute. For example, each
    data sample is (nd_data, label). You can wrap two batchify functions using
    `Wrap(DataBatchify, LabelBatchify)` to batchify nd_data and label correspondingly.
    Parameters
    ----------
    fn : list or tuple or callable
        The batchify functions to wrap.
    *args : tuple of callable
        The additional batchify functions to wrap.
    Examples
    --------
    >>> from gluoncv.data import batchify
    >>> a = ([1, 2, 3, 4], 0)
    >>> b = ([5, 7], 1)
    >>> c = ([1, 2, 3, 4, 5, 6, 7], 0)
    >>> batchify.Tuple(batchify.Pad(), batchify.Stack())([a, b])
    (
     [[1 2 3 4]
      [5 7 0 0]]
     <NDArray 2x4 @cpu(0)>,
     [0. 1.]
     <NDArray 2 @cpu(0)>)
    >>> # Input can also be a list
    >>> batchify.Tuple([batchify.Pad(), batchify.Stack()])([a, b])
    (
     [[1 2 3 4]
      [5 7 0 0]]
     <NDArray 2x4 @cpu(0)>,
     [0. 1.]
     <NDArray 2 @cpu(0)>)
    >>> # Another example
    >>> a = ([1, 2, 3, 4], [5, 6], 1)
    >>> b = ([1, 2], [3, 4, 5, 6], 0)
    >>> c = ([1], [2, 3, 4, 5, 6], 0)
    >>> batchify.Tuple(batchify.Pad(), batchify.Pad(), batchify.Stack())([a, b, c])
    (
     [[1 2 3 4]
      [1 2 0 0]
      [1 0 0 0]]
     <NDArray 3x4 @cpu(0)>,
     [[5 6 0 0 0]
      [3 4 5 6 0]
      [2 3 4 5 6]]
     <NDArray 3x5 @cpu(0)>,
     [1. 0. 0.]
     <NDArray 3 @cpu(0)>)
    """

    def __init__(self, fn, *args):
        if isinstance(fn, (list, tuple)):
            assert len(args) == 0, 'Input pattern not understood. The input of Tuple can be ' \
                                   'Tuple(A, B, C) or Tuple([A, B, C]) or Tuple((A, B, C)). ' \
                                   'Received fn=%s, args=%s' % (str(fn), str(args))
            self._fn = fn
        else:
            self._fn = (fn,) + args
        for i, ele_fn in enumerate(self._fn):
            assert hasattr(ele_fn, '__call__'), 'Batchify functions must be callable! ' \
                                                'type(fn[%d]) = %s' % (i, str(type(ele_fn)))

    def __call__(self, data):
        """Batchify the input data.

        Parameters
        ----------
        data : list
            The samples to batchfy. Each sample should contain N attributes.
        Returns
        -------
        ret : tuple
            A tuple of length N. Contains the batchified result of each attribute in the input.
        """
        assert len(data[0]) == len(self._fn), \
            'The number of attributes in each data sample should contains' \
            ' {} elements, given {}.'.format(len(self._fn), len(data[0]))
        ret = []
        for i, ele_fn in enumerate(self._fn):
            ret.append(ele_fn([ele[i] for ele in data]))
        return tuple(ret)


class FasterRCNNTrainBatchify(object):
    """Batchify FasterRCNN data with 5 elements: img, bbox, rpn_cls_targets, rpn_box_targets,
    and rpn_box_masks
    Parameters
    ----------
    net : mxnet.gluon.HybridBlock.
        The faster-rcnn network used to infer output shape.
    num_shards : int, default 1
        Number of shards. Each shard corresponds to one device.
    """

    def __init__(self, net, num_shards=1):
        self._feat_sym = net.features(mx.sym.var(name='data'))
        self._num_shards = num_shards
        self._img_pad = Pad(axis=(1, 2), num_shards=num_shards, ret_length=False)
        self._label_pad = Pad(pad_val=-1, num_shards=num_shards, ret_length=False)
        self.NUM_ELEMENTS = 5

    def __call__(self, data):
        """Batchify the input data.
        Parameters
        ----------
        data : list
            The input data samples
        Returns
        -------
        batch_data : a tuple of NDArray
        """
        assert len(data[0]) == self.NUM_ELEMENTS, \
            'The number of attributes in each data sample should contains' \
            ' {} elements, given {}.'.format(self.NUM_ELEMENTS, len(data[0]))
        data = tuple(zip(*data))
        sharded_img = self._img_pad(data[0])
        sharded_label = self._label_pad(data[1])
        # deal with rpn targets
        in_shapes = [img.shape if len(img.shape) == 2 else img.shape[1:] for shard in sharded_img
                     for img in shard]
        sharded_cls_targets, sharded_box_targets, sharded_box_masks = [], [], []
        for in_shape, cls_targets, box_targets, box_masks in zip(in_shapes, *data[2:]):
            padded_cls_targets, padded_box_targets, padded_box_masks = [], [], []
            for feat_sym, cls_target, box_target, box_mask in zip(self._feat_sym, cls_targets,
                                                                  box_targets, box_masks):
                _, _, w, h = feat_sym.infer_shape(data=(1, 3, in_shape[0], in_shape[1]))[1][0]
                padded_cls_target = mx.nd.ones(shape=(w, h, cls_target.shape[-1])) * -1.0
                padded_box_target = mx.nd.zeros(shape=(w, h, box_target.shape[-1]))
                padded_box_mask = mx.nd.zeros(shape=(w, h, box_mask.shape[-1]))
                padded_cls_target[:cls_target.shape[0], :cls_target.shape[1]] = cls_target
                padded_box_target[:box_target.shape[0], :box_target.shape[1]] = box_target
                padded_box_mask[:box_mask.shape[0], :box_mask.shape[1]] = box_mask
                padded_cls_targets.append(padded_cls_target.reshape(1, -1))
                padded_box_targets.append(padded_box_target.reshape(1, -1, 4))
                padded_box_masks.append(padded_box_mask.reshape(1, -1, 4))
            sharded_cls_targets.append(mx.nd.concat(*padded_cls_targets, dim=1))
            sharded_box_targets.append(mx.nd.concat(*padded_box_targets, dim=1))
            sharded_box_masks.append(mx.nd.concat(*padded_box_masks, dim=1))
        shard_size = int(np.ceil(1.0 * len(sharded_cls_targets) / self._num_shards))
        for i in range(self._num_shards):
            start_ind = int(i * shard_size)
            end_ind = int(start_ind + shard_size)
            sharded_cls_targets[i], sharded_box_targets[i], sharded_box_masks[i] \
                = mx.nd.concat(*sharded_cls_targets[start_ind:end_ind], dim=0).as_in_context(
                    mx.Context('cpu_shared', 0)), \
                  mx.nd.concat(*sharded_box_targets[start_ind:end_ind], dim=0).as_in_context(
                      mx.Context('cpu_shared', 0)), \
                  mx.nd.concat(*sharded_box_masks[start_ind:end_ind], dim=0).as_in_context(
                      mx.Context('cpu_shared', 0))

        return sharded_img, sharded_label, tuple(sharded_cls_targets[:self._num_shards]), \
               tuple(sharded_box_targets[:self._num_shards]), \
               tuple(sharded_box_masks[:self._num_shards])


class MaskRCNNTrainBatchify(object):
    """Batchify FasterRCNN data with 5 elements: img, bbox, rpn_cls_targets, rpn_box_targets,
    and rpn_box_masks
    Parameters
    ----------
    net : mxnet.gluon.HybridBlock.
        The faster-rcnn network used to infer output shape.
    num_shards : int, default 1
        Number of shards. Each shard corresponds to one device.
    """

    def __init__(self, net, num_shards=1):
        self._feat_sym = net.features(mx.sym.var(name='data'))
        self._num_shards = num_shards
        self._faster_batchify = FasterRCNNTrainBatchify(net, num_shards)
        self._mask_pad = Pad(axis=(0, 1, 2), pad_val=0, num_shards=num_shards, ret_length=False)
        self.NUM_ELEMENTS = 6

    def __call__(self, data):
        """Batchify the input data.
        Parameters
        ----------
        data : list
            The input data samples
        Returns
        -------
        batch_data : a tuple of NDArray
        """
        assert len(data[0]) == self.NUM_ELEMENTS, \
            'The number of attributes in each data sample should contains' \
            ' {} elements, given {}.'.format(self.NUM_ELEMENTS, len(data[0]))
        sharded_img, sharded_label, sharded_cls_targets, sharded_box_targets, sharded_box_masks = \
            self._faster_batchify([ele[:5] for ele in data])
        sharded_masks = self._mask_pad([ele[5] for ele in data])

        return sharded_img, sharded_label, sharded_masks, tuple(
            sharded_cls_targets[:self._num_shards]), \
               tuple(sharded_box_targets[:self._num_shards]), \
               tuple(sharded_box_masks[:self._num_shards])
