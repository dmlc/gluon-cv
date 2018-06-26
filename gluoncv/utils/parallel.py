"""Utils for Semantic Segmentation"""
# pylint: disable=consider-using-enumerate,redefined-builtin,broad-except
import threading

from mxnet import autograd
from mxnet.ndarray import NDArray
from mxnet.gluon.utils import split_and_load

__all__ = ['DataParallelModel', 'DataParallelCriterion', 'parallel_backward']

class DataParallelModel(object):
    """Data parallelism

    Hide the difference of single/multiple GPUs to the user.
    Inputs and outputs are both list of NDArrays in different contexts.
    In the forward pass, the module is replicated on each device,
    and each replica handles a portion of the input. During the backwards
    pass, gradients from each replica are summed into the original module.

    Parameters
    ----------
    module : object
        Network to be parallelized.
    ctx_list : list
        A list of contexts
    sync : bool
        enable synchronization (default: False).


    Inputs:
        - **inputs**: list of input (NDArrays)

    Outputs:
        - **outputs**: list of output (NDArrays)

    Example::
        >>> ctx = [mx.gpu(0), mx.gpu(1)]
        >>> net = DataParallelModel(model, ctx_list=ctx)
        >>> y = net(x)
    """
    def __init__(self, module, ctx_list=None, sync=False):
        module.collect_params().reset_ctx(ctx=ctx_list)
        self.ctx_list = ctx_list
        self.module = module
        self.sync = sync

    def __call__(self, *inputs, **kwargs):
        if not self.ctx_list:
            return self.module(*inputs, **kwargs)
        inputs, kwargs = split_load_kwargs(inputs, kwargs, self.ctx_list)
        assert(len(inputs) == len(self.ctx_list))
        if len(self.ctx_list) == 1:
            return tuple([tuple_map(self.module(*inputs[0], **kwargs[0]))])
        return parallel_apply(self.module, inputs, kwargs, self.sync)

    def __repr__(self):
        return 'DataParallel:\n module = {' + self.module.__repr__() + '}'


class DataParallelCriterion(object):
    """Criterion data parallelism

    Parameters
    ----------
    module : object
        Network to be parallelized.
    ctx : list
        A list of contexts to use.


    Inputs:

        - **inputs**: list of inputs (NDArrays)
        - **targets**: list of labels (NDArrays)

    Outputs:

        - **outputs**: list of output (NDArrays)

    Example::

        >>> ctx = [mx.gpu(0), mx.gpu(1)]
        >>> net = DataParallelModel(model, ctx=ctx)
        >>> criterion = DataParallelCriterion(criterion)
        >>> y = net(x)
        >>> losses = criterion(y, t)
    """
    def __init__(self, module, ctx_list=None, sync=False):
        self.module = module
        self.ctx_list = ctx_list
        self.sync = sync

    def __call__(self, inputs, *targets, **kwargs):
        # the inputs should be the outputs of DataParallelModel
        if not self.ctx_list:
            return self.module(inputs, *targets, **kwargs)
        targets, kwargs = split_load_kwargs(targets, kwargs, self.ctx_list)
        assert(len(targets) == len(self.ctx_list))
        if len(self.ctx_list) == 1:
            return tuple_map(self.module(*(inputs[0] + targets[0]), **kwargs[0]))
        assert(len(inputs) == len(self.ctx_list))
        return criterion_parallel_apply(self.module, inputs, targets, kwargs, self.sync)


def split_load_kwargs(inputs, kwargs, ctx_list, batch_axis=0):
    r"""Split with support for kwargs dictionary"""
    def split_map(obj):
        if isinstance(obj, NDArray):
            return split_and_load(obj, ctx_list, batch_axis, even_split=False)
        if isinstance(obj, tuple) and len(obj) > 0:
            return list(zip(*map(split_map, obj)))
        if isinstance(obj, list) and len(obj) > 0:
            return list(map(list, zip(*map(split_map, obj))))
        if isinstance(obj, dict) and len(obj) > 0:
            return list(map(type(obj), zip(*map(split_map, obj.items()))))
        return [obj for targets in ctx_list]
    inputs = split_map(inputs) if inputs else []
    kwargs = split_map(kwargs) if kwargs else []
    if len(inputs) < len(kwargs):
        inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
    elif len(kwargs) < len(inputs):
        kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
    inputs = tuple(inputs)
    kwargs = tuple(kwargs)
    return inputs, kwargs


def tuple_map(obj):
    if isinstance(obj, NDArray):
        return (obj,)
    if isinstance(obj, list) and len(obj) > 0:
        return tuple(obj)
    return obj


def parallel_apply(module, inputs, kwargs_tup=None, sync=False):
    """Parallel applying model forward"""
    if kwargs_tup is not None:
        assert len(inputs) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(inputs)

    lock = threading.Lock()
    results = {}

    def _worker(i, module, input, kwargs, results, is_recording, is_training, lock):
        try:
            if is_recording:
                with autograd.record(is_training):
                    output = tuple_map(module(*input, **kwargs))
                    for out in output:
                        out.wait_to_read()
            else:
                output = tuple_map(module(*input, **kwargs))
                for out in output:
                    out.wait_to_read()
            with lock:
                results[i] = output
        except Exception as e:
            with lock:
                results[i] = e

    is_training = autograd.is_training()
    is_recording = autograd.is_recording()
    threads = [threading.Thread(target=_worker,
                                args=(i, module, input, kwargs, results,
                                      is_recording, is_training, lock),
                               )
               for i, (input, kwargs) in
               enumerate(zip(inputs, kwargs_tup))]

    if sync:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        outputs = []
        for i in range(len(inputs)):
            output = results[i]
            if isinstance(output, Exception):
                raise output
            outputs.append(output)
        return tuple(outputs)
    else:
        outputs = [tuple_map(module(*input, **kwargs))
                   for (input, kwargs) in zip(inputs, kwargs_tup)]
        return tuple(outputs)


def criterion_parallel_apply(module, inputs, targets, kwargs_tup=None, sync=False):
    """Data Parallel Criterion"""
    if kwargs_tup:
        assert len(inputs) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(inputs)

    lock = threading.Lock()
    results = {}

    def _worker(i, module, input, target, kwargs, results, is_recording, is_training, lock):
        try:
            if is_recording:
                with autograd.record(is_training):
                    output = module(*(input + target), **kwargs)
                    output.wait_to_read()
            else:
                output = module(*(input + target), **kwargs)
                output.wait_to_read()
            with lock:
                results[i] = output
        except Exception as e:
            with lock:
                results[i] = e

    is_training = bool(autograd.is_training())
    is_recording = autograd.is_recording()

    threads = [threading.Thread(target=_worker,
                                args=(i, module, input, target,
                                      kwargs, results, is_recording, is_training, lock),
                               )
               for i, (input, target, kwargs) in
               enumerate(zip(inputs, targets, kwargs_tup))]

    if sync:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        outputs = []
        for i in range(len(inputs)):
            output = results[i]
            if isinstance(output, Exception):
                raise output
            outputs.append(output)
        return tuple(outputs)
    else:
        outputs = [module(*(input + target), **kwargs) \
            for (input, target, kwargs) in zip(inputs, targets, kwargs_tup)]
        return tuple(outputs)

def parallel_backward(losses, sync=True):
    """Parallel Backward for CustomOp"""
    def _worker(loss):
        autograd.backward(loss)
    threads = [threading.Thread(target=_worker, args=(loss,)) for loss in losses]
    if sync:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        for loss in losses:
            loss.backward()
