# pylint: disable=consider-using-enumerate,redefined-builtin,broad-except
"""Utils for Semantic Segmentation"""
import threading

from mxnet import autograd
from mxnet.ndarray import NDArray

__all__ = ['ModelDataParallel', 'CriterionDataParallel']

class ModelDataParallel(object):
    """Data parallelism

    Inputs and outputs are both list of NDArrays in different contexts.
    In the forward pass, the module is replicated on each device,
    and each replica handles a portion of the input. During the backwards
    pass, gradients from each replica are summed into the original module.

    Parameters
    ----------
    module : object
        Network to be parallelized.
    ctx : list
        A list of contexts to use.


    Inputs:

        - **inputs**: list of input (NDArrays)

    Outputs:

        - **outputs**: list of output (NDArrays)

    Example::

        >>> ctx = [mx.gpu(0), mx.gpu(1)]
        >>> net = ModelDataParallel(model, ctx=ctx)
        >>> x = gluon.utils.split_and_load(data, ctx_list=ctx)
        >>> y = net(x)
    """
    def __init__(self, module, ctx, sync=False):
        #super(ModelDataParallel, self).__init__()
        self.ctx = ctx
        module.collect_params().reset_ctx(ctx=ctx)
        self.module = module
        self.sync = sync

    def __call__(self, inputs):
        if self.sync:
            return _parallel_apply(self.module, inputs)
        else:
            if isinstance(inputs, NDArray):
                return self.module(inputs)
            if len(inputs) == 1:
                return (self.module(inputs[0]), )

            outputs = [self.module(X) for X in inputs]
            return outputs


class CriterionDataParallel(object):
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
        >>> net = ModelDataParallel(model, ctx=ctx)
        >>> criterion = CriterionDataParallel(criterion)
        >>> x = gluon.utils.split_and_load(data, ctx_list=ctx)
        >>> t = gluon.utils.split_and_load(target, ctx_list=ctx)
        >>> y = net(x)
        >>> losses = criterion(y, t)
    """
    def __init__(self, module, sync=False):
        #super(CriterionDataParallel, self).__init__()
        self.module = module
        self.sync = sync

    def __call__(self, inputs, targets):
        if self.sync:
            return _criterion_parallel_apply(self.module, inputs, targets)
        else:
            if isinstance(inputs, NDArray):
                return self.module(inputs, targets)
            if len(inputs) == 1:
                return (self.module(inputs[0]), targets[0])

            outputs = []
            for i in range(len(inputs)):
                output = self.module(inputs[i], targets[i])
                outputs.append(output)
            return outputs


def _parallel_apply(module, inputs, kwargs_tup=None):
    if kwargs_tup:
        assert len(inputs) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(inputs)

    if isinstance(inputs, NDArray):
        return module(inputs, **kwargs_tup[0])
    if len(inputs) == 1:
        return (module(inputs[0], **kwargs_tup[0]), )

    lock = threading.Lock()
    results = {}

    def _worker(i, module, input, kwargs, results, is_training, lock):
        try:
            if is_training:
                with autograd.record():
                    output = module(input, **kwargs)
                    output.wait_to_read()
            else:
                output = module(input, **kwargs)
                output.wait_to_read()
            with lock:
                results[i] = output
        except Exception as e:
            with lock:
                results[i] = e

    is_training = bool(autograd.is_training())
    threads = [threading.Thread(target=_worker,
                                args=(i, module, input, kwargs, results,
                                      is_training, lock),
                               )
               for i, (input, kwargs) in
               enumerate(zip(inputs, kwargs_tup))]

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
    return outputs


def _criterion_parallel_apply(module, inputs, targets, kwargs_tup=None):
    assert len(targets) == len(inputs)
    if kwargs_tup:
        assert len(inputs) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(inputs)
    # Fast track
    if len(inputs) == 1:
        return (module(inputs[0], targets[0], **kwargs_tup[0]), )

    lock = threading.Lock()
    results = {}

    def _worker(i, module, input, target, kwargs, results, is_training, lock):
        try:
            if is_training:
                with autograd.record():
                    output = module(input, target, **kwargs)
                    output.wait_to_read()
            else:
                output = module(input, target, **kwargs)
                output.wait_to_read()
            with lock:
                results[i] = output
        except Exception as e:
            with lock:
                results[i] = e

    is_training = bool(autograd.is_training())

    threads = [threading.Thread(target=_worker,
                                args=(i, module, input, target,
                                      kwargs, results, is_training, lock),
                               )
               for i, (input, target, kwargs) in
               enumerate(zip(inputs, targets, kwargs_tup))]

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
    return outputs
