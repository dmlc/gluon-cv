"""Utils for Semantic Segmentation"""
# pylint: disable=consider-using-enumerate,redefined-builtin,broad-except
import threading

from mxnet import autograd
from mxnet.gluon.utils import split_and_load
from mxnet.ndarray import NDArray

try:
    import Queue as queue
except ImportError:
    import queue

__all__ = ['DataParallelModel', 'DataParallelCriterion', 'parallel_backward', 'Parallelizable',
           'Parallel']


class Parallelizable(object):
    """Base class for parallelizable unit of work, which can be invoked by `Parallel`.
    The subclass must implement the `forward_backward` method, and be used
    together with `Parallel`. For example::
        class ParallelNet(Parallelizable):
            def __init__(self):
                self._net = Model()
                self._loss = gluon.loss.SoftmaxCrossEntropyLoss()
            def forward_backward(self, x):
                data, label = x
                with mx.autograd.record():
                    out = self._net(data)
                    loss = self._loss(out, label)
                loss.backward()
                return loss
        net = ParallelNet()
        ctx = [mx.gpu(0), mx.gpu(1)]
        parallel = Parallel(len(ctx), net)
        # Gluon block is initialized after forwarding the first batch
        initialized = False
        for batch in batches:
            for x in gluon.utils.split_and_load(batch, ctx):
                parallel.put(x)
            losses = [parallel.get() for _ in ctx]
            trainer.step()
    """

    def forward_backward(self, x):
        """ Forward and backward computation. """
        raise NotImplementedError()


class Parallel(object):
    """Class for parallel processing with `Parallelizable`s. It invokes a
    `Parallelizable` with multiple Python threads. For example::
        class ParallelNet(Parallelizable):
            def __init__(self):
                self._net = Model()
                self._loss = gluon.loss.SoftmaxCrossEntropyLoss()
            def forward_backward(self, x):
                data, label = x
                mx.autograd.record():
                    out = self._net(data)
                    loss = self._loss(out, label)
                loss.backward()
                return loss
        net = ParallelNet()
        ctx = [mx.gpu(0), mx.gpu(1)]
        parallel = Parallel(len(ctx), net)
        for batch in batches:
            for x in gluon.utils.split_and_load(batch, ctx):
                parallel.put(x)
            losses = [parallel.get() for _ in ctx]
            trainer.step()
    Parameters
    ----------
    num_workers : int
        Number of worker threads. If set to 0, the main thread is used as the worker for
        debugging purpose.
    parallelizable :
        Parallelizable net whose `forward` and `backward` methods are invoked
        by multiple worker threads.
    serial_init : bool, default True
        Execute the first `num_workers` inputs in main thread, so that the `Block`
        used in `parallizable` is initialized serially. Initialize a `Block` with
        multiple threads may cause unexpected behavior.
    """

    class _StopSignal:
        """Internal class to signal stop. """

        def __init__(self, msg):
            self._msg = msg

    def __init__(self, num_workers, parallizable, serial_init=True):
        self._in_queue = queue.Queue(-1)
        self._out_queue = queue.Queue(-1)
        self._num_workers = num_workers
        self._threads = []
        self._parallizable = parallizable
        self._num_serial = num_workers if serial_init else 0

        def _worker(in_queue, out_queue, parallel):
            while True:
                x = in_queue.get()
                if isinstance(x, Parallel._StopSignal):
                    return
                out = parallel.forward_backward(x)
                out_queue.put(out)

        arg = (self._in_queue, self._out_queue, self._parallizable)
        for _ in range(num_workers):
            thread = threading.Thread(target=_worker, args=arg)
            self._threads.append(thread)
            thread.start()

    def put(self, x):
        """Assign input `x` to an available worker and invoke
        `parallizable.forward_backward` with x. """
        if self._num_serial > 0 or len(self._threads) == 0:
            self._num_serial -= 1
            out = self._parallizable.forward_backward(x)
            self._out_queue.put(out)
        else:
            self._in_queue.put(x)

    def get(self):
        """Get an output of previous `parallizable.forward_backward` calls.
        This method blocks if none of previous `parallizable.forward_backward`
        calls have return any result. """
        return self._out_queue.get()

    def __del__(self):
        for thread in self._threads:
            if thread.is_alive():
                self._in_queue.put(self._StopSignal('stop'))
        for thread in self._threads:
            thread.join(10)


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
        assert (len(inputs) == len(self.ctx_list))
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
        assert (len(targets) == len(self.ctx_list))
        if len(self.ctx_list) == 1:
            return tuple_map(self.module(*(inputs[0] + targets[0]), **kwargs[0]))
        assert (len(inputs) == len(self.ctx_list))
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
        return [obj for _ in ctx_list]

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
                                      is_recording, is_training, lock),)
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
                                      kwargs, results, is_recording, is_training, lock),)
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
