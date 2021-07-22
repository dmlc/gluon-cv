"""Utils for deep learning framework related functions"""
import numpy as np

__all__ = ['EarlyStopperOnPlateau', '_suggest_load_context', 'create_dummy_estimator']

def _dummy_constructor(self, *arg, **kwargs):
    raise RuntimeError(self.reason.format(type(self).__name__))

def create_dummy_estimator(name, reason):
    assert isinstance(reason, str)
    DummyEstimator = type(name, (object, ), {
        # constructor
        "__init__": _dummy_constructor,

        # data members
        "reason": reason,
    })
    return DummyEstimator


class EarlyStopperOnPlateau:
    """Early stopping on plateau helper.

    Parameters
    ----------
    patience : int, default is -1
        How many epochs with no improvement after which train will be early stopped.
        Negative patience means infinite petience.
    metric_fn : function, default is None
        The function to apply to metric value if any. For example, you can use
        the `metric_fn` to cast loss to negative values where lower loss is better.
        `min_delta`, `baseline_value` and `max_value` are all based on output of `metric_fn`.
    min_delta : float, default is 1e-4
        Early stopper ignores changes less than `min_delta` for metrics to ignore tiny fluctuates.
    baseline_value : float, default is 0.0
        The baseline metric value to be considered.
    max_value : float, default is 1.0
        Instantly early stop if reaching max value.

    """
    def __init__(self, patience=10, metric_fn=None,
                 min_delta=1e-4, baseline_value=None, max_value=np.Inf):
        self.patience = patience if patience > 0 else np.Inf
        self.metric_fn = metric_fn
        self.min_delta = np.abs(min_delta)
        self.baseline_value = baseline_value
        self.max_value = max_value
        self.reset()

    def reset(self):
        """reset the early stopper"""
        self.last_epoch = 0
        self.wait = 0
        self._should_stop = False
        self._message = ''
        if self.baseline_value is not None:
            self.best = self.baseline_value
        else:
            self.best = -np.Inf

    def update(self, metric_value, epoch=None):
        """Update with end of epoch metric.

        Parameters
        ----------
        metric_value : float
            The end of epoch metric.
        epoch : int, optional
            The real epoch in case the update function is not called in every epoch.

        """
        if _is_real_number(epoch):
            if _is_real_number(self.last_epoch):
                diff_epoch = epoch - self.last_epoch
            else:
                diff_epoch = 1
            self.last_epoch = epoch
        else:
            diff_epoch = 1
        if not _is_real_number(metric_value):
            return
        if self.metric_fn is not None:
            metric_value = self.metric_fn(metric_value)

        if metric_value > self.max_value:
            self._should_stop = True
            self._message = 'EarlyStop given {} vs. max {}'.format(metric_value, self.max_value)
        else:
            if metric_value - self.min_delta > self.best:
                self.best = metric_value
                self.wait = 0
            else:
                self.wait += diff_epoch
                if self.wait >= self.patience:
                    self._should_stop = True
                    self._message = 'EarlyStop after {} epochs: no better than {}'.format(self.patience, self.best)

    def get_early_stop_advice(self):
        """Get the early stop advice.

        Returns
        -------
        (bool, str)
            should_stop : bool
                Whether the stopper suggest OnPlateau pattern is active.
            message : str
                The detailed message why early stop is suggested, if `should_stop` is True.
        """
        return self._should_stop, self._message

def _is_real_number(x):
    """Check if x is a real number"""
    return isinstance(x, (int, float, complex)) and not isinstance(x, bool)

def _suggest_load_context(model, mode, orig_ctx):
    """Get the correct context given the mode"""
    if not isinstance(orig_ctx, (list, tuple)):
        orig_ctx = [orig_ctx]
    try:
        import mxnet as mx
    except ImportError:
        mx = None
    try:
        import torch
    except ImportError:
        torch = None
    if mx is not None and isinstance(model, mx.gluon.Block):
        if mode == 'auto':
            if orig_ctx[0].device_type == 'gpu':
                mode = 'gpu'
            else:
                mode = 'cpu'
        if mode == 'cpu':
            return [mx.cpu()]
        if mode == 'gpu':
            return [mx.gpu(i) for i in range(mx.context.num_gpus())]
        if isinstance(mode, (list, tuple)):
            if not all(isinstance(i, int) for i in mode):
                raise ValueError('Requires integer gpu id, given {}'.format(mode))
            return [mx.gpu(i) for i in mode if i in range(mx.context.num_gpus())]
    if torch is not None and isinstance(model, (torch.nn.Module, torch.nn.DataParallel)):
        if mode == 'auto':
            if orig_ctx[0] == torch.device('cpu'):
                mode = 'cpu'
            else:
                mode = 'gpu'
        if mode == 'cpu':
            return [torch.device('cpu')]
        if mode == 'gpu':
            return [torch.device(f'cuda:{gid}') for gid in range(torch.cuda.device_count())]
        if isinstance(mode, (list, tuple)):
            if not all(isinstance(i, int) for i in mode):
                raise ValueError('Requires integer gpu id, given {}'.format(mode))
            return [torch.device(f'cuda:{gid}') for gid in mode if gid in range(torch.cuda.device_count())]

    return None
