"""Synchronized Cross GPU Batch Normalization"""
# pylint: disable=arguments-differ
import threading

import mxnet as mx
from mxnet import gluon, autograd, test_utils

class SharedTensor(object):
    """Shared Tensor for Syncing"""
    def __init__(self, key, nchannels, num_devices):
        self._mutex = threading.Lock()
        self._all_tasks_done = threading.Condition(self._mutex)
        self._key = key
        self.num_devices = int(num_devices)
        self.out = mx.nd.zeros(nchannels)
        self._clear()

    def _clear(self):
        self.list = []
        self.push_tasks = self.num_devices
        self.reduce_tasks = self.num_devices

    def push(self, t):
        """push value to SharedTensor"""
        with self._mutex:
            if self.push_tasks == 0:
                self._clear()
            #t.wait_to_read()
            self.list.append(t)
            self.push_tasks -= 1
        with self._all_tasks_done:
            if self.push_tasks == 0:
                self._all_tasks_done.notify_all()
            while self.push_tasks:
                self._all_tasks_done.wait()

    def _reduce(self, kv):
        with self._mutex:
            if self.reduce_tasks == 1:
                assert(len(self.list) == self.num_devices)
                kv.push(self._key, self.list)
                self.reduce_tasks -= 1
            else:
                self.reduce_tasks -= 1
        with self._all_tasks_done:
            if self.reduce_tasks == 0:
                self._all_tasks_done.notify_all()
            while self.reduce_tasks:
                self._all_tasks_done.wait()

    def pull(self, kv):
        """Get value form SharedTensor"""
        self._reduce(kv)
        kv.pull(self._key, out=self.out)
        return self.out

    def __len__(self):
        return len(self.list)


class SharedTDict(object):
    """Shared Dict for Syncing"""
    def __init__(self):
        self.stdict = {}
        self.keys = []
        self._mutex = threading.Lock()
        self.kv = mx.kv.create('local')

    def register(self, key, nchannels, num_devices):
        with self._mutex:
            if key in self.keys:
                return
            print('registerring {}'.format(key))
            self.stdict[key] = SharedTensor(key, nchannels, num_devices)
            self.kv.init(key, mx.nd.zeros(nchannels))
            self.keys.append(key)

    def push(self, key, value):
        self.stdict[key].push(value)

    def pull(self, key):
        out = self.stdict[key].pull(self.kv)
        return out

sharedTensorDict = SharedTDict()

class AllReduce(autograd.Function):
    """All Reduce Operation"""
    def __init__(self, key):
        super(AllReduce, self).__init__()
        self.xsumkey = key + 'sum'
        self.xsqukey = key + 'squ'

    def forward(self, isum, isqu):
        """AllReduce forward.

        Parameters
        ----------
        isum : mxnet.nd.NDArray
            Sum array
        isqu : mxnet.nd.NDArray
            Squre arrays
        """
        sharedTensorDict.push(self.xsumkey, isum)
        sharedTensorDict.push(self.xsqukey, isqu)
        osum = sharedTensorDict.pull(self.xsumkey).as_in_context(isum.context)
        osqu = sharedTensorDict.pull(self.xsqukey).as_in_context(isqu.context)
        return osum, osqu

    def backward(self, dsum, dsqu):
        """AllReduce backward.

        Parameters
        ----------
        dsum : mxnet.nd.NDArray
            Gradient of Sum array
        dsqu : mxnet.nd.NDArray
            Gradient of Squre arrays
        """
        sharedTensorDict.push(self.xsumkey, dsum)
        sharedTensorDict.push(self.xsqukey, dsqu)
        disum = sharedTensorDict.pull(self.xsumkey).as_in_context(dsum.context)
        disqu = sharedTensorDict.pull(self.xsqukey).as_in_context(dsqu.context)
        return disum, disqu


class BatchNorm(gluon.nn.BatchNorm):
    """Cross-GPU Synchronized Batch normalization (SyncBN)
    Standard BN [1]_ implementation only normalize the data within each device.
    SyncBN normalizes the input within the whole mini-batch.
    We follow the sync-onece implmentation described in the paper [2]_ .

    Parameters
    ----------
    axis : int, default 1
        The axis that should be normalized. This is typically the channels
        (C) axis. For instance, after a `Conv2D` layer with `layout='NCHW'`,
        set `axis=1` in `BatchNorm`. If `layout='NHWC'`, then set `axis=3`.
    momentum: float, default 0.9
        Momentum for the moving average.
    epsilon: float, default 1e-5
        Small float added to variance to avoid dividing by zero.
    center: bool, default True
        If True, add offset of `beta` to normalized tensor.
        If False, `beta` is ignored.
    scale: bool, default True
        If True, multiply by `gamma`. If False, `gamma` is not used.
        When the next layer is linear (also e.g. `nn.relu`),
        this can be disabled since the scaling
        will be done by the next layer.
    use_global_stats: bool, default False
        If True, use global moving statistics instead of local batch-norm. This will force
        change batch-norm into a scale shift operator.
        If False, use local batch-norm.
    beta_initializer: str or `Initializer`, default 'zeros'
        Initializer for the beta weight.
    gamma_initializer: str or `Initializer`, default 'ones'
        Initializer for the gamma weight.
    moving_mean_initializer: str or `Initializer`, default 'zeros'
        Initializer for the moving mean.
    moving_variance_initializer: str or `Initializer`, default 'ones'
        Initializer for the moving variance.
    in_channels : int, default 0
        Number of channels (feature maps) in input data. If not specified,
        initialization will be deferred to the first time `forward` is called
        and `in_channels` will be inferred from the shape of input data.
    num_devices : int, default number of visible GPUs
    Inputs:
        - **data**: input tensor with arbitrary shape.
    Outputs:
        - **out**: output tensor with the same shape as `data`.


    Reference:

        .. [1] Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating
        deep network training by reducing internal covariate shift." *ICML 2015*

        .. [2] Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang,
        Ambrish Tyagi, and Amit Agrawal. "Context Encoding for Semantic Segmentation." *CVPR 2018*
    """
    # pylint: disable=arguments-differ
    def __init__(self, in_channels, axis=1, momentum=0.9, epsilon=1e-5, ndevices=None, **kwargs):
        super(BatchNorm, self).__init__(axis, momentum, epsilon, in_channels=in_channels, **kwargs)

        self.eps = epsilon
        self.momentum = momentum
        self.in_channels = in_channels
        self.ndevices = self._get_num_devices() if ndevices is None else ndevices
        self.updater = _SharedUpdater(self.ndevices)
        sharedTensorDict.register(self._prefix + 'sum', in_channels, self.ndevices)
        sharedTensorDict.register(self._prefix + 'squ', in_channels, self.ndevices)

    def _get_num_devices(self):
        # caution: if not using all the GPUs, please mannually set num_devices
        num_devices = len(test_utils.list_gpus())
        # for CPU
        num_devices = num_devices if num_devices > 0 else 1
        return num_devices

    def hybrid_forward(self, F, x, gamma, beta, running_mean, running_var):
        """Hybrid forward"""
        if not autograd.is_training():
            return F.BatchNorm(x, gamma, beta, running_mean, running_var, name='fwd',
                               **self._kwargs)
        isum, isqu = F.SumSquare(x)
        #isum = x.sum(axis=1, exclude=True)
        #isqu = (x**2).sum(axis=1, exclude=True)
        N = self.ndevices * x.shape[0] * x.shape[2] * x.shape[3]
        allreduce = AllReduce(self._prefix)
        osum, osqu = allreduce(isum, isqu)
        # calc mean and std
        mean = osum / N
        sumvar = osqu - osum * osum / N
        bias_var = sumvar / N
        std = F.sqrt(F.maximum(bias_var, self.eps))
        # update running mean and var
        with autograd.pause():
            unbias_var = sumvar / (N - 1)
            self.updater(self.running_mean, self.running_var, mean, unbias_var,
                         self.momentum, x.context)
        # update running mean and var
        output = F.DecoupleBatchNorm(x, gamma, beta, mean, std)
        return output


class _SharedUpdater(object):
    # update only once
    def __init__(self, num_devices):
        self._mutex = threading.Lock()
        self.num_devices = num_devices
        self._clear()

    def _clear(self):
        self.tasks = self.num_devices

    def __call__(self, running_mean, running_var, mean, unbias_var, momentum, ctx):
        with self._mutex:
            if self.tasks == self.num_devices:
                running_mean.set_data(momentum * running_mean.data(ctx) + \
                    (1.0 - momentum) * mean)
                running_var.set_data(momentum * running_var.data(ctx) + \
                    (1.0 - momentum) * unbias_var)
            self.tasks -= 1
        if self.tasks == 0:
            self._clear()
