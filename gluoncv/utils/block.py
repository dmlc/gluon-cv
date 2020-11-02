"""Utility functions for gluon parameters."""
import re
import numpy as np

def recursive_visit(net, callback, **kwargs):
    """Recursively visit and apply callback to a net and its sub-net

    Parameters
    ----------
    net : mxnet.gluon.Block
        The network to recursively visit
    callback : function
        The callback function to apply to each net block.
        Its first argument needs to be the block
    """
    callback(net, **kwargs)
    for _, child in net._children.items():
        recursive_visit(child, callback, **kwargs)

def set_lr_mult(net, pattern, mult=1.0, verbose=False):
    """Reset lr_mult to new value for all parameters that match :obj:`pattern`

    Parameters
    ----------
    net : mxnet.gluon.Block
        The network whose parameters are going to be adjusted.
    pattern : str
        Regex matching pattern for targeting parameters.
    mult : float, default 1.0
        The new learning rate multiplier.
    verbose : bool
        Print which parameters being modified if set `True`.

    Returns
    -------
    mxnet.gluon.Block
        Original network with learning rate multipliers modified.

    """
    pattern = re.compile(pattern)
    for key, value in net.collect_params().items():
        if not re.match(pattern, key):
            continue
        value.lr_mult = mult
        if verbose:
            print("Set lr_mult of {} to {}".format(value.name, mult))

def _freeze_bn_callback(net, use_global_stats=True):
    from mxnet.gluon.nn import BatchNorm
    if isinstance(net, BatchNorm):
        net._kwargs['use_global_stats'] = use_global_stats

def freeze_bn(net, use_global_stats=True):
    """Freeze BatchNorm layers by setting `use_global_stats` to `True`

    Parameters
    ----------
    net : mxnet.gluon.Block
        The network whose BatchNorm layers are going to be modified
    use_global_stats : bool
        The value of `use_global_stats` to set for all BatchNorm layers

    Returns
    ------
    mxnet.gluon.Block
        Original network with BatchNorm layers modified.

    """
    recursive_visit(net, _freeze_bn_callback, use_global_stats=use_global_stats)

def purge_model_nan(net, nan=0.0, posinf=0.0, neginf=0.0, verbose=False):
    """Purge non infinite values in model parameters. GPU trained model may
    contain nan/inf/-inf values which is hidden since CUDNN may handle nan
    implicitly. This may cause model to produce nan during CPU inference.

    Weights will be overwritten inplace.

    Parameters
    ----------
    net : mxnet.gluon.Block
        The network whose weights will be purged to remove nan/inf/-inf.
    nan : float, default is 0.0
        Value to be used to fill NaN values.
        If no value is passed then NaN values will be replaced with 0.0.
    posinf : float, default is 0.0
        Value to be used to fill +Inf values.
        If no value is passed then +Inf values will be replaced with 0.0.
    neginf : float, default is 0.0
        Value to be used to fill -Inf values.
        If no value is passed then -Inf values will be replaced with 0.0.
    verbose : bool
        If True, will print out what parameters are modified.
    """
    for k, v in net.collect_params().items():
        np_data = v.data().asnumpy()
        if not np.isfinite(np_data).all():
            if verbose:
                print(k, ': Overwritten {} values...'.format(
                    np_data.size - np.isfinite(np_data).sum()))
            new_data = np.nan_to_num(np_data, nan=nan, posinf=posinf, neginf=neginf)
            v.set_data(new_data)
