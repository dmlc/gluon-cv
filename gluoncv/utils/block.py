"""Utility functions for gluon parameters."""
import re


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
        Print which parameters being modifed if set `True`.

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
