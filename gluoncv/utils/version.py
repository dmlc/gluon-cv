"""Utility functions for version checking."""
import sys
import warnings

__all__ = ['check_version', '_require_mxnet_version', '_deprecate_python2', '_mxnet_v2_legacy_hook']

def check_version(min_version, warning_only=False):
    """Check the version of gluoncv satisfies the provided minimum version.
    An exception is thrown if the check does not pass.

    Parameters
    ----------
    min_version : str
        Minimum version
    warning_only : bool
        Printing a warning instead of throwing an exception.
    """
    from .. import __version__
    from distutils.version import LooseVersion
    bad_version = LooseVersion(__version__) < LooseVersion(min_version)
    if bad_version:
        msg = 'Installed GluonCV version (%s) does not satisfy the ' \
              'minimum required version (%s)'%(__version__, min_version)
        if warning_only:
            warnings.warn(msg)
        else:
            raise AssertionError(msg)


def _require_mxnet_version(mx_version):
    try:
        import mxnet as mx
        from distutils.version import LooseVersion
        if LooseVersion(mx.__version__) < LooseVersion(mx_version):
            msg = (
                "Legacy mxnet-mkl=={} detected, some new modules may not work properly. "
                "mxnet-mkl>={} is required. You can use pip to upgrade mxnet "
                "`pip install -U --pre mxnet -f https://dist.mxnet.io/python/mkl` "
                "or `pip install -U --pre mxnet -f https://dist.mxnet.io/python/cu100mkl`\
                ").format(mx.__version__, mx_version)
            raise ImportError(msg)
    except ImportError:
        raise ImportError(
            "Unable to import dependency mxnet. "
            "A quick tip is to install via "
            "`pip install --pre mxnet -f https://dist.mxnet.io/python/cu100mkl`. "
            "please refer to https://gluon-cv.mxnet.io/#installation for details.")

def _deprecate_python2():
    if sys.version_info[0] < 3:
        msg = 'Python2 has reached the end of its life on January 1st, 2020. ' + \
            'A future version of gluoncv will drop support for Python 2.'
        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn(msg, DeprecationWarning)

def _mxnet_v2_legacy_hook():
    import mxnet
    try:
        from mxnet import metric
    except ImportError:
        mxnet.metric = mxnet.gluon.metric
