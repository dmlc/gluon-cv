"""Utility functions for version checking."""
import sys
import warnings

__all__ = ['check_version', '_require_mxnet_version', '_deprecate_python2']

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


def _require_mxnet_version(mx_version, max_mx_version='2.0.0'):
    try:
        import mxnet as mx
        from distutils.version import LooseVersion
        if LooseVersion(mx.__version__) < LooseVersion(mx_version) or \
            LooseVersion(mx.__version__) >= LooseVersion(max_mx_version):
            version_str = '>={},<{}'.format(mx_version, max_mx_version)
            msg = (
                "Legacy mxnet-mkl=={0} detected, some modules may not work properly. "
                "mxnet-mkl{1} is required. You can use pip to upgrade mxnet "
                "`pip install -U 'mxnet-mkl{1}'` "
                "or `pip install -U 'mxnet-cu100mkl{1}'`\
                ").format(mx.__version__, version_str)
            raise RuntimeError(msg)
    except ImportError:
        raise ImportError(
            "Unable to import dependency mxnet. "
            "A quick tip is to install via "
            "`pip install 'mxnet-cu100mkl<{}'`. "
            "please refer to https://gluon-cv.mxnet.io/#installation for details.".format(
                max_mx_version))

def _deprecate_python2():
    if sys.version_info[0] < 3:
        msg = 'Python2 has reached the end of its life on January 1st, 2020. ' + \
            'GluonCV has now dropped support for Python 2.'
        raise DeprecationWarning(msg)
