"""Utility functions for version checking."""
import warnings

__all__ = ['check_version']

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
