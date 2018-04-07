"""Filesystem utility functions."""
import os
import errno

def makedirs(path):
    """Create directory recursively if not exists.

    Parameters
    ----------
    path : str
        Path of the desired dir
    """
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
