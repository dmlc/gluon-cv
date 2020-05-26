"""Filesystem utility functions."""
import os
import errno

def makedirs(path):
    """Create directory recursively if not exists.
    Similar to `makedir -p`, you can skip checking existence before this function.

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

def try_import(package, message=None):
    """Try import specified package, with custom message support.

    Parameters
    ----------
    package : str
        The name of the targeting package.
    message : str, default is None
        If not None, this function will raise customized error message when import error is found.


    Returns
    -------
    module if found, raise ImportError otherwise

    """
    try:
        return __import__(package)
    except ImportError as e:
        if not message:
            raise e
        raise ImportError(message)

def try_import_cv2():
    """Try import cv2 at runtime.

    Returns
    -------
    cv2 module if found. Raise ImportError otherwise

    """
    msg = "cv2 is required, you can install by package manager, e.g. 'apt-get', \
        or `pip install opencv-python --user` (note that this is unofficial PYPI package)."
    return try_import('cv2', msg)

def try_import_colorama():
    """Try import colorama at runtime.

    Returns
    -------
    colorama module if found. Raise ImportError otherwise

    """
    msg = "colorama is required, you can install by `pip install colorama --user` \
         (note that this is unofficial PYPI package)."
    return try_import('colorama', msg)

def try_import_decord():
    """Try import decord at runtime.

    Returns
    -------
    Decord module if found. Raise ImportError otherwise

    """
    msg = "Decord is required, you can install by `pip install decord --user` \
        (note that this is unofficial PYPI package)."
    return try_import('decord', msg)

def try_import_mmcv():
    """Try import mmcv at runtime.

    Returns
    -------
    mmcv module if found. Raise ImportError otherwise

    """
    msg = "mmcv is required, you can install by first `pip install Cython --user` \
        and then `pip install mmcv --user` (note that this is unofficial PYPI package)."
    return try_import('mmcv', msg)

def try_import_rarfile():
    """Try import rarfile at runtime.

    Returns
    -------
    rarfile module if found. Raise ImportError otherwise

    """
    msg = "rarfile is required, you can install by first `sudo apt-get install unrar` \
        and then `pip install rarfile --user` (note that this is unofficial PYPI package)."
    return try_import('rarfile', msg)

def import_try_install(package, extern_url=None):
    """Try import the specified package.
    If the package not installed, try use pip to install and import if success.

    Parameters
    ----------
    package : str
        The name of the package trying to import.
    extern_url : str or None, optional
        The external url if package is not hosted on PyPI.
        For example, you can install a package using:
         "pip install git+http://github.com/user/repo/tarball/master/egginfo=xxx".
        In this case, you can pass the url to the extern_url.

    Returns
    -------
    <class 'Module'>
        The imported python module.

    """
    import tempfile
    import portalocker
    lockfile = os.path.join(tempfile.gettempdir(), package + '_install.lck')
    with portalocker.Lock(lockfile):
        try:
            return __import__(package)
        except ImportError:
            try:
                from pip import main as pipmain
            except ImportError:
                from pip._internal import main as pipmain
                from types import ModuleType
                # fix for pip 19.3
                if isinstance(pipmain, ModuleType):
                    from pip._internal.main import main as pipmain

            # trying to install package
            url = package if extern_url is None else extern_url
            pipmain(['install', '--user', url])  # will raise SystemExit Error if fails

            # trying to load again
            try:
                return __import__(package)
            except ImportError:
                import sys
                import site
                user_site = site.getusersitepackages()
                if user_site not in sys.path:
                    sys.path.append(user_site)
                return __import__(package)
    return __import__(package)

def try_import_dali():
    """Try import NVIDIA DALI at runtime.
    """
    try:
        dali = __import__('nvidia.dali', fromlist=['pipeline', 'ops', 'types'])
        dali.Pipeline = dali.pipeline.Pipeline
    except (ImportError, RuntimeError) as e:
        if isinstance(e, ImportError):
            msg = "DALI not found, please check if you installed it correctly."
        elif isinstance(e, RuntimeError):
            msg = "No CUDA-capable device is detected ({}).".format(e)
        class dali:
            class Pipeline:
                def __init__(self):
                    raise NotImplementedError(msg)
    return dali

def try_import_html5lib():
    """Try import html5lib at runtime.

    Returns
    -------
    html5lib module if found. Raise ImportError otherwise

    """
    msg = "html5lib is required, you can install by package manager, " \
          "e.g. pip install html5lib --user` (note that this is unofficial PYPI package)."
    return try_import('html5lib', msg)

def try_import_gdfDownloader():
    """Try import googleDriveFileDownloader at runtime.

    Returns
    -------
    googleDriveFileDownloader module if found. Raise ImportError otherwise

    """
    msg = "googleDriveFileDownloader is required, you can install by package manager, " \
          "e.g. pip install googleDriveFileDownloader --user` " \
          "(note that this is unofficial PYPI package)."
    return try_import('googleDriveFileDownloader', msg)
