"""Filesystem utility functions."""
import os
import errno
import contextlib
from pathlib import Path
import tarfile
import zipfile

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

def try_import(package, message=None, fromlist=None):
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
        return __import__(package, fromlist=fromlist)
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

def try_import_munkres():
    """Try import munkres at runtime.

    Returns
    -------
    munkres module if found. Raise ImportError otherwise
    Munkres (Hungarian) algorithm for the Assignment Problem

    """
    msg = "munkres is required, you can install by `pip install munkres --user`. "
    return try_import('munkres', msg)

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

def unzip(zip_file_path, root='./', strict=False):
    """Unzips files located at `zip_file_path` into parent directory specified by `root`.
    """
    root = os.path.expanduser(root)
    with zipfile.ZipFile(zip_file_path) as zf:
        if strict or not os.path.exists(os.path.join(root, zf.namelist()[-1])):
            zf.extractall(root)
        folder = os.path.commonprefix(zf.namelist())
    return os.path.join(root, folder)

def untar(tar_file_path, root='./', strict=False):
    """Untars files located at `tar_file_path` into parent directory specified by `root`.
    """
    root = os.path.expanduser(root)
    with tarfile.open(tar_file_path, 'r:gz') as zf:
        if strict or not os.path.exists(os.path.join(root, zf.getnames()[-1])):
            zf.extractall(root)
        folder = os.path.commonprefix(zf.getnames())
    return os.path.join(root, folder)

@contextlib.contextmanager
def temporary_filename(suffix=None):
    """Context that introduces a temporary file.

    Creates a temporary file, yields its name, and upon context exit, deletes it.
    (In contrast, tempfile.NamedTemporaryFile() provides a 'file' object and
    deletes the file as soon as that file object is closed, so the temporary file
    cannot be safely re-opened by another library or process.)

    Parameters
    ----------
    suffix: desired filename extension (e.g. '.mp4').

    Yields
    ----------
    The name of the temporary file.
    """
    import tempfile
    try:
        f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp_name = f.name
        f.close()
        yield tmp_name
    finally:
        os.unlink(tmp_name)

class _DisplayablePath:
    """A util class for displaying the tree structure of root path.

    Example:

    >>> paths = _DisplayablePath.make_tree(Path('doc'))
    >>> for path in paths:
    >>>    print(path.displayable())

    Parameters
    ----------
    path : str
        The path.
    parent_path : str
        The parent parth.
    is_last : bool
        Whether it's the last node in this depth.

    """
    display_filename_prefix_middle = '├──'
    display_filename_prefix_last = '└──'
    display_parent_prefix_middle = '    '
    display_parent_prefix_last = '│   '

    def __init__(self, path, parent_path, is_last):
        self.path = Path(str(path))
        self.parent = parent_path
        self.is_last = is_last
        if self.parent:
            self.depth = self.parent.depth + 1
        else:
            self.depth = 0

    # pylint: disable=inconsistent-return-statements
    @classmethod
    def make_tree(cls, root, parent=None, is_last=False, criteria=None, max_depth=1):
        """Make tree structure from root.

        Parameters
        ----------
        root : str
            The root dir.
        parent : _DisplayablePath
            The parent displayable path.
        is_last : bool
            Whether it's the last in this level.
        criteria : function
            The criteria used to filter dir/path.
        max_depth : int
            Maximum depth for search.

        """
        root = Path(str(root))
        criteria = criteria or cls._default_criteria

        displayable_root = cls(root, parent, is_last)
        if displayable_root.depth > max_depth:
            return displayable_root
        yield displayable_root

        children = sorted(list(path
                               for path in root.iterdir()
                               if criteria(path)),
                          key=lambda s: str(s).lower())
        count = 1
        for path in children:
            is_last = count == len(children)
            if path.is_dir() and displayable_root.depth < max_depth - 1:
                yield from cls.make_tree(path,
                                         parent=displayable_root,
                                         is_last=is_last,
                                         criteria=criteria,
                                         max_depth=max_depth)
            else:
                yield cls(path, displayable_root, is_last)
            count += 1

    @classmethod
    def _default_criteria(cls, path):
        _ = path
        return True

    @property
    def displayname(self):
        if self.path.is_dir():
            return self.path.name + '/'
        return self.path.name

    def displayable(self):
        """Display string"""
        if self.parent is None:
            return self.displayname

        _filename_prefix = (self.display_filename_prefix_last
                            if self.is_last
                            else self.display_filename_prefix_middle)

        parts = ['{!s} {!s}'.format(_filename_prefix,
                                    self.displayname)]

        parent = self.parent
        while parent and parent.parent is not None:
            parts.append(self.display_parent_prefix_middle
                         if parent.is_last
                         else self.display_parent_prefix_last)
            parent = parent.parent

        return ''.join(reversed(parts))


class PathTree:
    """A directory tree structure viewer.

    Parameters
    ----------
    root : str or pathlib.Path
        The root directory.
    max_depth : int
        Max depth for recursive sub-folders, please be conservative to not spam the filesystem.

    """
    def __init__(self, root, max_depth=1):
        self._disp_path = _DisplayablePath.make_tree(Path(root), max_depth=max_depth)

    def __str__(self):
        s = '\n'.join([p.displayable() for p in self._disp_path])
        return s

def try_import_skimage():
    """Try import scikit-image at runtime.

    Returns
    -------
    scikit-image module if found. Raise ImportError otherwise

    """
    msg = "skimage is required, you can install by package manager, e.g. " \
          "`pip install scikit-image --user` (note that this is unofficial PYPI package)."
    return try_import('skimage', msg)
