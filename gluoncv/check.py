"""Pre init checks"""
import sys
import warnings


def _deprecate_python2():
    if sys.version_info[0] < 3:
        msg = 'Python2 has reached the end of its life on January 1st, 2020. ' + \
            'GluonCV has now dropped support for Python 2.'
        raise DeprecationWarning(msg)


def _require_mxnet_version(mx_version='1.4.0', max_mx_version='2.0.0'):
    from .backend import BackendManager, Backend
    
    manager = BackendManager()
    
    if not manager.is_available(Backend.MXNET):
        raise ImportError(
            "Unable to import dependency mxnet. "
            "A quick tip is to install via "
            "`pip install 'mxnet-cu100<{}'`. "
            "please refer to https://gluon-cv.mxnet.io/#installation for details.".format(
                max_mx_version))
    
    from distutils.version import LooseVersion
    current_version = manager.mxnet_version
    
    if LooseVersion(current_version) < LooseVersion(mx_version) or \
       LooseVersion(current_version) >= LooseVersion(max_mx_version):
        version_str = '>={},<{}'.format(mx_version, max_mx_version)
        msg = (
            "Legacy mxnet=={0} detected, some modules may not work properly. "
            "mxnet{1} is required. You can use pip to upgrade mxnet "
            "`pip install -U 'mxnet{1}'` "
            "or `pip install -U 'mxnet-cu100{1}'`"
        ).format(current_version, version_str)
        raise RuntimeError(msg)


def _require_pytorch_version(torch_version='1.4.0', max_torch_version='2.0.0'):
    from .backend import BackendManager, Backend
    
    manager = BackendManager()
    
    if not manager.is_available(Backend.PYTORCH):
        raise ImportError(
            "Unable to import dependency pytorch. Please use pip or conda to install.")
    
    from distutils.version import LooseVersion
    current_version = manager.pytorch_version
    
    if LooseVersion(current_version) < LooseVersion(torch_version) or \
       LooseVersion(current_version) >= LooseVersion(max_torch_version):
        version_str = '>={},<{}'.format(torch_version, max_torch_version)
        msg = (
            "Legacy torch=={0} detected, some modules may not work properly. "
            "torch{1} is required. You can use pip or conda to upgrade"
        ).format(current_version, version_str)
        raise RuntimeError(msg)


def check_mxnet_version():
    from .backend import BackendManager, Backend
    
    manager = BackendManager()
    if manager.is_available(Backend.MXNET):
        try:
            _require_mxnet_version()
            return True
        except (ImportError, RuntimeError):
            return False
    return False


def check_pytorch_version():
    from .backend import BackendManager, Backend
    
    manager = BackendManager()
    if manager.is_available(Backend.PYTORCH):
        try:
            _require_pytorch_version()
            return True
        except (ImportError, RuntimeError):
            return False
    return False
