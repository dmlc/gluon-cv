"""Pre init checks"""
import sys

def _require_mxnet_version(mx_version, max_mx_version='2.0.0'):
    try:
        import mxnet as mx
        from distutils.version import LooseVersion
        if LooseVersion(mx.__version__) < LooseVersion(mx_version) or \
            LooseVersion(mx.__version__) >= LooseVersion(max_mx_version):
            version_str = '>={},<{}'.format(mx_version, max_mx_version)
            msg = (
                "Legacy mxnet=={0} detected, some modules may not work properly. "
                "mxnet{1} is required. You can use pip to upgrade mxnet "
                "`pip install -U 'mxnet{1}'` "
                "or `pip install -U 'mxnet-cu100{1}'`\
                ").format(mx.__version__, version_str)
            raise RuntimeError(msg)
    except ImportError:
        raise ImportError(
            "Unable to import dependency mxnet. "
            "A quick tip is to install via "
            "`pip install 'mxnet-cu100<{}'`. "
            "please refer to https://gluon-cv.mxnet.io/#installation for details.".format(
                max_mx_version))

def _require_pytorch_version(torch_version, max_torch_version='2.0.0'):
    try:
        import torch
        from distutils.version import LooseVersion
        if LooseVersion(torch.__version__) < LooseVersion(torch_version) or \
            LooseVersion(torch.__version__) >= LooseVersion(max_torch_version):
            version_str = '>={},<{}'.format(torch_version, max_torch_version)
            msg = (
                "Legacy torch=={0} detected, some modules may not work properly. "
                "torch{1} is required. You can use pip or conda to upgrade".format(torch.__version__, version_str))
            raise RuntimeError(msg)
    except ImportError:
        raise ImportError(
            "Unable to import dependency pytorch. Please use pip or conda to install.")

def _deprecate_python2():
    if sys.version_info[0] < 3:
        msg = 'Python2 has reached the end of its life on January 1st, 2020. ' + \
            'GluonCV has now dropped support for Python 2.'
        raise DeprecationWarning(msg)
