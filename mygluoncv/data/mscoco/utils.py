"""Import helper for pycocotools"""
# NOTE: for developers
# please do not import any pycocotools in __init__ because we are trying to lazy
# import pycocotools to avoid install it for other users who may not use it.
# only import when you actually use it
from __future__ import absolute_import

from ...utils.filesystem import import_try_install

def try_import_pycocotools():
    """Tricks to optionally install and import pycocotools"""
    # first we can try import pycocotools
    try:
        import pycocotools as _
    except ImportError:
        import os
        # we need to install pycootools, which is a bit tricky
        # pycocotools sdist requires Cython, numpy(already met)
        import_try_install('cython')
        # pypi pycocotools is not compatible with windows
        win_url = 'git+https://github.com/zhreshold/cocoapi.git#subdirectory=PythonAPI'
        try:
            if os.name == 'nt':
                import_try_install('pycocotools', win_url)
            else:
                import_try_install('pycocotools')
        except ImportError:
            faq = 'cocoapi FAQ'
            raise ImportError('Cannot import or install pycocotools, please refer to %s.' % faq)
