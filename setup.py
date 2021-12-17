#!/usr/bin/env python
import io
import os
import re
import sys
try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = None
from setuptools import setup, find_packages, Extension
try:
    import cv2
except ImportError:
    cv2 = None

with_cython = False
if '--with-cython' in sys.argv:
    if not cythonize:
        print("Cython not found, please run `pip install Cython`")
        exit(1)
    with_cython = True
    sys.argv.remove('--with-cython')


def read(*names, **kwargs):
    with io.open(
            os.path.join(os.path.dirname(__file__), *names),
            encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


try:
    import pypandoc

    long_description = pypandoc.convert('README.md', 'rst')
except(IOError, ImportError):
    long_description = open('README.md').read()

VERSION = find_version('gluoncv', '__init__.py')

requirements = [
    'numpy',
    'tqdm',
    'requests',
    # 'mxnet',
    'matplotlib',
    'portalocker',
    'Pillow',
    'scipy',
    #'tensorboardx',
    #'decord',
    #'opencv-python',
    'yacs',
    'pandas',
    'pyyaml',
    'autocfg',
    #'autogluon.core'
]

# do not duplicate opencv module if already compiled from source
if cv2 is None:
    requirements.append('opencv-python')

extra_requirements = {
    'full': ['tensorboardx', 'decord', 'autogluon.core', 'cython', 'pycocotools'],
    'auto': ['autogluon.core==0.3.1']
}

if with_cython:
    import numpy as np
    _NP_INCLUDE_DIRS = np.get_include()

    # Extension modules
    ext_modules = cythonize([
        Extension(
            name='gluoncv.nn.cython_bbox',
            sources=['gluoncv/nn/cython_bbox.pyx'],
            extra_compile_args=['-Wno-cpp', '-O3'],
            include_dirs=[_NP_INCLUDE_DIRS]),
        Extension(
            name='gluoncv.model_zoo.rcnn.rpn.cython_rpn_target',
            sources=['gluoncv/model_zoo/rcnn/rpn/cython_rpn_target.pyx'],
            extra_compile_args=['-Wno-cpp', '-O3'],
            include_dirs=[_NP_INCLUDE_DIRS]),
    ])
else:
    ext_modules = []


setup(
    # Metadata
    name='gluoncv',
    version=VERSION,
    author='Gluon CV Toolkit Contributors',
    url='https://github.com/dmlc/gluon-cv',
    description='MXNet Gluon CV Toolkit',
    long_description=long_description,
    license='Apache-2.0',

    # Package info
    packages=find_packages(exclude=('docs', 'tests', 'scripts')),
    zip_safe=True,
    include_package_data=True,
    install_requires=requirements,
    ext_modules=ext_modules,
    extras_require=extra_requirements
)
