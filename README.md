# Gluon CV Toolkit

[![Build Status](http://ci.mxnet.io/job/gluon-cv/job/master/badge/icon)](http://ci.mxnet.io/job/gluon-cv/job/master/)
[![GitHub license](http://dmlc.github.io/img/apache2.svg)](./LICENSE)
[![Code Coverage](http://gluon-cv.mxnet.io/coverage.svg?)](http://gluon-cv.mxnet.io/coverage.svg)
[![PyPI](https://img.shields.io/pypi/v/gluoncv.svg)](https://pypi.python.org/pypi/gluoncv)
[![PyPI Pre-release](https://img.shields.io/badge/pypi--prerelease-v0.3.0-ff69b4.svg)](https://pypi.python.org/pypi/gluoncv)
[![Downloads](http://pepy.tech/badge/gluoncv)](http://pepy.tech/project/gluoncv)

| [Installation](http://gluon-cv.mxnet.io) | [Documentation](http://gluon-cv.mxnet.io) | [Tutorials](http://gluon-cv.mxnet.io) |

GluonCV provides implementations of the state-of-the-art (SOTA) deep learning models in computer vision.

It is designed for engineers, researchers, and
students to fast prototype products and research ideas based on these
models. This toolkit offers four main features:

1. Training scripts to reproduce SOTA results reported in research papers
2. A large number of pre-trained models
3. Carefully designed APIs that greatly reduce the implementation complexity
4. Community supports

# Installation

GluonCV supports Python 2.7/3.5 or later. The easiest way to install is via pip, the following command installs the latest nightly build of GluonCV and MXNet:

```bash
pip install gluoncv --pre
pip install mxnet --pre --upgrade
# if cuda 9.2 is installed
pip install mxnet-cu92 --pre --upgrade
```

There are multiple versions of MXNet pre-built package available. Please refer to [mxnet packages](https://gluon-crash-course.mxnet.io/mxnet_packages.html) if you need more details about MXNet versions.

# Docs 📖
GluonCV documentation is available at [our website](https://gluon-cv.mxnet.io/index.html).

# Examples

All tutorials are available at [our website](https://gluon-cv.mxnet.io/index.html)!

- [Image Classification](http://gluon-cv.mxnet.io/build/examples_classification/index.html)

- [Object Detection](http://gluon-cv.mxnet.io/build/examples_detection/index.html)

- [Semantic Segmentation](http://gluon-cv.mxnet.io/build/examples_segmentation/index.html)

- [Instance Segmentation](http://gluon-cv.mxnet.io/build/examples_instance/index.html)

# Resources

Check out how to use GluonCV for your own research or projects.

If you are new to Gluon, please check out [our 60-minute crash course](http://gluon-crash-course.mxnet.io/).

For getting started quickly, refer to notebook runnable examples at [Examples](https://gluon-cv.mxnet.io/build/examples_classification/index.html).

For advanced examples, check out our [Scripts](http://gluon-cv.mxnet.io/master/scripts/index.html).

For experienced users, check out our [API Notes](https://gluon-cv.mxnet.io/api/data.datasets.html#).
