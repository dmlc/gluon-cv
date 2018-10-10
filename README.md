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

# Supported Applications

| Application  | Illustration  | Note |
|:-----------------------:|:---:|:---:|
| [Image Classification](https://gluon-cv.mxnet.io/model_zoo/classification.html)  | <a href="https://gluon-cv.mxnet.io/model_zoo/classification.html"><img  src="docs/_static/image-classification.png" alt="classification" height="200"/></a>  | Classify an entire image |
| [Object Detection](https://gluon-cv.mxnet.io/model_zoo/detection.html)      |  <a href="https://gluon-cv.mxnet.io/model_zoo/detection.html"><img src="docs/_static/object-detection.png" alt="detection" height="200"/></a> | Detect objects of interest in complex scenes |
| [Semantic Segmentation](https://gluon-cv.mxnet.io/model_zoo/segmentation.html#semantic-segmentation) |  <a href="https://gluon-cv.mxnet.io/model_zoo/segmentation.html#semantic-segmentation"><img src="docs/_static/semantic-segmentation.png" alt="semantic" height="200"/></a> | Associate each pixel of an image with categorial labels |
| [Instance Segmentation](https://gluon-cv.mxnet.io/model_zoo/segmentation.html#instance-segmentation) | <a href="https://gluon-cv.mxnet.io/model_zoo/segmentation.html#instance-segmentation"><img src="docs/_static/instance-segmentation.png" alt="instance" height="200"/></a> | Detect and segment individual objects|
| [GAN](https://github.com/dmlc/gluon-cv/tree/master/scripts/gan) | <a href="https://github.com/dmlc/gluon-cv/tree/master/scripts/gan"><img src="https://github.com/dmlc/gluon-cv/raw/master/scripts/gan/wgan/fake_samples_400000.png" alt="lsun" height="200"/></a> | Generate visually deceptive images |
| [Person Re-ID](https://github.com/dmlc/gluon-cv/tree/master/scripts/re-id/baseline) | <a href="https://github.com/dmlc/gluon-cv/tree/master/scripts/re-id/baseline"><img src="https://user-images.githubusercontent.com/3307514/46702937-f4311800-cbd9-11e8-8eeb-c945ec5643fb.png" alt="re-id" height="160"/></a> | Re-identify pedestrians across scenes|

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

- [Generative Adversarial Network](https://github.com/dmlc/gluon-cv/tree/master/scripts/gan)

- [Person Re-identification](https://github.com/dmlc/gluon-cv/tree/master/scripts/re-id/)

# Resources

Check out how to use GluonCV for your own research or projects.

If you are new to Gluon, please check out [our 60-minute crash course](http://gluon-crash-course.mxnet.io/).

For getting started quickly, refer to notebook runnable examples at [Examples](https://gluon-cv.mxnet.io/build/examples_classification/index.html).

For advanced examples, check out our [Scripts](http://gluon-cv.mxnet.io/master/scripts/index.html).

For experienced users, check out our [API Notes](https://gluon-cv.mxnet.io/api/data.datasets.html#).
