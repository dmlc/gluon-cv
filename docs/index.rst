GluonVision: a Deep Learning Toolkit for Computer Vision
========================================================

GluonVision provides implementations of the sate-of-the-art (SOTA) deep learning
models in computer vision. It is designed for engineers, researchers, and
students to fast prototype products and research ideas based on these
models. This toolkit offers four main features:

1. Training scripts to reproduce SOTA results reported in [research papers](link to the
paper to code list)
2. A large number of [pre-trained models](link to model zoo)
3. Carefully designed [APIs](link to API doc) that greatly reduce the
implementation complexity
4. [Community supports](link to how to contact us)

But also note that this project is still at an early stage. We are continuously
adding more models and making it easier to use. Any contribution (link to how to
contribute) is warmly welcome.

Quick Installation
------------------

GluonVision relies on the recent version of MXNet. The easiest way to install
MXNet is through pip(lin). The following command installs a nightly build CPU
version of MXNet.

.. code-block:: bash

   pip install --pre mxnet

Then clone the GluonVision project and install it locally

.. code-block:: bash

   git clone https://github.com/dmlc/gluon-vision
   cd gluon-vision && python setup.py install --user

A Quick Example
----------------

(get a pretrained SSD model and then run forward)



.. module:: gluonvision

.. toctree::
   :maxdepth: 1
   :caption: Get Started

   get_started/install
   tutorials/index
   experiments/index
   how_to/contribute
