GluonVision: a Deep Learning Toolkit for Computer Vision
========================================================

GluonVision provides implementations of state-of-the-art (SOTA) deep learning
algorithms in computer vision.
It is designed for helping engineers, researchers, and students to quickly
prototype products, validate new ideas, and learning computer vision.

GluonVision features:

1. training scripts that reproduces SOTA results reported in latest papers,

2. a large set of pre-trained models,

3. carefully designed APIs and easy to understand implementations,

4. community support.

GluonVision tutorials assume users have basic knowledges about deep learning and
computer vision.
Otherwise, please refer to our introductory deep learning course
`MXNet-the-Streight-Dope <http://gluon.mxnet.io/>`_.

.. note::

   This project is at an early stage. Please expect frequent updates.
   We welcome feedback and contributions.

Installation
------------------

Install via PyPI
^^^^^^^^^^^^^^^^

The easiest way to install GluonVision is through `pip <https://pip.pypa.io/en/stable/installing/>`_.

.. code-block:: bash

  pip install gluonvision

Install from Source
^^^^^^^^^^^^^^^^^^^

Optionally, you can clone the GluonVision project and install it locally

.. code-block:: bash

   git clone https://github.com/dmlc/gluon-vision
   cd gluon-vision && python setup.py install --user

Install MXNet
^^^^^^^^^^^^^

GluonVision depends on the recent version of MXNet. The easiest way to install MXNet
is through `pip <https://pip.pypa.io/en/stable/installing/>`_. The following
command installs a nightly build CPU version of MXNet.

.. code-block:: bash

   pip install --pre mxnet

.. note::

   There are other pre-build MXNet binaries that enable GPU support and
   accelerate CPU performance, please refer to `this tutorial
   <http://gluon-crash-course.mxnet.io/mxnet_packages.html>`_ for details.

   Some training scripts are recommended to run on GPUs, if you don't have a GPU
   machine at hands, you may consider to `run on AWS
   <http://gluon-crash-course.mxnet.io/use_aws.html>`_.


A Quick Example
----------------


:ref:`sphx_glr_build_examples_detection_demo_ssd.py`



.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   build/examples_classification/index
   build/examples_detection/index
   build/examples_segmentation/index
   build/examples_datasets/index

.. toctree::
   :maxdepth: 1
   :caption: Model Zoo

   model_zoo/index

.. toctree::
   :maxdepth: 1
   :caption: API Reference


   api/datasets
   api/transforms
   api/models
   api/utils

.. toctree::
   :maxdepth: 1
   :caption: Community

   how_to/support
   how_to/contribute
