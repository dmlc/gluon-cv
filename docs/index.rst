GluonVision: a Deep Learning Toolkit for Computer Vision
========================================================

GluonVision provides implementations of the sate-of-the-art (SOTA) deep learning
models in computer vision. It is designed for engineers, researchers, and
students to fast prototype products and research ideas based on these
models. This toolkit offers four main features:

1. Training scripts to reproduce SOTA results reported in research papers
2. A large number of pre-trained models
3. Carefully designed APIs that greatly reduce the implementation complexity
4. Community supports

This toolkit assume users has basic knowledges about deep learning and computer
vision. Otherwise, please refer to introduction courses such as `Stanford
CS231n <http://cs231n.stanford.edu/>`_.

.. note::

   This project is still at an early stage. Please expect that it will
   be updated frequently. We also welcome any contributions.

Installation
------------------

Install via PyPI
^^^^^^^^^^^^^^^^

The easiest way to install GluonVision is through `pip <https://pip.pypa.io/en/stable/installing/>`_.

.. code-block:: bash

  pip install gluonvision

Install from Source
^^^^^^^^^^^^^^^^^^^

Optionally you can clone the GluonVision project and install it locally

.. code-block:: bash

   git clone https://github.com/dmlc/gluon-vision
   cd gluon-vision && python setup.py install --user

Install MXNet
^^^^^^^^^^^^^

GluonVision relies on the recent version of MXNet. The easiest way to install MXNet
is through `pip <https://pip.pypa.io/en/stable/installing/>`_. The following
command installs a nightly build CPU version of MXNet.

.. code-block:: bash

   pip install --pre mxnet

.. note::

   There are other pre-build MXNet packages that enables GPU supports and
   accelerate CPU performance, please refer to `this tutorial
   <http://gluon-crash-course.mxnet.io/mxnet_packages.html>`_ for details. Some
   training scripts are recommended to run on GPUs, if you don't have a GPU
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
