GluonCV: a Deep Learning Toolkit for Computer Vision
========================================================

GluonCV provides implementations of state-of-the-art (SOTA) deep learning algorithms in computer vision. It aims to help engineers, researchers, and students quickly prototype products, validate new ideas and learn computer vision.

GluonCV features:

1. training scripts that reproduce SOTA results reported in latest papers,

2. a large set of pre-trained models,

3. carefully designed APIs and easy to understand implementations,

4. community support.

Supported Applications
----------------------

.. raw:: html
   :file: applications.html

Installation
------------------

Install MXNet
^^^^^^^^^^^^^

GluonCV depends on the recent version of MXNet. The easiest way to install MXNet
is through `pip <https://pip.pypa.io/en/stable/installing/>`_. The following
command installs CPU version of MXNet.

.. code-block:: bash

   # the oldest stable version of mxnet required is 1.3.0
   pip install mxnet>=1.3.0 --upgrade

   # you can install nightly build of mxnet to access up-to-date features
   pip install --pre --upgrade mxnet

.. note::

   There are other pre-build MXNet binaries that enable GPU support and
   accelerate CPU performance, please refer to `this tutorial
   <http://gluon-crash-course.mxnet.io/mxnet_packages.html>`_ for details.

   Some training scripts are recommended to run on GPUs, if you don't have a GPU
   machine at hands, you may consider to `run on AWS
   <http://gluon-crash-course.mxnet.io/use_aws.html>`_.

Install GluonCV
^^^^^^^^^^^^^^^^

The easiest way to install GluonCV is through `pip <https://pip.pypa.io/en/stable/installing/>`_.

.. code-block:: bash

 pip install gluoncv --upgrade

 # if you are eager to try new features, try nightly build instead

 pip install gluoncv --pre --upgrade

.. hint::

  Nightly build is updated daily around 12am UTC to match master progress.

  Optionally, you can clone the GluonCV project and install it locally

  .. code-block:: bash

    git clone https://github.com/dmlc/gluon-cv
    cd gluon-cv && python setup.py install --user


A Quick Example
----------------


:ref:`Object Detection Demo <sphx_glr_build_examples_detection_demo_ssd.py>`


.. toctree::
   :maxdepth: 2
   :caption: Model Zoo

   model_zoo/classification
   model_zoo/detection
   model_zoo/segmentation

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   build/examples_classification/index
   build/examples_detection/index
   build/examples_instance/index
   build/examples_segmentation/index
   build/examples_datasets/index
   build/examples_deployment/index

.. toctree::
   :maxdepth: 1
   :caption: API Reference


   api/data.datasets
   api/data.batchify
   api/data.transforms
   api/model_zoo
   api/nn
   api/loss
   api/utils

.. toctree::
   :maxdepth: 1
   :caption: Community

   how_to/support
   how_to/contribute
