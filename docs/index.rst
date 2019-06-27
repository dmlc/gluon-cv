GluonCV: a Deep Learning Toolkit for Computer Vision
========================================================

.. toctree::
    :maxdepth: 2
    :hidden:

    model_zoo/index
    tutorials/index
    api/index
    how_to/index
    slides

GluonCV provides implementations of state-of-the-art (SOTA) deep learning algorithms in computer vision. It aims to help engineers, researchers, and students quickly prototype products, validate new ideas and learn computer vision.

GluonCV features:

1. training scripts that reproduce SOTA results reported in latest papers,

2. a large set of pre-trained models,

3. carefully designed APIs and easy to understand implementations,

4. community support.

Demo
----

.. raw:: html

    <div align="center">
        <img src="_static/short_demo.gif">
    </div>

    <br>


Check the HD video at `Youtube <https://www.youtube.com/watch?v=nfpouVAzXt0>`_ or `Bilibili <https://www.bilibili.com/video/av55619231>`_.

Supported Applications
----------------------

.. raw:: html
   :file: applications.html

.. raw:: html

   <a id="installation.html"></a>

Installation
------------

Install MXNet
^^^^^^^^^^^^^

.. Ignore prerequisites to make the index page concise, which will be shown at
   the install page

.. raw:: html

   <style>.admonition-prerequisite {display: none;}</style>

.. include:: install/install-include.rst

Check :doc:`install/index` for more installation instructions and options.

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


New to Deep Learning or CV?
---------------------------

For background knowledge of deep learning or CV, please refer to the open source book `Dive into Deep Learning <http://en.diveintodeeplearning.org/>`_.

Adoptions
---------

Companies and organizations using GluonCV:

.. include:: /_static/logos/embed.html
