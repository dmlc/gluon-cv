"""2. GluonCV C++ Inference Demo
================================

This is a demo tutorial which illustrates how to use existing GluonCV
models in c++ environments given exported JSON and PARAMS files.

Please checkout :ref:`Export Network <sphx_glr_build_examples_deployment_export_network.py>` for instructions of how to export pre-trained models.

Demo usage
----------

.. code:: bash

   # with yolo3_darknet53_voc-0000.params and yolo3_darknet53-symbol.json on disk
   ./gluoncv-detect yolo3_darknet53_voc demo.jpg

.. figure:: https://user-images.githubusercontent.com/3307514/45458507-d76ff600-b6a8-11e8-92e1-0b1966e4344f.jpg
   :alt: demo

   demo

Usage:

::

   SYNOPSIS
           ./gluoncv-detect <model file> <image file> [-o <outfile>] [--class-file <classfile>] [-e
                            <epoch>] [--gpu <gpu>] [-q] [--no-disp] [-t <thresh>]

   OPTIONS
           -o, --output <outfile>
                       output image, by default no output

           --class-file <classfile>
                       plain text file for class names, one name per line

           -e, --epoch <epoch>
                       Epoch number to load parameters, by default is 0

           --gpu <gpu> Which gpu to use, by default is -1, means cpu only.
           -q, --quite Quite mode, no screen output
           --no-disp   Do not display image
           -t, --thresh <thresh>
                       Visualize threshold, from 0 to 1, default 0.3.

Source Code and Build Instructions
----------------------------------

The C++ demo code and build instructions are available in our repository `scripts
<https://github.com/dmlc/gluon-cv/tree/master/scripts/deployment/cpp-inference>`_.

.. hint::

    Prebuilt binaries are `available <https://github.com/dmlc/gluon-cv/tree/master/scripts/deployment/cpp-inference#download-prebuilt-binaries>`_ for Linus, Mac OS and Windows.

    And you can also build MXNet from `source <https://github.com/dmlc/gluon-cv/tree/master/scripts/deployment/cpp-inference#build-from-source>`_ to support C++ inference demo.
"""
