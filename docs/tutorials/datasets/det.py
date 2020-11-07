"""Prepare ILSVRC 2015 DET dataset
=========================================

`ILSVRC DET dataset <http://image-net.org/>`_ is
a dataset for object localization/detection and image/scene classification task.
It contains two hundred full labeled categories for detection and 1000 categories for Object localization

This tutorial helps you to download ILSVRC DET and set it up for later experiments.

.. hint::

   You need 121G free disk space to download and extract this dataset.
   SSD harddrives are recommended for faster speed.
   The time it takes to prepare the dataset depends on your Internet connection
   and disk speed. 

Prepare the dataset
-------------------

We will download and unzip the following files:

+-------------------------------------------------------------------------------------------------------+--------+
| File name                                                                                             | Size   |
+=======================================================================================================+========+
| `ILSVRC2015_DET.tar.gz <http://image-net.org/image/ILSVRC2015/ILSVRC2015_DET.tar.gz>`_   |  49 G  |
+-------------------------------------------------------------------------------------------------------+--------+


we suggest run the command because it included download dataset and data processing，
the easiest way is to run this script:

 :download:`Download script: ilsvrc_det.py<../../../scripts/datasets/ilsvrc_det.py>`

.. code-block:: bash

   python ilsvrc_det.py

"""
