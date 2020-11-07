"""Prepare ILSVRC 2015 VId dataset
=========================================

`ILSVRC  dataset <http://image-net.org/>`is Object detection from video
There are a total of 3862 snippets for training.
The number of snippets for each synest(category)ranges from 56 to 458
There are 555 validation snippets and 937 test snippets.

This tutorial helps you to download ILSVRC VID and set it up for later experiments.

.. hint::

   You need 267G free disk space to download and extract this dataset.
   SSD harddrives are recommended for faster speed.
   The time it takes to prepare the dataset depends on your Internet connection
   and disk speed. 

Prepare the dataset
-------------------

We will download and unzip the following files:

+-------------------------------------------------------------------------------------------------------+--------+
| File name                                                                                             | Size   |
+=======================================================================================================+========+
| `ILSVRC2015_VID.tar.gz <http://bvisionweb1.cs.unc.edu/ilsvrc2015/ILSVRC2015_VID.tar.gz>`_   |  100 G  |
+-------------------------------------------------------------------------------------------------------+--------+


we suggest run the command because it included download dataset and data processing，
the easiest way is to run this script:

 :download:`Download script: ilsvrc_vid.py<../../../scripts/datasets/ilsvrc_vid.py>`

.. code-block:: bash

   python ilsvrc_vid.py

"""
