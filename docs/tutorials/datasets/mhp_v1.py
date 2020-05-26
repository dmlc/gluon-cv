"""Prepare Multi-Human Parsing V1 dataset
=========================================

`Multi-Human Parsing V1 (MHP-v1) <https://github.com/ZhaoJ9014/Multi-Human-Parsing/>`_ is
a human-centric dataset for multi-human parsing task. It contains
five thousands images annotated with 18 categories.
This tutorial helps you to download MHP-v1 and set it up for later experiments.

.. image:: https://raw.githubusercontent.com/ZhaoJ9014/Multi-Human-Parsing/master/Figures/Fig1.png
   :width: 600 px

.. hint::

   You need 850 MB free disk space to download and extract this dataset.
   SSD harddrives are recommended for faster speed.
   The time it takes to prepare the dataset depends on your Internet connection
   and disk speed. For example, it takes around 1 mins on an AWS EC2 instance
   with EBS.

Prepare the dataset
-------------------

We will download and unzip the following files:

+-------------------------------------------------------------------------------------------------------+--------+
| File name                                                                                             | Size   |
+=======================================================================================================+========+
| `LV-MHP-v1.zip <https://drive.google.com/uc?id=1hTS8QJBuGdcppFAr_bvW2tsD9hW_ptr5&export=download>`_   | 850 MB |
+-------------------------------------------------------------------------------------------------------+--------+


The easiest way is to run this script:

 :download:`Download script: mhp_v1.py<../../../scripts/datasets/mhp_v1.py>`

.. code-block:: bash

   python mhp_v1.py

If you have already downloaded the above files and unzipped them,
you can set the folder name through ``--download-dir`` to avoid
downloading them again. For example

.. code-block:: python

   python mhp_v1.py --download-dir ~/.mxnet/datasets/mhp/LV-MHP-v1

"""
