"""Prepare OTB 2015 dataset
=========================================

`OTB 2015 dataset <http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html/>`is Visual Tracker Benchmark.
The full benchmark contains 100 sequences from recent literatures.

This tutorial helps you to download OTB 2015 and set it up for later experiments.

.. hint::

   You need 5G free disk space to download and extract this dataset.
   SSD harddrives are recommended for faster speed.
   The time it takes to prepare the dataset depends on your Internet connection
   and disk speed. 

Prepare the dataset
-------------------

We will download and unzip the 100 videoes,for example you download Basketball video:

+-------------------------------------------------------------------------------------------------------+--------+
| File name                                                                                             | Size   |
+=======================================================================================================+========+
| `video.zip <http://cvlab.hanyang.ac.kr/tracker_benchmark/seq/Basketball.zip>`_   |  50.9 mb  |
+-------------------------------------------------------------------------------------------------------+--------+


we suggest run the command because it included download dataset and data processing，
the easiest way is to run this script:

 :download:`Download script: otb2015.py<../../../scripts/datasets/otb2015.py>`

.. code-block:: bash

   python otb2015.py

If you want to get json, you should follow `OTB2015_json <https://github.com/STVIR/pysot/tree/master/testing_dataset>`_

"""

