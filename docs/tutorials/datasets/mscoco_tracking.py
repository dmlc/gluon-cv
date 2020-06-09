"""Prepare COCO datasets
==============================

`COCO <http://cocodataset.org/#home>`_ is a large-scale object detection, segmentation, and captioning datasetself.
This tutorial will walk through the steps of preparing this dataset for object tracking in GluonCV.

.. image:: http://cocodataset.org/images/coco-logo.png

.. hint::

   You need 42.7 GB disk space to download and extract this dataset. SSD is
   preferred over HDD because of its better performance.

   The total time to prepare the dataset depends on your Internet speed and disk
   performance. For example, it often takes 20 min on AWS EC2 with EBS.

Prepare the dataset
-------------------

We need the following four files from `COCO <http://cocodataset.org/#download>`_:

+------------------------------------------------------------------------------------------------------------------------+--------+------------------------------------------+
| Filename                                                                                                               | Size   | SHA-1                                    |
+========================================================================================================================+========+==========================================+
| `train2017.zip <http://images.cocodataset.org/zips/train2017.zip>`_                                                    | 18 GB  | 10ad623668ab00c62c096f0ed636d6aff41faca5 |
+------------------------------------------------------------------------------------------------------------------------+--------+------------------------------------------+
| `val2017.zip <http://images.cocodataset.org/zips/val2017.zip>`_                                                        | 778 MB | 4950dc9d00dbe1c933ee0170f5797584351d2a41 |
+------------------------------------------------------------------------------------------------------------------------+--------+------------------------------------------+
| `annotations_trainval2017.zip  <http://images.cocodataset.org/annotations/annotations_trainval2017.zip>`_              | 241 MB | 8551ee4bb5860311e79dace7e79cb91e432e78b3 |
+------------------------------------------------------------------------------------------------------------------------+--------+------------------------------------------+
| `stuff_annotations_trainval2017.zip <http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip>`_   | 401 MB | e7aa0f7515c07e23873a9f71d9095b06bcea3e12 |
+------------------------------------------------------------------------------------------------------------------------+--------+------------------------------------------+


The easiest way to download and unpack these files is to download helper script and we suggest run the command because it included download dataset and data processing
:download:`mscoco_tracking.py<../../../scripts/datasets/mscoco_tracking.py>` and run
the following command:

The easiest way is to run this script:

 :download:`Download script: coco_tracking.py<../../../scripts/datasets/mscoco_racking.py>`

.. code-block:: bash

   python mscoco_tracking.py

"""
