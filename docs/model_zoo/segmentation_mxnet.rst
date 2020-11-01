Visualization of Inference Throughputs vs. Validation mIoU of COCO pre-trained models is illustrated in the following graph. Throughputs are measured with single V100 GPU and batch size 16.

.. image:: /_static/plot_help.png
  :width: 100%

.. raw:: html
   :file: ../_static/semantic_segmentation_throughputs.html

.. hint::

  The model names contain the training information. For instance, ``fcn_resnet50_voc``:

  - ``fcn`` indicate the algorithm is "Fully Convolutional Network for Semantic Segmentation" [2]_.

  - ``resnet50`` is the name of backbone network.

  - ``voc`` is the training dataset.

Semantic Segmentation
~~~~~~~~~~~~~~~~~~~~~

Table of pre-trained models for semantic segmentation and their performance.

.. hint::

  The test script :download:`Download test.py<../../scripts/segmentation/test.py>` can be used for
  evaluating the models (VOC results are evaluated using the official server). For example ``fcn_resnet50_ade``::

    python test.py --dataset ade20k --model-zoo fcn_resnet50_ade --eval

  The training commands work with the script: :download:`Download train.py<../../scripts/segmentation/train.py>`


.. role:: raw-html(raw)
   :format: html

ADE20K Dataset
--------------

.. csv-table::
   :file: ./csv_tables/Segmentations/SS_ADE20K.csv
   :header-rows: 1
   :class: tight-table
   :widths: 35 15 10 10 15 15

MS-COCO Dataset Pretrain
------------------------

.. csv-table::
   :file: ./csv_tables/Segmentations/SS_MS-COCO.csv
   :header-rows: 1
   :class: tight-table
   :widths: 35 15 10 10 15 15

Pascal VOC Dataset
------------------

.. csv-table::
   :file: ./csv_tables/Segmentations/SS_Pascal-VOC.csv
   :header-rows: 1
   :class: tight-table
   :widths: 35 15 10 10 15 15

.. _83.6:  http://host.robots.ox.ac.uk:8080/anonymous/YB1AN5.html
.. _85.1:  http://host.robots.ox.ac.uk:8080/anonymous/9RTTZC.html
.. _86.2:  http://host.robots.ox.ac.uk:8080/anonymous/ZPN6II.html
.. _86.7:  http://host.robots.ox.ac.uk:8080/anonymous/XZEXL2.html

Cityscapes Dataset
------------------

.. csv-table::
   :file: ./csv_tables/Segmentations/SS_Cityscapes.csv
   :header-rows: 1
   :class: tight-table
   :widths: 35 15 10 10 15 15

MHP-V1 Dataset
--------------

.. csv-table::
   :file: ./csv_tables/Segmentations/SS_MHP-V1.csv
   :header-rows: 1
   :class: tight-table
   :widths: 35 15 10 10 15 15

Instance Segmentation
~~~~~~~~~~~~~~~~~~~~~

Table of pre-trained models for instance segmentation and their performance.

.. hint::

  The training commands work with the following scripts:

  - For Mask R-CNN networks: :download:`Download train_mask_rcnn.py<../../scripts/instance/mask_rcnn/train_mask_rcnn.py>`

  For COCO dataset, training imageset is train2017 and validation imageset is val2017.

  Average precision with IoU threshold 0.5:0.95 (averaged 10 values), 0.5 and 0.75 are reported together in the format (AP 0.5:0.95)/(AP 0.5)/(AP 0.75).

  For instance segmentation task, both box overlap and segmentation overlap based AP are evaluated and reported.


MS COCO
-------

.. csv-table::
   :file: ./csv_tables/Segmentations/IS_MS-COCO.csv
   :header-rows: 1
   :class: tight-table
   :widths: 35 18 18 14 15
