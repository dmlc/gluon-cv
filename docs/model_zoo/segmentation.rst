.. _gluoncv-model-zoo-segmentation:

Segmentation
============

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

+-----------------------+-----------------+-----------+-----------+------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------+
| Name                  | Method          | pixAcc    | mIoU      | Command                                                                                                                      | log                                                                                                                 |
+=======================+=================+===========+===========+==============================================================================================================================+=====================================================================================================================+
| fcn_resnet101_coco    | FCN [2]_        | 92.2      | 66.2      | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/segmentation/fcn_resnet101_coco.sh>`_     | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/segmentation/fcn_resnet101_coco.log>`_    |
+-----------------------+-----------------+-----------+-----------+------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------+
| psp_resnet101_coco    | PSP [3]_        | 92.4      | 70.4      | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/segmentation/psp_resnet101_coco.sh>`_     | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/segmentation/psp_resnet101_coco.log>`_    |
+-----------------------+-----------------+-----------+-----------+------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------+
| deeplab_resnet101_coco| DeepLabV3 [4]_  | 92.5      | 70.4      | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/segmentation/deeplab_resnet101_voc.sh>`_  | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/segmentation/deeplab_resnet101_voc.log>`_ |
+-----------------------+-----------------+-----------+-----------+------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------+
| fcn_resnet101_voc     | FCN [2]_        | N/A       | 83.6_     | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/segmentation/fcn_resnet101_voc.sh>`_      | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/segmentation/fcn_resnet101_voc.log>`_     |
+-----------------------+-----------------+-----------+-----------+------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------+
| psp_resnet101_voc     | PSP [3]_        | N/A       | 85.1_     | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/segmentation/psp_resnet101_voc.sh>`_      | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/segmentation/psp_resnet101_voc.log>`_     |
+-----------------------+-----------------+-----------+-----------+------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------+
| deeplab_resnet101_voc | DeepLabV3 [4]_  | N/A       | 86.2_     | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/segmentation/deeplab_resnet101_voc.sh>`_  | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/segmentation/deeplab_resnet101_voc.log>`_ |
+-----------------------+-----------------+-----------+-----------+------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------+
| fcn_resnet50_ade      | FCN [2]_        | 79.0      | 39.5      | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/segmentation/fcn_resnet50_ade.sh>`_       | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/segmentation/fcn_resnet50_ade.log>`_      |
+-----------------------+-----------------+-----------+-----------+------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------+
| fcn_resnet101_ade     | FCN [2]_        | 80.6      | 41.6      | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/segmentation/fcn_resnet101_ade.sh>`_      | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/segmentation/fcn_resnet101_ade.log>`_     |
+-----------------------+-----------------+-----------+-----------+------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------+
| psp_resnet50_ade      | PSP [3]_        | 80.1      | 41.6      | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/segmentation/psp_resnet50_ade.sh>`_       | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/segmentation/psp_resnet50_ade.log>`_      |
+-----------------------+-----------------+-----------+-----------+------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------+
| psp_resnet101_ade     | PSP [3]_        | 80.8      | 42.9      | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/segmentation/psp_resnet101_ade.sh>`_      | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/segmentation/psp_resnet101_ade.log>`_     |
+-----------------------+-----------------+-----------+-----------+------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------+
| deeplab_resnet50_ade  | DeepLabV3 [4]_  | 80.5      | 42.5      | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/segmentation/deeplab_resnet50_ade.sh>`_   | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/segmentation/deeplab_resnet50_ade.log>`_  |
+-----------------------+-----------------+-----------+-----------+------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------+
| deeplab_resnet101_ade | DeepLabV3 [4]_  | 81.1      | 44.1      | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/segmentation/deeplab_resnet101_ade.sh>`_  | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/segmentation/deeplab_resnet101_ade.log>`_ |
+-----------------------+-----------------+-----------+-----------+------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------+

.. _83.6:  http://host.robots.ox.ac.uk:8080/anonymous/YB1AN5.html
.. _85.1:  http://host.robots.ox.ac.uk:8080/anonymous/9RTTZC.html
.. _86.2:  http://host.robots.ox.ac.uk:8080/anonymous/ZPN6II.html

Instance Segmentation
~~~~~~~~~~~~~~~~~~~~~

Table of pre-trained models for instance segmentation and their performance.

.. hint::

  The training commands work with the following scripts:

  - For Mask R-CNN networks: :download:`Download train_mask_rcnn.py<../../scripts/instance/mask_rcnn/train_mask_rcnn.py>`

  For COCO dataset, training imageset is train2017 and validation imageset is val2017.

  Average precision with IoU threshold 0.5:0.95 (averaged 10 values), 0.5 and 0.75 are reported together in the format (AP 0.5:0.95)/(AP 0.5)/(AP 0.75).

  For instance segmentation task, both box overlap and segmentation overlap based AP are evaluated and reported.

+------------------------------+---------------------------+--------------------------+------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------+
| Model                        | Box AP                    | Segm AP                  | Command                                                                                                                            | Training Log                                                                                                                         |
+==============================+===========================+==========================+====================================================================================================================================+======================================================================================================================================+
| mask_rcnn_resnet50_v1b_coco  | 38.3/58.7/41.4            | 33.1/54.8/35.0           | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/instance/mask_rcnn_resnet50_v1b_coco.sh>`_      | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/instance/mask_rcnn_resnet50_v1b_coco_train.log>`_          |
+------------------------------+---------------------------+--------------------------+------------------------------------------------------------------------------------------------------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------+

.. [1] He, Kaming, Georgia Gkioxari, Piotr Dollár and Ross Girshick. \
        "Mask R-CNN." \
        In IEEE International Conference on Computer Vision (ICCV), 2017.
.. [2] Long, Jonathan, Evan Shelhamer, and Trevor Darrell. \
       "Fully convolutional networks for semantic segmentation." \
       Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.
.. [3] Zhao, Hengshuang, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, and Jiaya Jia. \
       "Pyramid scene parsing network." *CVPR*, 2017
.. [4] Chen, Liang-Chieh, et al. "Rethinking atrous convolution for semantic image segmentation." \
       arXiv preprint arXiv:1706.05587 (2017).
