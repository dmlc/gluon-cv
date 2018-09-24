.. _gluoncv-model-zoo-detection:

Detection
================

The following table lists pre-trained models for object detection
and their performances.

.. hint::

  Model attributes are coded in their names.
  For instance, ``ssd_300_vgg16_atrous_voc`` consists of four parts:

  - ``ssd`` indicate the algorithm is "Single Shot Multibox Object Detection" [1]_.

  - ``300`` is the training image size, which means training images are resized to 300x300 and all anchor boxes are designed to match this shape. This may not apply to some models.

  - ``vgg16_atrous`` is the type of base feature extractor network.

  - ``voc`` is the training dataset. You can choose ``voc`` or ``coco``, etc.

  - ``@ 320x320`` indicate that the model was evaluated with resolution 320x320. If not otherwise specified, all detection models in GluonCV can take various input shapes for prediction. Some models are trained with various input data shapes, e.g., Faster-RCNN and YOLO models.

.. hint::

  The training commands work with the following scripts:

  - For SSD [1]_ networks: :download:`Download train_ssd.py<../../scripts/detection/ssd/train_ssd.py>`
  - For Faster-RCNN [2]_ networks: :download:`Download train_faster_rcnn.py<../../scripts/detection/faster_rcnn/train_faster_rcnn.py>`
  - For YOLO v3 [3]_ networks: :download:`Download train_yolo3_rand_size.py<../../scripts/detection/yolo/train_yolo3_rand_size.py>` or :download:`Download train_yolo3.py<../../scripts/detection/yolo/train_yolo3.py>` with fixed size training, which is faster than random size training pipeline.

.. https://bit.ly/2JLnI2R

.. hint::

  For Pascal VOC dataset, training image set is the union of 2007trainval and 2012trainval and validation image set is 2007test.

  The VOC metric, mean Average Precision (mAP) across all classes with IoU threshold 0.5 is reported.

+----------------------------------+-------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+
| Model                            | mAP   | Training Command                                                                                                                     | Training log                                                                                                                        |
+==================================+=======+======================================================================================================================================+=====================================================================================================================================+
| ssd_300_vgg16_atrous_voc         | 77.6  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/ssd_300_vgg16_atrous_voc.sh>`_          | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/ssd_300_vgg16_atrous_voc_train.log>`_           |
+----------------------------------+-------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+
| ssd_512_vgg16_atrous_voc         | 79.2  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/ssd_512_vgg16_atrous_voc.sh>`_          | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/ssd_512_vgg16_atrous_voc_train.log>`_           |
+----------------------------------+-------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+
| ssd_512_resnet50_v1_voc          | 80.1  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/ssd_512_resnet50_v1_voc.sh>`_           | `log  <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/ssd_512_resnet50_v1_voc_train.log>`_           |
+----------------------------------+-------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+
| ssd_512_mobilenet1.0_voc         | 75.4  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/ssd_512_mobilenet1_0_voc.sh>`_          | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/ssd_512_mobilenet1_0_voc_train.log>`_           |
+----------------------------------+-------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+
| faster_rcnn_resnet50_v1b_voc     | 78.3  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/faster_rcnn_resnet50_v1b_voc.sh>`_      | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/faster_rcnn_resnet50_v1b_voc_train.log>`_       |
+----------------------------------+-------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+
| yolo3_darknet53_voc @ 320x320    | 79.3  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/yolo3_darknet53_voc.sh>`_               | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/yolo3_darknet53_voc.log>`_                      |
+----------------------------------+-------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+
| yolo3_darknet53_voc @ 416x416    | 81.5  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/yolo3_darknet53_voc.sh>`_               | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/yolo3_darknet53_voc.log>`_                      |
+----------------------------------+-------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+

.. https://bit.ly/2JM82we

.. hint::

  For COCO dataset, training imageset is train2017 and validation imageset is val2017.

  The COCO metric, Average Precision (AP) with IoU threshold 0.5:0.95 (averaged 10 values, AP 0.5:0.95), 0.5 (AP 0.5) and 0.75 (AP 0.75) are reported together in the format (AP 0.5:0.95)/(AP 0.5)/(AP 0.75).

  For object detection task, only box overlap based AP is evaluated and reported.

+-----------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+
| Model                             | Box AP          | Training Command                                                                                                                  | Training Log                                                                                                                     |
+===================================+=================+===================================================================================================================================+==================================================================================================================================+
| ssd_300_vgg16_atrous_coco         | 25.1/42.9/25.8  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/ssd_300_vgg16_atrous_coco.sh>`_      | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/ssd_300_vgg16_atrous_coco_train.log>`_       |
+-----------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+
| ssd_512_vgg16_atrous_coco         | 28.9/47.9/30.6  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/ssd_512_vgg16_atrous_coco.sh>`_      | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/ssd_512_vgg16_atrous_coco_train.log>`_       |
+-----------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+
| ssd_512_resnet50_v1_coco          | 30.6/50.0/32.2  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/ssd_512_resnet50_v1_coco.sh>`_       | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/ssd_512_resnet50_v1_coco_train.log>`_        |
+-----------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+
| faster_rcnn_resnet50_v1b_coco     | 36.8/57.3/39.6  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/faster_rcnn_resnet50_v1b_coco.sh>`_  | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/faster_rcnn_resnet50_v1b_coco_train.log>`_   |
+-----------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+
| yolo3_darknet53_coco @ 320x320    | 31.9/52.7/33.3  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/yolo3_darknet53_coco.sh>`_           | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/yolo3_darknet53_coco_train.log>`_            |
+-----------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+
| yolo3_darknet53_coco @ 416x416    | 34.3/55.1/36.7  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/yolo3_darknet53_coco.sh>`_           | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/yolo3_darknet53_coco_train.log>`_            |
+-----------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+
| yolo3_darknet53_coco @ 608x608    | 35.6/57.1/38.2  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/yolo3_darknet53_coco.sh>`_           | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/yolo3_darknet53_coco_train.log>`_            |
+-----------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+


.. [1] Wei Liu, Dragomir Anguelov, Dumitru Erhan,
       Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.
.. [2] Ren, Shaoqing, Kaiming He, Ross Girshick, and Jian Sun. \
       "Faster r-cnn: Towards real-time object detection with region proposal networks." \
       In Advances in neural information processing systems, pp. 91-99. 2015.
.. [3] Redmon, Joseph, and Ali Farhadi. \
       "Yolov3: An incremental improvement." \
       arXiv preprint arXiv:1804.02767 (2018).
