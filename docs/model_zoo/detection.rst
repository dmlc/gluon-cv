.. _gluoncv-model-zoo-detection:

Detection
================

.. role:: gray

Visualization of Inference Throughputs vs. Validation mAP of COCO pre-trained models is illustrated in the first graph.

.. image:: /_static/plot_help.png
  :width: 100%

.. raw:: html
   :file: ../_static/detection_throughputs.html

We also provide a detailed interactive analysis of all 80 object categories.

.. raw:: html
   :file: ../_static/detection_coco_per_class.html

The following tables list pre-trained models for object detection
and their performances with more details.

.. hint::

  Model attributes are coded in their names.
  For instance, ``ssd_300_vgg16_atrous_voc`` consists of four parts:

  - ``ssd`` indicate the algorithm is "Single Shot Multibox Object Detection" [1]_.

  - ``300`` is the training image size, which means training images are resized to 300x300 and all anchor boxes are designed to match this shape. This may not apply to some models.

  - ``vgg16_atrous`` is the type of base feature extractor network.

  - ``voc`` is the training dataset. You can choose ``voc`` or ``coco``, etc.

  - ``(320x320)`` indicate that the model was evaluated with resolution 320x320. If not otherwise specified, all detection models in GluonCV can take various input shapes for prediction. Some models are trained with various input data shapes, e.g., Faster-RCNN and YOLO models.

  - ``ssd_300_vgg16_atrous_voc_int8`` is a quantized model calibrated on Pascal VOC dataset for ``ssd_300_vgg16_atrous_voc``.

.. hint::

  The training commands work with the following scripts:

  - For SSD [1]_ networks: :download:`Download train_ssd.py<../../scripts/detection/ssd/train_ssd.py>`
  - For Faster-RCNN [2]_ networks: :download:`Download train_faster_rcnn.py<../../scripts/detection/faster_rcnn/train_faster_rcnn.py>`
  - For YOLO v3 [3]_ networks: :download:`Download train_yolo3.py<../../scripts/detection/yolo/train_yolo3.py>`

Pascal VOC
~~~~~~~~~~

.. https://bit.ly/2JLnI2R

.. hint::

  For Pascal VOC dataset, training image set is the union of 2007trainval and 2012trainval and validation image set is 2007test.

  The VOC metric, mean Average Precision (mAP) across all classes with IoU threshold 0.5 is reported.

  Quantized SSD models are evaluated with ``nms_thresh=0.45``, ``nms_topk=200``.


SSD
---

Checkout SSD demo tutorial here: :ref:`sphx_glr_build_examples_detection_demo_ssd.py`

.. table::
   :widths: 50 5 25 20

   +----------------------------------------+-------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+
   | Model                                  | mAP   | Training Command                                                                                                                     | Training log                                                                                                                        |
   +========================================+=======+======================================================================================================================================+=====================================================================================================================================+
   | ssd_300_vgg16_atrous_voc [1]_          | 77.6  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/ssd_300_vgg16_atrous_voc.sh>`_          | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/ssd_300_vgg16_atrous_voc_train.log>`_           |
   +----------------------------------------+-------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+
   | ssd_300_vgg16_atrous_voc_int8* [1]_    | 77.46 |                                                                                                                                      |                                                                                                                                     |
   +----------------------------------------+-------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+
   | ssd_512_vgg16_atrous_voc [1]_          | 79.2  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/ssd_512_vgg16_atrous_voc.sh>`_          | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/ssd_512_vgg16_atrous_voc_train.log>`_           |
   +----------------------------------------+-------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+
   | ssd_512_vgg16_atrous_voc_int8* [1]_    | 78.39 |                                                                                                                                      |                                                                                                                                     |
   +----------------------------------------+-------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+
   | ssd_512_resnet50_v1_voc  [1]_          | 80.1  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/ssd_512_resnet50_v1_voc.sh>`_           | `log  <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/ssd_512_resnet50_v1_voc_train.log>`_           |
   +----------------------------------------+-------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+
   | ssd_512_resnet50_v1_voc_int8* [1]_     | 80.16 |                                                                                                                                      |                                                                                                                                     |
   +----------------------------------------+-------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+
   | ssd_512_mobilenet1.0_voc [1]_          | 75.4  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/ssd_512_mobilenet1_0_voc.sh>`_          | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/ssd_512_mobilenet1_0_voc_train.log>`_           |
   +----------------------------------------+-------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+
   | ssd_512_mobilenet1.0_voc_int8* [1]_    | 75.04 |                                                                                                                                      |                                                                                                                                     |
   +----------------------------------------+-------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+

Faster-RCNN
-----------

Faster-RCNN models of VOC dataset are evaluated with native resolutions with ``shorter side >= 600`` but ``longer side <= 1000`` without changing aspect ratios.

Checkout Faster-RCNN demo tutorial here: :ref:`sphx_glr_build_examples_detection_demo_faster_rcnn.py`

.. table::
   :widths: 50 5 25 20

   +----------------------------------+-------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+
   | Model                            | mAP   | Training Command                                                                                                                     | Training log                                                                                                                        |
   +==================================+=======+======================================================================================================================================+=====================================================================================================================================+
   | faster_rcnn_resnet50_v1b_voc [2]_| 78.3  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/faster_rcnn_resnet50_v1b_voc.sh>`_      | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/faster_rcnn_resnet50_v1b_voc_train.log>`_       |
   +----------------------------------+-------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+

YOLO-v3
-------

YOLO-v3 models can be evaluated and used for prediction at different resolutions. Different mAPs are reported with various evaluation resolutions, however, the models are identical.

Checkout YOLO demo tutorial here: :ref:`sphx_glr_build_examples_detection_demo_yolo.py`

.. table::
   :widths: 50 5 25 20

   +----------------------------------------------+-------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+
   | Model                                        | mAP   | Training Command                                                                                                                     | Training log                                                                                                                        |
   +==============================================+=======+======================================================================================================================================+=====================================================================================================================================+
   | yolo3_darknet53_voc [3]_ :gray:`(320x320)`   | 79.3  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/yolo3_darknet53_voc.sh>`_               | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/yolo3_darknet53_voc.log>`_                      |
   +----------------------------------------------+-------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+
   | yolo3_darknet53_voc [3]_ :gray:`(416x416)`   | 81.5  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/yolo3_darknet53_voc.sh>`_               | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/yolo3_darknet53_voc.log>`_                      |
   +----------------------------------------------+-------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+
   | yolo3_mobilenet1.0_voc [3]_ :gray:`(320x320)`| 73.9  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/yolo3_mobilenet1.0_voc.sh>`_            | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/yolo3_mobilenet1.0_voc.log>`_                   |
   +----------------------------------------------+-------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+
   | yolo3_mobilenet1.0_voc [3]_ :gray:`(416x416)`| 75.8  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/yolo3_mobilenet1.0_voc.sh>`_            | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/yolo3_mobilenet1.0_voc.log>`_                   |
   +----------------------------------------------+-------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+

MS COCO
~~~~~~~~~~

.. https://bit.ly/2JM82we

.. hint::

  For COCO dataset, training imageset is train2017 and validation imageset is val2017.

  The COCO metric, Average Precision (AP) with IoU threshold 0.5:0.95 (averaged 10 values, AP 0.5:0.95), 0.5 (AP 0.5) and 0.75 (AP 0.75) are reported together in the format (AP 0.5:0.95)/(AP 0.5)/(AP 0.75).

  For object detection task, only box overlap based AP is evaluated and reported.

SSD
---

Checkout SSD demo tutorial here: :ref:`sphx_glr_build_examples_detection_demo_ssd.py`

.. table::
   :widths: 50 5 25 20

   +-----------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+
   | Model                             | Box AP          | Training Command                                                                                                                  | Training Log                                                                                                                     |
   +===================================+=================+===================================================================================================================================+==================================================================================================================================+
   | ssd_300_vgg16_atrous_coco [1]_    | 25.1/42.9/25.8  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/ssd_300_vgg16_atrous_coco.sh>`_      | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/ssd_300_vgg16_atrous_coco_train.log>`_       |
   +-----------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+
   | ssd_512_vgg16_atrous_coco [1]_    | 28.9/47.9/30.6  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/ssd_512_vgg16_atrous_coco.sh>`_      | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/ssd_512_vgg16_atrous_coco_train.log>`_       |
   +-----------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+
   | ssd_512_resnet50_v1_coco [1]_     | 30.6/50.0/32.2  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/ssd_512_resnet50_v1_coco.sh>`_       | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/ssd_512_resnet50_v1_coco_train.log>`_        |
   +-----------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+
   | ssd_512_mobilenet1.0_coco [1]_    | 21.7/39.2/21.3  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/ssd_512_mobilenet1_0_coco.sh>`_      | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/ssd_512_mobilenet1_0_coco_train.log>`_       |
   +-----------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+


Faster-RCNN
-----------

Faster-RCNN models of VOC dataset are evaluated with native resolutions with ``shorter side >= 800`` but ``longer side <= 1300`` without changing aspect ratios.

Checkout Faster-RCNN demo tutorial here: :ref:`sphx_glr_build_examples_detection_demo_faster_rcnn.py`

.. table::
   :widths: 50 5 25 20

   +-------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------+
   | Model                                     | Box AP          | Training Command                                                                                                                        | Training Log                                                                                                                          |
   +===========================================+=================+=========================================================================================================================================+=======================================================================================================================================+
   | faster_rcnn_resnet50_v1b_coco [2]_        | 37.0/57.8/39.6  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/faster_rcnn_resnet50_v1b_coco.sh>`_        | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/faster_rcnn_resnet50_v1b_coco_train.log>`_        |
   +-------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------+
   | faster_rcnn_resnet101_v1d_coco [2]_       | 40.1/60.9/43.3  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/faster_rcnn_resnet101_v1d_coco.sh>`_       | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/faster_rcnn_resnet101_v1d_coco_train.log>`_       |
   +-------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------+
   | faster_rcnn_fpn_resnet50_v1b_coco [4]_    | 38.4/60.2/41.6  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/faster_rcnn_fpn_resnet50_v1b_coco.sh>`_    | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/faster_rcnn_fpn_resnet50_v1b_coco_train.log>`_    |
   +-------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------+
   | faster_rcnn_fpn_resnet101_v1d_coco [4]_   | 40.8/62.4/44.7  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/faster_rcnn_fpn_resnet101_v1d_coco.sh>`_   | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/faster_rcnn_fpn_resnet101_v1d_coco_train.log>`_   |
   +-------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------+
   | faster_rcnn_fpn_bn_resnet50_v1b_coco [5]_ | 39.3/61.3/42.9  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/faster_rcnn_fpn_bn_resnet50_v1b_coco.sh>`_ | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/faster_rcnn_fpn_bn_resnet50_v1b_coco_train.log>`_ |
   +-------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------------------------+

YOLO-v3
-------

YOLO-v3 models can be evaluated and used for prediction at different resolutions. Different mAPs are reported with various evaluation resolutions.

Checkout YOLO demo tutorial here: :ref:`sphx_glr_build_examples_detection_demo_yolo.py`

.. table::
   :widths: 50 5 25 20

   +------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+
   | Model                                          | Box AP          | Training Command                                                                                                                  | Training Log                                                                                                                     |
   +================================================+=================+===================================================================================================================================+==================================================================================================================================+
   | yolo3_darknet53_coco [3]_ :gray:`(320x320)`    | 33.6/54.1/35.8  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/yolo3_darknet53_coco.sh>`_           | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/yolo3_darknet53_coco_train.log>`_            |
   +------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+
   | yolo3_darknet53_coco [3]_ :gray:`(416x416)`    | 36.0/57.2/38.7  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/yolo3_darknet53_coco.sh>`_           | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/yolo3_darknet53_coco_train.log>`_            |
   +------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+
   | yolo3_darknet53_coco [3]_ :gray:`(608x608)`    | 37.0/58.2/40.1  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/yolo3_darknet53_coco.sh>`_           | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/yolo3_darknet53_coco_train.log>`_            |
   +------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+
   | yolo3_mobilenet1.0_coco [3]_ :gray:`(320x320)` | 26.7/46.1/27.5  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/yolo3_mobilenet1.0_coco.sh>`_        | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/yolo3_mobilenet1.0_coco.log>`_               |
   +------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+
   | yolo3_mobilenet1.0_coco [3]_ :gray:`(416x416)` | 28.6/48.9/29.9  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/yolo3_mobilenet1.0_coco.sh>`_        | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/yolo3_mobilenet1.0_coco.log>`_               |
   +------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+
   | yolo3_mobilenet1.0_coco [3]_ :gray:`(608x608)` | 28.0/49.8/27.8  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/yolo3_mobilenet1.0_coco.sh>`_        | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/yolo3_mobilenet1.0_coco.log>`_               |
   +------------------------------------------------+-----------------+-----------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+



.. [1] Wei Liu, Dragomir Anguelov, Dumitru Erhan,
       Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.
.. [2] Ren, Shaoqing, Kaiming He, Ross Girshick, and Jian Sun. \
       "Faster r-cnn: Towards real-time object detection with region proposal networks." \
       In Advances in neural information processing systems, pp. 91-99. 2015.
.. [3] Redmon, Joseph, and Ali Farhadi. \
       "Yolov3: An incremental improvement." \
       arXiv preprint arXiv:1804.02767 (2018).
.. [4] Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, Serge Belongie. \
       "Feature Pyramid Networks for Object Detection." \
       IEEE Conference on Computer Vision and Pattern Recognition 2017.
.. [5] Kaiming He, Ross Girshick, Piotr Dollár. \
       "Rethinking ImageNet Pre-training." \
       arXiv preprint arXiv:1811.08883 (2018).
