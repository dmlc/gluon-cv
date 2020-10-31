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

.. csv-table::
   :file: ./csv_tables/Detections/Pascal_SSD.csv
   :header-rows: 1
   :class: tight-table
   :widths: 50 10 20 20

Faster-RCNN
-----------

Faster-RCNN models of VOC dataset are evaluated with native resolutions with ``shorter side >= 600`` but ``longer side <= 1000`` without changing aspect ratios.

Checkout Faster-RCNN demo tutorial here: :ref:`sphx_glr_build_examples_detection_demo_faster_rcnn.py`

.. csv-table::
   :file: ./csv_tables/Detections/Pascal_Faster-RCNN.csv
   :header-rows: 1
   :class: tight-table
   :widths: 50 10 20 20

YOLO-v3
-------

YOLO-v3 models can be evaluated and used for prediction at different resolutions. Different mAPs are reported with various evaluation resolutions, however, the models are identical.

Checkout YOLO demo tutorial here: :ref:`sphx_glr_build_examples_detection_demo_yolo.py`

.. csv-table::
   :file: ./csv_tables/Detections/Pascal_YOLO-v3.csv
   :header-rows: 1
   :class: tight-table
   :widths: 50 10 20 20

CenterNet
---------

CenterNet models are evaluated at 512x512 resolution. mAPs with flipped inference(F) are also reported, however, the models are identical.
Checkout CenterNet demo tutorial here: :ref:`sphx_glr_build_examples_detection_demo_center_net.py`

Note that ``dcnv2`` indicate that models include Modulated Deformable Convolution (DCNv2) layers, you may need to upgrade MXNet in order to use them.

.. csv-table::
   :file: ./csv_tables/Detections/Pascal_CenterNet.csv
   :header-rows: 1
   :class: tight-table
   :widths: 50 15 15 20

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

.. csv-table::
   :file: ./csv_tables/Detections/MSCOCO_SSD.csv
   :header-rows: 1
   :class: tight-table
   :widths: 50 20 15 15

Faster-RCNN
-----------

Faster-RCNN models of VOC dataset are evaluated with native resolutions with ``shorter side >= 800`` but ``longer side <= 1333`` without changing aspect ratios.

Checkout Faster-RCNN demo tutorial here: :ref:`sphx_glr_build_examples_detection_demo_faster_rcnn.py`

.. csv-table::
   :file: ./csv_tables/Detections/MSCOCO_Faster-RCNN.csv
   :header-rows: 1
   :class: tight-table
   :widths: 50 20 15 15

YOLO-v3
-------

YOLO-v3 models can be evaluated and used for prediction at different resolutions. Different mAPs are reported with various evaluation resolutions.

Checkout YOLO demo tutorial here: :ref:`sphx_glr_build_examples_detection_demo_yolo.py`

.. csv-table::
   :file: ./csv_tables/Detections/MSCOCO_YOLO-v3.csv
   :header-rows: 1
   :class: tight-table
   :widths: 50 20 15 15

CenterNet
---------

CenterNet models are evaluated at 512x512 resolution. mAPs with flipped inference(F) are also reported, however, the models are identical.
Checkout CenterNet demo tutorial here: :ref:`sphx_glr_build_examples_detection_demo_center_net.py`.

Note that ``dcnv2`` indicate that models include Modulated Deformable Convolution (DCNv2) layers, you may need to upgrade MXNet in order to use them.

.. csv-table::
   :file: ./csv_tables/Detections/MSCOCO_CenterNet.csv
   :header-rows: 1
   :class: tight-table
   :widths: 50 15 15 20
