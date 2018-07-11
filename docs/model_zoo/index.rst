.. _gluoncv-model-zoo:

GluonCV Model Zoo
================

GluonCV Model Zoo, similar to the upstream `Gluon Model Zoo
<https://mxnet.incubator.apache.org/api/python/gluon/model_zoo.html>`_,
provides pre-defined and pre-trained models to help bootstrap computer vision
applications.

Model Zoo API
-------------

.. code-block:: python

    from gluoncv import model_zoo
    # load a ResNet model trained on CIFAR10
    cifar_resnet20 = model_zoo.get_model('cifar_resnet20_v1', pretrained=True)
    # load a pre-trained ssd model
    ssd0 = model_zoo.get_model('ssd_300_vgg16_atrous_voc', pretrained=True)
    # load ssd model with pre-trained feature extractors
    ssd1 = model_zoo.get_model('ssd_512_vgg16_atrous_voc', pretrained_base=True)
    # load ssd model without initialization
    ssd2 = model_zoo.get_model('ssd_512_resnet50_v1_voc', pretrained_base=False)

We recommend using :py:meth:`gluoncv.model_zoo.get_model` for loading
pre-defined models, because it provides name checking and list available choices.

However, you can still load models by directly instantiate it like

.. code-block:: python

    from gluoncv import model_zoo
    cifar_resnet20 = model_zoo.cifar_resnet20_v1(pretrained=True)

.. hint::

  Detailed ``model_zoo`` APIs are available in API reference: :py:meth:`gluoncv.model_zoo`.

Summary of Available Models
---------------------------

GluonCV is still under development, stay tuned for more models!

Image Classification
~~~~~~~~~~~~~~~~~~~~

The following table lists pre-trained models on ImageNet. We will keep
adding new models and training scripts to the table.

Besides the listed, we provide more models trained on ImageNet in the upstream
`Gluon Model Zoo <https://mxnet.incubator.apache.org/api/python/gluon/model_zoo.html>`_.

**ImageNet**

.. hint::

    Training commands work with this script:

    :download:`Download train_imagenet.py<../../scripts/classification/imagenet/train_imagenet.py>`

    The `resnet_v1b` family is a modified version of `resnet_v1`, specifically we set stride at the 3x3 layer for a bottleneck block. `ResNet18` and `ResNet34` have identical `v1` and `v1b` network structures. This modification has been mentioned in recent literatures, e.g. [8]_ .

+-----------------------+--------+--------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
| Model                 | Top-1  | Top-5  | Training Command                                                                                                                   | Training Log                                                                                                                  |
+=======================+========+========+====================================================================================================================================+===============================================================================================================================+
| ResNet18_v1 [1]_      | 70.93  | 89.92  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet18_v1.sh>`_       | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet18_v1.log>`_          |
+-----------------------+--------+--------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
| ResNet34_v1 [1]_      | 74.37  | 91.87  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet34_v1.sh>`_       | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet34_v1.log>`_          |
+-----------------------+--------+--------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
| ResNet50_v1 [1]_      | 76.47  | 93.13  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet50_v1.sh>`_       | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet50_v1.log>`_          |
+-----------------------+--------+--------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
| ResNet101_v1 [1]_     | 78.34  | 94.01  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet101_v1.sh>`_      | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet101_v1.log>`_         |
+-----------------------+--------+--------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
| ResNet152_v1 [1]_     | 79.00  | 94.38  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet152_v1.sh>`_      | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet152_v1.log>`_         |
+-----------------------+--------+--------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
| ResNet18_v1b [1]_     | 70.94  | 89.83  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet18_v1b.sh>`_      | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet18_v1b.log>`_         |
+-----------------------+--------+--------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
| ResNet34_v1b [1]_     | 74.65  | 92.08  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet34_v1b.sh>`_      | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet34_v1b.log>`_         |
+-----------------------+--------+--------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
| ResNet50_v1b [1]_     | 77.07  | 93.55  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet50_v1b.sh>`_      | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet50_v1b.log>`_         |
+-----------------------+--------+--------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
| ResNet101_v1b [1]_    | 78.81  | 94.39  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet101_v1b.sh>`_     | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet101_v1b.log>`_        |
+-----------------------+--------+--------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
| ResNet152_v1b [1]_    | 79.44  | 94.61  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet152_v1b.sh>`_     | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet152_v1b.log>`_        |
+-----------------------+--------+--------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
| ResNet18_v2 [2]_      | 71.00  | 89.92  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet18_v2.sh>`_       | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet18_v2.log>`_          |
+-----------------------+--------+--------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
| ResNet34_v2 [2]_      | 74.40  | 92.08  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet34_v2.sh>`_       | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet34_v2.log>`_          |
+-----------------------+--------+--------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
| ResNet50_v2 [2]_      | 77.11  | 93.43  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet50_v2.sh>`_       | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet50_v2.log>`_          |
+-----------------------+--------+--------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
| ResNet101_v2 [2]_     | 78.53  | 94.17  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet101_v2.sh>`_      | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet101_v2.log>`_         |
+-----------------------+--------+--------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
| ResNet152_v2 [2]_     | 79.21  | 94.31  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet152_v2.sh>`_      | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/resnet152_v2.log>`_         |
+-----------------------+--------+--------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
| MobileNetV2_1.0 [7]_  | 71.92  | 90.56  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/mobilenetv2_1.0.sh>`_   | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/mobilenetv2_1.0.log>`_      |
+-----------------------+--------+--------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
| MobileNetV2_0.75 [7]_ | 69.61  | 88.95  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/mobilenetv2_0.75.sh>`_  | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/mobilenetv2_0.75.log>`_     |
+-----------------------+--------+--------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
| MobileNetV2_0.5 [7]_  | 64.49  | 85.47  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/mobilenetv2_0.5.sh>`_   | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/mobilenetv2_0.5.log>`_      |
+-----------------------+--------+--------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+
| MobileNetV2_0.25 [7]_ | 50.74  | 74.56  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/mobilenetv2_0.25.sh>`_  | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/imagenet/mobilenetv2_0.25.log>`_     |
+-----------------------+--------+--------+------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------+

**CIFAR10**

The following table lists pre-trained models trained on CIFAR10.

.. hint::

    Our pre-trained models reproduce results from "Mix-Up" [4]_ .
    Please check the reference paper for further information.

    Training commands in the table work with the following scripts:

    - For vanilla training: :download:`Download train_cifar10.py<../../scripts/classification/cifar/train_cifar10.py>`
    - For mix-up training: :download:`Download train_mixup_cifar10.py<../../scripts/classification/cifar/train_mixup_cifar10.py>`

+------------------------------+----------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Model                        | Acc (Vanilla/Mix-Up [4]_ ) | Training Command                                                                                                                                                                                                                                                         | Training Log                                                                                                                                                                                                                                                               |
+==============================+============================+==========================================================================================================================================================================================================================================================================+============================================================================================================================================================================================================================================================================+
| CIFAR_ResNet20_v1 [1]_       | 92.1 / 92.9                | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnet20_v1.sh>`_ / `Mix-Up <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnet20_v1_mixup.sh>`_             | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnet20_v1.log>`_ / `Mix-Up <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnet20_v1_mixup.log>`_             |
+------------------------------+----------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| CIFAR_ResNet56_v1 [1]_       | 93.6 / 94.2                | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnet56_v1.sh>`_ / `Mix-Up <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnet56_v1_mixup.sh>`_             | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnet56_v1.log>`_ / `Mix-Up <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnet56_v1_mixup.log>`_             |
+------------------------------+----------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| CIFAR_ResNet110_v1 [1]_      | 93.0 / 95.2                | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnet110_v1.sh>`_ / `Mix-Up <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnet110_v1_mixup.sh>`_           | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnet110_v1.log>`_ / `Mix-Up <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnet110_v1_mixup.log>`_           |
+------------------------------+----------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| CIFAR_ResNet20_v2 [2]_       | 92.1 / 92.7                | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnet20_v2.sh>`_ / `Mix-Up <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnet20_v2_mixup.sh>`_             | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnet20_v2.log>`_ / `Mix-Up <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnet20_v2_mixup.log>`_             |
+------------------------------+----------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| CIFAR_ResNet56_v2 [2]_       | 93.7 / 94.6                | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnet56_v2.sh>`_ / `Mix-Up <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnet56_v2_mixup.sh>`_             | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnet56_v2.log>`_ / `Mix-Up <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnet56_v2_mixup.log>`_             |
+------------------------------+----------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| CIFAR_ResNet110_v2 [2]_      | 94.3 / 95.5                | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnet110_v2.sh>`_ / `Mix-Up <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnet110_v2_mixup.sh>`_           | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnet110_v2.log>`_ / `Mix-Up <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnet110_v2_mixup.log>`_           |
+------------------------------+----------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| CIFAR_WideResNet16_10 [3]_   | 95.1 / 96.7                | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_wideresnet16_10.sh>`_ / `Mix-Up <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_wideresnet16_10_mixup.sh>`_     | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_wideresnet16_10.log>`_ / `Mix-Up <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_wideresnet16_10_mixup.log>`_     |
+------------------------------+----------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| CIFAR_WideResNet28_10 [3]_   | 95.6 / 97.2                | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_wideresnet28_10.sh>`_ / `Mix-Up <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_wideresnet28_10_mixup.sh>`_     | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_wideresnet28_10.log>`_ / `Mix-Up <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_wideresnet28_10_mixup.log>`_     |
+------------------------------+----------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| CIFAR_WideResNet40_8 [3]_    | 95.9 / 97.3                | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_wideresnet40_8.sh>`_ / `Mix-Up <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_wideresnet40_8_mixup.sh>`_       | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_wideresnet40_8.log>`_ / `Mix-Up <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_wideresnet40_8_mixup.log>`_       |
+------------------------------+----------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| CIFAR_ResNeXt29_16x64d [8]_  | 96.3 / 97.3                | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnext29_16x64d.sh>`_ / `Mix-Up <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnext29_16x64d_mixup.sh>`_   | `Vanilla <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnext29_16x64d.log>`_ / `Mix-Up <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/classification/cifar/cifar_resnext29_16x64d_mixup.log>`_   |
+------------------------------+----------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Object Detection
~~~~~~~~~~~~~~~~

The following table lists pre-trained models for object detection
and their performances.

.. hint::

  Model attributes are coded in their names.
  For instance, ``ssd_300_vgg16_atrous_voc`` consists of four parts:

  - ``ssd`` indicate the algorithm is "Single Shot Multibox Object Detection" [5]_.

  - ``300`` is the training image size, which means training images are resized to 300x300 and all anchor boxes are designed to match this shape.

  - ``vgg16_atrous`` is the type of base feature extractor network.

  - ``voc`` is the training dataset. You can choose ``voc`` or ``coco``, etc.

.. hint::

  The training commands work with the following scripts:

  - For SSD networks: :download:`Download train_ssd.py<../../scripts/detection/ssd/train_ssd.py>`
  - For Faster-RCNN networks: :download:`Download train_faster_rcnn.py<../../scripts/detection/faster_rcnn/train_faster_rcnn.py>`


.. https://bit.ly/2JLnI2R

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
| faster_rcnn_resnet50_v2a_voc     | 77.9  | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/faster_rcnn_resnet50_v2a_voc.sh>`_      | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/faster_rcnn_resnet50_v2a_voc_train.log>`_       |
+----------------------------------+-------+--------------------------------------------------------------------------------------------------------------------------------------+-------------------------------------------------------------------------------------------------------------------------------------+

.. https://bit.ly/2JM82we

+-----------------------------------+----------+------+------+-----------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+
| Model                             | 0.5:0.95 | 0.5  | 0.75 | Training Command                                                                                                                  | Training Log                                                                                                                     |
+===================================+==========+======+======+===================================================================================================================================+==================================================================================================================================+
| ssd_300_vgg16_atrous_coco         | 25.1     | 42.9 | 25.8 | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/ssd_300_vgg16_atrous_coco.sh>`_      | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/ssd_300_vgg16_atrous_coco_train.log>`_       |
+-----------------------------------+----------+------+------+-----------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+
| ssd_512_vgg16_atrous_coco         | 28.9     | 47.9 | 30.6 | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/ssd_512_vgg16_atrous_coco.sh>`_      | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/ssd_512_vgg16_atrous_coco_train.log>`_       |
+-----------------------------------+----------+------+------+-----------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+
| ssd_512_resnet50_v1_coco          | 30.6     | 50.0 | 32.2 | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/ssd_512_resnet50_v1_coco.sh>`_       | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/detection/ssd_512_resnet50_v1_coco_train.log>`_        |
+-----------------------------------+----------+------+------+-----------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------+

Semantic Segmentation
~~~~~~~~~~~~~~~~~~~~~

Table of pre-trained models for semantic segmentation and their performance.

.. hint::

  The model names contain the training information. For instance, ``fcn_resnet50_voc``:

  - ``fcn`` indicate the algorithm is "Fully Convolutional Network for Semantic Segmentation" [6]_.

  - ``resnet50`` is the name of backbone network.

  - ``voc`` is the training dataset.

  The test script :download:`Download test.py<../../scripts/segmentation/test.py>` can be used for
  evaluating the models (VOC results are evaluated using the official server). For example ``fcn_resnet50_ade``::

    python test.py --dataset ade20k --model-zoo fcn_resnet50_ade --eval

  The training commands work with the script: :download:`Download train.py<../../scripts/segmentation/train.py>`


.. role:: raw-html(raw)
   :format: html

+-------------------+--------------+-----------+-----------+-----------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------+
| Name              | Method       | pixAcc    | mIoU      | Command                                                                                                                     | log                                                                                                                 |
+===================+==============+===========+===========+=============================================================================================================================+=====================================================================================================================+
| fcn_resnet50_voc  | FCN [6]_     | N/A       | 69.4_     | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/segmentation/fcn_resnet50_voc.sh>`_      | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/segmentation/fcn_resnet50_voc.log>`_      |
+-------------------+--------------+-----------+-----------+-----------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------+
| fcn_resnet101_voc | FCN [6]_     | N/A       | 70.9_     | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/segmentation/fcn_resnet101_voc.sh>`_     | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/segmentation/fcn_resnet101_voc.log>`_     |
+-------------------+--------------+-----------+-----------+-----------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------+
| fcn_resnet50_ade  | FCN [6]_     | 78.6      | 38.7      | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/segmentation/fcn_resnet50_ade.sh>`_      | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/segmentation/fcn_resnet50_ade.log>`_      |
+-------------------+--------------+-----------+-----------+-----------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------+
| psp_resnet50_ade  | PSP [9]_     | 78.4      | 41.1      | `shell script <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/segmentation/psp_resnet50_ade.sh>`_      | `log <https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/logs/segmentation/psp_resnet50_ade.log>`_      |
+-------------------+--------------+-----------+-----------+-----------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------------------------------------------------+

.. _69.4:  http://host.robots.ox.ac.uk:8080/anonymous/TC12D2.html
.. _70.9:  http://host.robots.ox.ac.uk:8080/anonymous/FTIQXJ.html


.. [1] He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. \
       "Deep residual learning for image recognition." \
       In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778. 2016.
.. [2] He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. \
       "Identity mappings in deep residual networks." \
       In European Conference on Computer Vision, pp. 630-645. Springer, Cham, 2016.
.. [3] Zagoruyko, Sergey, and Nikos Komodakis. \
       "Wide residual networks." \
       arXiv preprint arXiv:1605.07146 (2016).
.. [4] Zhang, Hongyi, Moustapha Cisse, Yann N. Dauphin, and David Lopez-Paz. \
       "mixup: Beyond empirical risk minimization." \
       arXiv preprint arXiv:1710.09412 (2017).
.. [5] Wei Liu, Dragomir Anguelov, Dumitru Erhan,
       Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.
.. [6] Long, Jonathan, Evan Shelhamer, and Trevor Darrell. \
       "Fully convolutional networks for semantic segmentation." \
       Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.
.. [7] Sandler, Mark, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. \
       "Inverted Residuals and Linear Bottlenecks: Mobile Networks for Classification, Detection and Segmentation." \
       arXiv preprint arXiv:1801.04381 (2018).
.. [8] Xie, Saining, Ross Girshick, Piotr Dollár, Zhuowen Tu, and Kaiming He. \
       "Aggregated residual transformations for deep neural networks." \
       In Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on, pp. 5987-5995. IEEE, 2017.
