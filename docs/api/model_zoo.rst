.. role:: hidden
    :class: hidden-section

gluoncv.model_zoo
=================

.. automodule:: gluoncv.model_zoo
.. currentmodule:: gluoncv.model_zoo


gluoncv.model_zoo.get_model
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Returns a pre-defined GluonCV model by name.

.. hint::

  This is the recommended method for getting a pre-defined model.

  It support directly loading models from `Gluon Model Zoo <https://mxnet.incubator.apache.org/api/python/gluon/model_zoo.html>`_ as well.

.. autosummary::
    :nosignatures:

    get_model

Image Classification
^^^^^^^^^^^^^^^^^^^^

:hidden:`CIFAR`
~~~~~~~~~~~~~~~

.. autosummary::
    :nosignatures:

    get_cifar_resnet

    cifar_resnet20_v1
    cifar_resnet56_v1
    cifar_resnet110_v1
    cifar_resnet20_v2
    cifar_resnet56_v2
    cifar_resnet110_v2

    get_cifar_wide_resnet

    cifar_wideresnet16_10
    cifar_wideresnet28_10
    cifar_wideresnet40_8

:hidden:`ImageNet`
~~~~~~~~~~~~~~~~~~

We apply dilattion strategy to pre-trained ResNet models (with stride of 8). Please see :class:`gluoncv.model_zoo.SegBaseModel` for how to use it.

.. currentmodule:: gluoncv.model_zoo

.. autosummary::
    :nosignatures:

    ResNetV1b

    resnet18_v1b


    resnet34_v1b


    resnet50_v1b


    resnet101_v1b


    resnet152_v1b

Object Detection
^^^^^^^^^^^^^^^^

:hidden:`SSD`
~~~~~~~~~~~~~

.. currentmodule:: gluoncv.model_zoo

.. autosummary::
    :nosignatures:

    SSD

    get_ssd

    ssd_300_vgg16_atrous_voc

    ssd_300_vgg16_atrous_coco

    ssd_300_vgg16_atrous_custom

    ssd_512_vgg16_atrous_voc

    ssd_512_vgg16_atrous_coco

    ssd_512_vgg16_atrous_custom

    ssd_512_resnet50_v1_voc

    ssd_512_resnet50_v1_coco

    ssd_512_resnet50_v1_custom

    ssd_512_resnet101_v2_voc

    ssd_512_resnet152_v2_voc

    VGGAtrousExtractor

    get_vgg_atrous_extractor
    vgg16_atrous_300
    vgg16_atrous_512

:hidden:`Faster RCNN`
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :nosignatures:

    FasterRCNN

    get_faster_rcnn

    faster_rcnn_resnet50_v1b_voc

    faster_rcnn_resnet50_v1b_coco

    faster_rcnn_resnet50_v1b_custom

:hidden:`YOLOv3`
~~~~~~~~~~~~~~~~

.. autosummary::
    :nosignatures:

    YOLOV3

    get_yolov3

    yolo3_darknet53_voc

    yolo3_darknet53_coco

    yolo3_darknet53_custom


Instance Segmentation
^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: gluoncv.model_zoo

:hidden:`Mask RCNN`
~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :nosignatures:

    MaskRCNN

    get_mask_rcnn

    mask_rcnn_resnet50_v1b_coco

Semantic Segmentation
^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: gluoncv.model_zoo

:hidden:`FCN`
~~~~~~~~~~~~~

.. autosummary::
    :nosignatures:

    FCN

    get_fcn

    get_fcn_resnet50_voc

    get_fcn_resnet101_voc

    get_fcn_resnet101_coco

    get_fcn_resnet50_ade

    get_fcn_resnet101_ade

:hidden:`PSPNet`
~~~~~~~~~~~~~~~~

.. autosummary::
    :nosignatures:

    PSPNet

    get_psp

    get_psp_resnet101_coco

    get_psp_resnet101_voc

    get_psp_resnet50_ade

    get_psp_resnet101_ade


:hidden:`DeepLabV3`
~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :nosignatures:

    DeepLabV3

    get_deeplab

    get_deeplab_resnet101_coco

    get_deeplab_resnet101_voc

    get_deeplab_resnet50_ade

    get_deeplab_resnet101_ade


API Reference
^^^^^^^^^^^^^

Network definitions of GluonCV models

.. currentmodule:: gluoncv.model_zoo

.. automodule:: gluoncv.model_zoo
    :members:
    :imported-members:
