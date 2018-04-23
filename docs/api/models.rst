.. role:: hidden
    :class: hidden-section

gluonvision.model_zoo
=====================

.. automodule:: gluonvision.model_zoo
.. currentmodule:: gluonvision.model_zoo

CIFAR
-----

.. autofunction:: get_cifar_resnet

.. autofunction:: cifar_resnet20_v1
.. autofunction:: cifar_resnet56_v1
.. autofunction:: cifar_resnet110_v1
.. autofunction:: cifar_resnet20_v2
.. autofunction:: cifar_resnet56_v2
.. autofunction:: cifar_resnet110_v2

.. autofunction:: get_cifar_wide_resnet

.. autofunction:: cifar_wideresnet16_10
.. autofunction:: cifar_wideresnet28_10
.. autofunction:: cifar_wideresnet40_8



Object Detection
----------------

:hidden:`SSD`
~~~~~~~~~~~~~

.. currentmodule:: gluonvision.model_zoo.ssd

.. autoclass:: SSD
    :members:

.. autofunction:: get_ssd

.. autofunction:: ssd_300_vgg16_atrous_voc

.. autofunction:: ssd_512_vgg16_atrous_voc

.. autofunction:: ssd_512_resnet50_v1_voc

.. autofunction:: ssd_512_resnet101_v2_voc

.. autofunction:: ssd_512_resnet152_v2_voc

.. autoclass:: VGGAtrousExtractor
    :members:

.. autofunction:: get_vgg_atrous_extractor
.. autofunction:: vgg16_atrous_300
.. autofunction:: vgg16_atrous_512

.. currentmodule:: gluonvision.model_zoo

Semantic Segmentation
---------------------

:hidden:`BaseModel`
~~~~~~~~~~~~~~~~~~~

.. autoclass:: segbase.SegBaseModel
    :members:

:hidden:`FCN`
~~~~~~~~~~~~~

.. autoclass:: FCN
    :members:

.. autofunction:: get_fcn

.. autofunction:: get_fcn_voc_resnet50

.. autofunction:: get_fcn_voc_resnet101

Dilated Network
---------------

We apply dilattion strategy to pre-trained ResNet models (with stride of 8). Please see :class:`gluonvision.model_zoo.SegBaseModel` for how to use it.


:hidden:`DilatedResNetV0`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: gluonvision.model_zoo.dilated.dilatedresnetv0

.. autoclass:: DilatedResNetV0
    :members:

.. autofunction:: dilated_resnet18


.. autofunction:: dilated_resnet34


.. autofunction:: dilated_resnet50


.. autofunction:: dilated_resnet101


.. autofunction:: dilated_resnet152


:hidden:`DilatedResNetV2`
~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: gluonvision.model_zoo.dilated.dilatedresnetv2

.. autoclass:: DilatedResNetV2
    :members:

.. autofunction:: get_dilated_resnet


.. autofunction:: dilated_resnet18


.. autofunction:: dilated_resnet34


.. autofunction:: dilated_resnet50


.. autofunction:: dilated_resnet101


.. autofunction:: dilated_resnet152


Common Components
-----------------

:hidden:`Bounding Box`
~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: gluonvision.model_zoo.bbox

.. autoclass:: BBoxCornerToCenter
    :members:

.. autoclass:: BBoxCenterToCorner
    :members:

:hidden:`Coders`
~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: gluonvision.model_zoo.coders

.. autoclass:: NormalizedBoxCenterEncoder
    :members:

.. autoclass:: NormalizedBoxCenterDecoder
    :members:

.. autoclass:: MultiClassEncoder
    :members:

.. autoclass:: MultiClassDecoder
    :members:

.. autoclass:: MultiPerClassDecoder
    :members:


:hidden:`Features`
~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: gluonvision.model_zoo.features

.. autoclass:: FeatureExtractor
    :members:

.. autoclass:: FeatureExpander
    :members:

:hidden:`Losses`
~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: gluonvision.model_zoo.losses

.. autoclass:: FocalLoss
    :members:


:hidden:`Matchers`
~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: gluonvision.model_zoo.matchers

.. autoclass:: CompositeMatcher
    :members:

.. autoclass:: BipartiteMatcher
    :members:

.. autoclass:: MaximumMatcher
    :members:

:hidden:`Predictors`
~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: gluonvision.model_zoo.predictors

.. autoclass:: ConvPredictor
    :members:

.. autoclass:: FCPredictor
    :members:

:hidden:`Samplers`
~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: gluonvision.model_zoo.samplers

.. autoclass:: NaiveSampler
    :members:

.. autoclass:: OHEMSampler
    :members:
