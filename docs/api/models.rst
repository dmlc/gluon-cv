.. role:: hidden
    :class: hidden-section

gluoncv.model_zoo
=====================

.. automodule:: gluoncv.model_zoo
.. currentmodule:: gluoncv.model_zoo

.. autofunction:: get_model

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

.. currentmodule:: gluoncv.model_zoo.ssd

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

.. currentmodule:: gluoncv.model_zoo

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

.. autofunction:: get_fcn_ade_resnet50


:hidden:`ResNetV1b`
-------------------

We apply dilattion strategy to pre-trained ResNet models (with stride of 8). Please see :class:`gluoncv.model_zoo.SegBaseModel` for how to use it.

.. currentmodule:: gluoncv.model_zoo.resnetv1b

.. autoclass:: ResNetV1b
    :members:

.. autofunction:: resnet18_v1b


.. autofunction:: resnet34_v1b


.. autofunction:: resnet50_v1b


.. autofunction:: resnet101_v1b


.. autofunction:: resnet152_v1b


Neural Networks
---------------

:hidden:`Bounding Box`
~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: gluoncv.nn.bbox

.. autoclass:: BBoxCornerToCenter
    :members:

.. autoclass:: BBoxCenterToCorner
    :members:

:hidden:`Coders`
~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: gluoncv.nn.coder

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

.. currentmodule:: gluoncv.nn.feature

.. autoclass:: FeatureExtractor
    :members:

.. autoclass:: FeatureExpander
    :members:

:hidden:`Losses`
~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: gluoncv.loss

.. autoclass:: FocalLoss
    :members:


:hidden:`Matchers`
~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: gluoncv.utils.nn.matcher

.. autoclass:: CompositeMatcher
    :members:

.. autoclass:: BipartiteMatcher
    :members:

.. autoclass:: MaximumMatcher
    :members:

:hidden:`Predictors`
~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: gluoncv.model_zoo.nn.predictor

.. autoclass:: ConvPredictor
    :members:

.. autoclass:: FCPredictor
    :members:

:hidden:`Samplers`
~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: gluoncv.utils.nn.sampler

.. autoclass:: NaiveSampler
    :members:

.. autoclass:: OHEMSampler
    :members:
