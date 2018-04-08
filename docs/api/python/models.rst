.. role:: hidden
    :class: hidden-section

Vision Models
=============

.. automodule:: gluonvision.model_zoo
.. currentmodule:: gluonvision.model_zoo

CIFAR
-----

.. currentmodule:: gluonvision.model_zoo.cifar.resnet

.. autoclass:: ResNetV1
    :members:

.. autoclass:: ResNetV2
    :members:

.. autofunction:: get_resnet

.. currentmodule:: gluonvision.model_zoo

Dilated Network
---------------

We apply dilattion strategy to pre-trained ResNet models (with stride of 8). Please see :class:`gluonvision.model_zoo.SegBaseModel` for how to use it.

.. autoclass:: Dilated_ResNetV2
    :members:

.. autoclass:: DilatedBasicBlockV2
    :members:

.. autoclass:: DilatedBottleneckV2
    :members:

.. autofunction:: get_dilated_resnet


.. autofunction:: dilated_resnet18


.. autofunction:: dilated_resnet34


.. autofunction:: dilated_resnet50


.. autofunction:: dilated_resnet101


.. autofunction:: dilated_resnet152

Object Detection
----------------

:hidden:`SSD`
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SSD
    :members:

Semantic Segmentation
---------------------

:hidden:`SegBaseModel`
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: SegBaseModel
    :members:

:hidden:`FCN`
~~~~~~~~~~~~~

.. autoclass:: FCN
    :members:

:hidden:`PSPNet`
~~~~~~~~~~~~~~~~

.. autoclass:: PSPNet
    :members:


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
