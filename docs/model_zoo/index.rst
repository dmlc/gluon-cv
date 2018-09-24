.. _gluoncv-model-zoo:

GluonCV Model Zoo
================

GluonCV Model Zoo provides pre-defined and pre-trained models to help bootstrap computer vision
applications.

Available Models
---------------------------

Please visit pages for tasks:

.. toctree::
   :maxdepth: 1
   :caption: Model Zoo

  classification
  detection
  segmentation

GluonCV is still under development, stay tuned for more models!

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
pre-defined models, because it provides name checking and lists available choices.

However, you can still load models by directly instantiate it like

.. code-block:: python

    from gluoncv import model_zoo
    cifar_resnet20 = model_zoo.cifar_resnet20_v1(pretrained=True)

.. hint::

  Detailed ``model_zoo`` APIs are available in API reference: :py:meth:`gluoncv.model_zoo`.
