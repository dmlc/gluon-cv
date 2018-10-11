.. _gluoncv-model-zoo:

GluonCV Model Zoo
================

GluonCV Model Zoo provides pre-defined and pre-trained models to help bootstrap computer vision
applications.

Check out different tasks that are divided into separate pages:

.. toctree::
   :maxdepth: 1
   :caption: Model Zoo

  classification

  detection

  segmentation

GluonCV is still under rapid development, stay tuned for more models!

Model Zoo API
-------------

We may frequently update the pretrained weights.
To reproduce previous results with earlier weights, please set the hashtag
for the weights to ``pretrained``.

.. code-block:: python

    from gluoncv import model_zoo
    # load a ResNet model trained on CIFAR10
    cifar_resnet20 = model_zoo.get_model('cifar_resnet20_v1', pretrained=True)
    # load a MobileNet model trained on ImageNet
    mobilenet = model_zoo.get_model('mobilenet1.0', pretrained='efbb2ca3')
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
