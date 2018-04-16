Vision Model Zoo
================

The GluonVision Model Zoo,
similar to its upstream `Gluon Model Zoo
<https://mxnet.incubator.apache.org/api/python/gluon/model_zoo.html>`_,
provides pre-defined and pre-trained models to help bootstrap computer vision related applications.

Model Zoo API
-------------

.. code-block:: python

    from gluonvision import model_zoo
    cifar_resnet20 = model_zoo.get_model('cifar_resnet20_v1', pretrained=True)
    # a fully pretrained ssd model
    ssd0 = model_zoo.get_model('ssd_300_vgg16_atrous_voc', pretrained=True)
    # load ssd model with pretrained feature extractors
    ssd1 = model_zoo.get_model('ssd_512_vgg16_atrous_voc', pretrained_base=True)
    # load ssd from scratch
    ssd2 = model_zoo.get_model('ssd_512_resnet50_v1_voc', pretrained_base=False)

We recommend using `model_zoo.get_model` for loading pre-defined models, because it provides
name check and suggest you what models are available now.

However, you can still load models by directly instantiate it like

.. code-block:: python

    from gluonvision import model_zoo
    cifar_resnet20 = model_zoo.cifar_resnet20_v1(pretrained=True)


Available models
----------------

We are still in early development stage, more models will be made available for download.

Image Classification
~~~~~~~~~~~~~~~~~~~~

The following table summarizes the available models and there performances in additional to
`Gluon Model Zoo
<https://mxnet.incubator.apache.org/api/python/gluon/model_zoo.html>`_.

Object Detection
~~~~~~~~~~~~~~~~

The following table summarizes the available models and there performances for object detection.

+-------------------------------------+----------+---------+-------+
| Model                               | Dataset  | Input   | Perf  |
+=====================================+==========+=========+=======+
| ssd_300_vgg16_atrous_voc [Liu16]_   | VOC07+12 | 300x300 | 77.52 |
+-------------------------------------+----------+---------+-------+
| ssd_512_vgg16_atrous_voc [Liu16]_   | VOC07+12 | 512x512 |       |
+-------------------------------------+----------+---------+-------+
| ssd_512_resnet50_v1_voc [Liu16]_    | VOC07+12 | 512x512 | 80.25 |
+-------------------------------------+----------+---------+-------+

.. [Liu16] Wei Liu, Dragomir Anguelov, Dumitru Erhan,
       Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.
