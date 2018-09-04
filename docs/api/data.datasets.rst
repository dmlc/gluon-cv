gluoncv.data
================

.. automodule:: gluoncv.data
.. currentmodule:: gluoncv.data


.. hint::

   Please refer to :doc:`../build/examples_datasets/index` for the description
   of the datasets listed in this page, and how to download and extract them.

.. hint::

   For small dataset such as MNIST and CIFAR10, please refer to `GluonCV
   Datasets
   <https://mxnet.incubator.apache.org/api/python/gluon/data.html#vision-datasets>`_,
   which can be used directly without any downloading step.


`ImageNet <http://www.image-net.org/>`_
---------------------------------------

.. autosummary::
    :nosignatures:

    gluoncv.data.ImageNet

`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_
-------------------------------------------------------

.. autosummary::
    :nosignatures:

    gluoncv.data.VOCDetection
    gluoncv.data.VOCSegmentation
    gluoncv.data.VOCAugSegmentation

`COCO <http://cocodataset.org>`_
--------------------------------

.. autosummary::
    :nosignatures:

    gluoncv.data.COCODetection
    gluoncv.data.COCOInstance


`ADE20K <http://groups.csail.mit.edu/vision/datasets/ADE20K/>`_
---------------------------------------------------------------

.. autosummary::
    :nosignatures:

    gluoncv.data.ADE20KSegmentation


Customized Dataset
------------------

.. autosummary::
    :nosignatures:

    gluoncv.data.LstDetection
    gluoncv.data.RecordFileDetection

API Reference
-------------

.. autoclass:: gluoncv.data.ImageNet
.. autoclass:: gluoncv.data.VOCDetection
.. autoclass:: gluoncv.data.VOCSegmentation
.. autoclass:: gluoncv.data.VOCAugSegmentation
.. autoclass:: gluoncv.data.COCODetection
.. autoclass:: gluoncv.data.COCOInstance
.. autoclass:: gluoncv.data.ADE20KSegmentation
.. autoclass:: gluoncv.data.DetectionDataLoader
.. autoclass:: gluoncv.data.LstDetection
.. autoclass:: gluoncv.data.RecordFileDetection
