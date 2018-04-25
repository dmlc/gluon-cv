gluoncv.data
================

.. automodule:: gluoncv.data
.. currentmodule:: gluoncv.data


.. hint::

   Please refer to :doc:`../build/examples_datasets/index` for the description
   of the datasets listed in this page, and how to download and extract them.

.. hint::

   For small dataset such as MNIST and CIFAR10, please refer to `Gluon Vision
   Datasets
   <https://mxnet.incubator.apache.org/api/python/gluon/data.html#vision-datasets>`_,
   which can be used directly without any downloading step.


`ImageNet <http://www.image-net.org/>`_
---------------------------------------

.. autoclass:: gluoncv.data.ImageNet

`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_
-------------------------------------------------------

.. autoclass:: gluoncv.data.VOCDetection
.. autoclass:: gluoncv.data.VOCSegmentation
.. autoclass:: gluoncv.data.VOCAugSegmentation


`ADE20K <http://groups.csail.mit.edu/vision/datasets/ADE20K/>`_
---------------------------------------------------------------

.. autoclass:: gluoncv.data.ADE20KSegmentation

DataLoader
----------
.. autoclass:: gluoncv.data.DetectionDataLoader
