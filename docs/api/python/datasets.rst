Vision Datasets
===============
Popular datasets for vision tasks are provided in gluonvision.
By default, we require all datasets reside in ~/.mxnet/datasets/ in order to have
frustration-free user experience and less path-works.

- For small datasets, such as MNIST, MNIST-fashion and CIFAR-10, we provide pre-defined class for out-of-box usage.
- For larger datasets which require significant time for download/extracting, we provide scripts to ease preparation step.

.. note:: Please see :doc:`../../experiments/datasets` to understand how to initialize the datasets.


Pascal VOC
----------
Pascal VOC: http://host.robots.ox.ac.uk/pascal/VOC/ is a vision dataset which

- Provides standardised image data sets for object class recognition
- Provides a common set of tools for accessing the data sets and annotations
- Enables evaluation and comparison of different methods
- Ran challenges evaluating performance on object class recognition (from 2005-2012, now finished)

.. autoclass:: gluonvision.data.VOCDetection
.. autoclass:: gluonvision.data.VOCSegmentation
.. autoclass:: gluonvision.data.VOCAugSegmentation

ImageNet
--------
ImageNet classification dataset is a large scale dataset for image classification, localization, etc..

.. autoclass:: gluonvision.data.ImageNet

DataLoader
----------
.. autoclass:: gluonvision.data.DetectionDataLoader
