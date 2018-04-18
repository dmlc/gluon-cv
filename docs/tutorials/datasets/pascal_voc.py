"""Prepare PASCAL VOC datasets
==============================

`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ contains a collection of
datasets for object detection. The most commonly adopted version for
benchmarking is using *2007 trainval* and *2012 trainval* for training and *2007
test* for validation.  This tutorial will walk you through the steps for
preparing this dataset to be used by GluonVision.

.. image:: http://host.robots.ox.ac.uk/pascal/VOC/pascal2.png

Prepare the dataset
-------------------

The easiest way is simply downloading
:download:`pascal_voc.py<../../scripts/datasets/pascal_voc.py>` and then running
the following command

.. code-block:: bash

    python pascal_voc.py

which will automatically download and extract the data into ``~/.mxnet/datasets/voc``.


.. note::

   You need 8.4 GB disk space to download and extract this dataset. SSD is
   preferred over HDD because of its better performance.

.. note::

   The total time to prepare the dataset depends on your Internet speed and disk
   performance. For example, it often takes 10 min on AWS EC2 with EBS.

You can skip the download step if if you have already downloaded the following required files

+------------------------------------------------------------------------------------------------------------------------+--------+------------------------------------------+
| Filename                                                                                                               | Size   | SHA-1                                    |
+========================================================================================================================+========+==========================================+
| `VOCtrainval_06-Nov-2007.tar <http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar>`_            | 439 MB | 34ed68851bce2a36e2a223fa52c661d592c66b3c |
+------------------------------------------------------------------------------------------------------------------------+--------+------------------------------------------+
| `VOCtest_06-Nov-2007.tar <http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar>`_                    | 430 MB | 41a8d6e12baa5ab18ee7f8f8029b9e11805b4ef1 |
+------------------------------------------------------------------------------------------------------------------------+--------+------------------------------------------+
| `VOCtrainval_11-May-2012.tar  <http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar>`_           | 1.9 GB | 4e443f8a2eca6b1dac8a6c57641b67dd40621a49 |
+------------------------------------------------------------------------------------------------------------------------+--------+------------------------------------------+
| `benchmark.tgz <http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz>`_   | 1.4 GB | 7129e0a480c2d6afb02b517bb18ac54283bfaa35 |
+------------------------------------------------------------------------------------------------------------------------+--------+------------------------------------------+

then you can specify the folder name through ``--download-dir`` to use the
downloaded files.

For example, assume you downloaded all files into ``~/VOCdevkit/``, and you can run:

.. code-block:: bash

   python pascal_voc.py --download-dir ~/VOCdevkit

"""

################################################################
# How to load the dataset
# -----------------------
#
# Loading images and labels is straight-forward through
# :py:class:`gluonvision.data.VOCDetection`.


from gluonvision import data, utils
from matplotlib import pyplot as plt

train_dataset = data.VOCDetection(splits=[(2007, 'trainval'), (2012, 'trainval')])
val_dataset = data.VOCDetection(splits=[(2007, 'test')])
print('Num of training images:', len(train_dataset))
print('Num of validation images:', len(val_dataset))


################################################################
#
# Now let's visualize one example.

train_image, train_label = train_dataset[5]
bounding_boxes = train_label[:, :4]
class_ids = train_label[:, 4:5]
print('Image size (height, width, RGB):', train_image.shape)
print('Num of objects:', bounding_boxes.shape[0])
print('Bounding boxes (num_boxes, bottom_x, bottom_y, upper_x, upper_y):\n',
      bounding_boxes)
print('Class IDs (num_boxes, ):\n', class_ids)

utils.viz.plot_bbox(train_image.asnumpy(), bounding_boxes, scores=None,
                    labels=class_ids, class_names=train_dataset.classes)
plt.show()


##################################################################
# Finally, to use both ``train_dataset`` and ``val_dataset`` for training, we
# can pass them without data transformations and the batch size into
# :py:class:`gluonvions.data.DetectionDataLoader`, see :download:`train_ssd.py
# <../../scripts/detection/train_ssd.py>` for an example.
