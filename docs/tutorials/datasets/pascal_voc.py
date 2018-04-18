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

The easiest way is simply running this script:

 :download:`Download Pascal VOC Prepare Script: pascal_voc.py<../../scripts/datasets/pascal_voc.py>`

which will automatically download and extract the data into ``~/.mxnet/datasets/voc``.


.. code-block:: bash

    python pascal_voc.py

.. note::

   You need 8.4 GB disk space to download and extract this dataset. SSD is
   preferred over HDD because of its better performance.

.. note::

   The total time to prepare the dataset depends on your Internet speed and disk
   performance. For example, it often takes 10min on AWS EC2 with EBS.

If you have already downloaded the following required files

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

then you can specify the folder name through ``--dir`` to avoid
download them again.

For example, make sure you have these files exist in ``~/VOCdevkit/downloads``, and you can run

.. code-block:: python

   python pascal_voc.py --dir ~/VOCdevkit

to extract them.

"""

################################################################
# How to load the dataset
# -----------------------
#
# Load image and label from Pascal VOC is quite straight-forward


from gluonvision.data import VOCDetection
train_dataset = VOCDetection(splits=[(2007, 'trainval'), (2012, 'trainval')])
val_dataset = VOCDetection(splits=[(2007, 'test')])
print('Training images:', len(train_dataset))
print('Validation images:', len(val_dataset))

################################################################
# Check the first example
# -----------------------
#
train_image, train_label = train_dataset[0]
bboxes = train_label[:, :4]
cids = train_label[:, 4:5]
print('image size:', train_image.shape)
print('bboxes:', bboxes.shape, 'class ids:', cids.shape)

from matplotlib import pyplot as plt
from gluonvision.utils import viz
ax = viz.plot_bbox(train_image.asnumpy(), bboxes, scores=None, labels=cids, class_names=train_dataset.classes)
plt.show()
