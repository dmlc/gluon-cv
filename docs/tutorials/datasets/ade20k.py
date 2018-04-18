"""Prepare ADE20K dataset.
========================

This script download and prepare the `ADE20K
<http://sceneparsing.csail.mit.edu/>`_ dataset for scene parsing.  It contains
more than 20 thousands scene-centric images annotated with 150 object
categories.

.. image:: http://groups.csail.mit.edu/vision/datasets/ADE20K/assets/images/examples.png
   :width: 600 px

Prepare the dataset
-------------------

The easiest way is simply running this script:

:download:`Download ADE20K Prepare Script: ade20k.py<../../scripts/datasets/ade20k.py>`

which will automatically download and extract the data into ``~/.mxnet/datasets/ade``.

.. code-block:: bash

   python scripts/datasets/ade20k.py

.. note::

   You need 2.3 GB disk space to download and extract this dataset. SSD is
   preferred over HDD because of its better performance.
   The total time to prepare the dataset depends on your Internet speed and disk
   performance. For example, it may take 25 min on AWS EC2 with EBS.


If you have already downloaded the following required files and unziped them, whose URLs can be
obtained from the source codes at the end of this tutorial,

===========================  ======
Filename                     Size
===========================  ======
ADEChallengeData2016.zip     923 MB
release_test.zip             202 MB
===========================  ======

then you can specify the folder name through ``--download-dir`` to avoid
download them again. For example

.. code-block:: python

   python scripts/datasets/ade20k.py --download-dir ~/ade_downloads

How to load the dataset
-----------------------

.. code-block:: python

"""

################################################################
# How to load the dataset
# -----------------------
#
# Load image and label from ADE20K is quite straight-forward

from gluonvision.data import ADE20KSegmentation
train_dataset = ADE20KSegmentation(split='train')
val_dataset = ADE20KSegmentation(split='val')
print('Training images:', len(train_dataset))
print('Validation images:', len(val_dataset))


################################################################
# Visualize the data
# ------------------
#
img, mask = val_dataset[0]
img.show()
print(img)
print(mask)
# get pallete for the mask and show
import numpy as np
from gluonvision.utils.viz import get_color_pallete
mask = get_color_pallete(np.array(mask), dataset='ade20k')
mask.show()
