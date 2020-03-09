"""Prepare Multi-Human Parsing V1 dataset.
========================

`Multi-Human Parsing V1 (MHP-v1) <https://github.com/ZhaoJ9014/Multi-Human-Parsing/>`_ is a human-centric containing
5 thousands images annotated with 18 object categories.
This tutorial help you to download MHP-v1 and set it up for later experiments.

.. image:: https://github.com/ZhaoJ9014/Multi-Human-Parsing_MHP/blob/master/Figures/Fig1.png
   :width: 600 px

.. hint::

   You need 850 MB free disk space to download and extract this dataset.
   SSD harddrives are recommended for faster speed.
   The time it takes to prepare the dataset depends on your Internet connection
   and disk speed. For example, it takes around 1 mins on an AWS EC2 instance
   with EBS.

Prepare the dataset
-------------------

We will download and unzip the following files:

+-------------------------------------------------------------------------------------------------------+--------+
| File name                                                                                             | Size   |
+=======================================================================================================+========+
| `LV-MHP-v1.zip <https://drive.google.com/uc?id=1hTS8QJBuGdcppFAr_bvW2tsD9hW_ptr5&export=download>`_  | 850 MB |
+-------------------------------------------------------------------------------------------------------+--------+


The easiest way is to run this script:

 :download:`Download script: mhp_v1.py<../../../scripts/datasets/mhp_v1.py>`

.. code-block:: bash

   python mhp_v1.py

If you have already downloaded the above files and unzipped them,
you can set the folder name through ``--download-dir`` to avoid
downloading them again. For example

.. code-block:: python

   python mhp_v1.py --download-dir ~/.mxnet/datasets/mhp/LV-MHP-v1

"""

################################################################
# How to load the dataset
# -----------------------
#
# Loading images and labels from MHP-v1 is straight-forward
# with GluonCV's dataset utility:

from gluoncv.data import MHPV1Segmentation
train_dataset = MHPV1Segmentation(split='train')
val_dataset = MHPV1Segmentation(split='val')
test_dataset = MHPV1Segmentation(split='test', mode='testval')
print('Training images:', len(train_dataset))
print('Validation images:', len(val_dataset))
print('Testing images:', len(test_dataset))


################################################################
# Get the first sample
# --------------------
#
import numpy as np
img, mask = test_dataset[0]
# get pallete for the mask
from gluoncv.utils.viz import get_color_pallete
mask = get_color_pallete(mask.asnumpy(), dataset='mhpv1')
mask.save('mask.png')


################################################################
# Visualize data and label
# ------------------------
#
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
# subplot 1 for img
fig = plt.figure()
fig.add_subplot(1,2,1)
plt.imshow(img.asnumpy().astype('uint8'))
# subplot 2 for the mask
mmask = mpimg.imread('mask.png')
fig.add_subplot(1,2,2)
plt.imshow(mmask)
# display
plt.show()
