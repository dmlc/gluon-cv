"""Prepare Cityscapes dataset.
========================

`Cityscapes <http://sceneparsing.csail.mit.edu/>`_  focuses on semantic understanding of
urban street scenes. This tutorial help you to download Cityscapes and set it up for later experiments.

.. image:: https://www.cityscapes-dataset.com/wordpress/wp-content/uploads/2015/07/stuttgart02-2040x500.png
   :width: 600 px


Prepare the dataset
-------------------


Please login and download the files `gtFine_trainvaltest.zip` and `leftImg8bit_trainvaltest.zip` to
the current folder:

+---------------------------------------+--------+
| File name                             | Size   |
+=======================================+========+
| gtFine_trainvaltest.zip               | 253 MB |
+---------------------------------------+--------+
| leftImg8bit_trainvaltest.zip          | 12 GB  |
+---------------------------------------+--------+

Then run this script:

 :download:`Download script: cityscapes.py<../../../scripts/datasets/cityscapes.py>`

.. code-block:: bash

   python cityscapes.py

"""

################################################################
# How to load the dataset
# -----------------------
#
# Loading images and labels from Cityscapes is straight-forward
# with GluonCV's dataset utility:

from gluoncv.data import CitySegmentation
train_dataset = CitySegmentation(split='train')
val_dataset = CitySegmentation(split='val')
print('Training images:', len(train_dataset))
print('Validation images:', len(val_dataset))


################################################################
# Get the first sample
# --------------------
#
import numpy as np
img, mask = val_dataset[0]
# get pallete for the mask
from gluoncv.utils.viz import get_color_pallete
mask = get_color_pallete(mask.asnumpy(), dataset='citys')
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

