"""Prepare Custom Datasets for Object Detection
===============================================

With GluonCV, it is very natural to create a custom dataset of your choice for object detection tasks.

This tutorial is intend to provide you some hints to clear the path for you.

In practice, feel free to choose whatever method that most fit for your use case.
"""

##############################################################################
# Derive from PASCAL VOC format
# =============================
#
import os, zipfile
from gluoncv import utils
fname = utils.download('xx/VOCtemplate.zip', 'VOCtemplate.zip')
with zipfile.ZipFile(fname) as zf:
    zf.extractall('.')

##############################################################################
# A template of VOC-like dataset will have the following structure
#
"""
VOCtemplate
└── VOC2018
    ├── Annotations
    │   └── 000001.xml
    ├── ImageSets
    │   └── Main
    │       └── train.txt
    └── JPEGImages
        └── 000001.jpg
"""

##############################################################################
# And an example of annotation file
with open('VOCtemplate/VOC2018/Annotations/000001.xml', 'r') as fid:
    print(fid.read())

##############################################################################
# As long as your dataset can match the PASCAL VOC convension, it is convenient to
# derive custom dataset from `VOCDetection`
from gluoncv.data import VOCDetection
class VOCLike(VOCDetection):
    CLASSES = ['person', 'dog']
    def __init__(self, root, splits, transform=None, index_map=None, preload_label=True):
        super(VOCLike, self).__init__(root, splits, transform, index_map, preload_label)

dataset = VOCLike(root='VOCtemplate', splits=((2018, 'train'),))
print('length of dataset:', len(dataset))
print('label example:')
print(dataset[0][1])

##############################################################################
# The last column indicate the difficulties of labeled object
# You can ignore it if not necessary in the xml file.
"""<difficult>0</difficult>"""

##############################################################################
# Create GluonCV Object Detection Dataset
# =======================================
#
# Bounding Boxes
# --------------
#
# There are multiple ways to organize the label format for object detection task. We will briefly introduce the
# most widely used: `bounding box`.
#
# GluonCV expect all bounding boxes to be encoded as (xmin, ymin, xmax, ymax), aka (left, top, right, bottom) borders of each object of interest.
#
# First of all, let us plot a real image for example:


import mxnet as mx
import numpy as np
from matplotlib import pyplot as plt

im_fname = utils.download('https://raw.githubusercontent.com/zhreshold/' +
                          'mxnet-ssd/master/data/demo/dog.jpg',
                          path='dog.jpg')
img = mx.image.imread(im_fname)
ax = utils.viz.plot_image(img)
plt.show()


##############################################################################
# Now, let's label the image manually for demo.
#
# In practice, a dedicated GUI labeling tool is more convenient.
#
# We expect all bounding boxes follow this format: (xmin, ymin, xmax, ymax)

dog_label = [130, 220, 320, 530]
bike_label = [115, 120, 580, 420]
car_label = [480, 80, 700, 170]
all_boxes = np.array([dog_label, bike_label, car_label])
all_ids = np.array([0, 1, 2])
class_names = ['dog', 'bike', 'car']

# see how it looks by rendering the boxes into image
ax = utils.viz.plot_bbox(img, all_boxes, labels=all_ids, class_names=class_names)
plt.show()

# Preferred Label Format for GluonCV and MXNet
# --------------------------------------------

from gluoncv.data import ListDetection
lst_dataset = ListDetection('val.lst', root=os.path.expanduser('~/Dev/cache/pikachu/output'))
print('length:', len(lst_dataset))
print('Label example:')
print(lst_dataset[0][1])
