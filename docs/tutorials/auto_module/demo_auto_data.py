"""01. Load web datasets with GluonCV Auto Module
=================================================

This tutorial introduces the basic dataset preprocesses that can be used to
download and load arbitrary custom dataset as long as they follow certain
supported data formats.

The current version supports loading datasets for
- Image Classification(with csv lists and raw images, or folder separated raw images)
- Object Detection(as in Pascal VOC format or COCO json annatations)

Stay tuned for new applications and formats, we are also looking forward to seeing
contributions that brings new formats to GluonCV!

That's enough introduction, let's take a look at how web datasets can be loaded into
a recommended formats supported in GluonCV auto module.
"""

##########################################################
# Image Classification
# -------
#
# Managing the labels of an image classification dataset is pretty simple.
# In this example we show a few ways to organize them.
#
# First of all, we could infer labels from nested folder structure automatically
# like::
#     root/car/0001.jpg
#     root/car/xxxa.jpg
#     root/car/yyyb.jpg
#     root/bus/123.png
#     root/bus/023.jpg
#     root/bus/wwww.jpg
# or even more with train/val/test splits
# like::
#     root/train/car/0001.jpg
#     root/train/car/xxxa.jpg
#     root/train/bus/123.png
#     root/train/bus/023.jpg
#     root/test/car/yyyb.jpg
#     root/test/bus/wwww.jpg
#
# where root is the root folder, `car` and `bus` categories are well organized in
# sub-directories, respectively
from gluoncv.auto.tasks import ImageClassification

##########################################################
#
# We can use `ImageClassification.Dataset` to load dataset from a folder,
# here root can be a local path or url, if it's a url, the archieve file
# will be downloaded and extracted automatically to `~/.gluoncv` by default,
# to change the default behavior, you may edit `~/.gluoncv/config.yml`
#
train, val, test = ImageClassification.Dataset.from_folders(
    'https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip',
    train='train', val='val', test='test', exts=('.jpg', '.jpeg', '.png'))

##########################################################
# train split
print('train', train)

##########################################################
# test split
print('test', test)

##########################################################
# you may notice that the dataset is a pandas DataFrame, which are handy
# and it's okay that certain split is empty, as in this case, `validation` split is empty

print('validation', val)

##########################################################
# you may split the train set to `train` and `val` for training and validation

train, val, _ = train.random_split(val_size=0.1, test_size=0)
print(len(train), len(val))

##########################################################
# In some cases, you may get a raw folder without splits, you may use `from_folders` instead:

dataset = ImageClassification.Dataset.from_folder('https://s3.amazonaws.com/fast-ai-imageclas/oxford-iiit-pet.tgz')

##########################################################
#
print(dataset)

##########################################################
# Visualize Image Classification Dataset
# --------------------------------------
# you may plot the sample images with `show_images`, like:

train.show_images(nsample=16, ncol=4, shuffle=True, fontsize=64)

##########################################################
# Object Detection
# ----------------
# The labels for object detection is a little bit more complicated than image classification,
# addtional information such as bounding box coordinates have to be stored in certain formats.
#
# In GluonCV we support loading from common Pascal VOC and COCO formats.
#
# The key difference between VOC and COCO format is the way how annotations are stored.
#
# For VOC, raw images and annotations are stored in unique directory, where annotations are usually
# per image basis, e.g., `JPEGImages/0001.jpeg` and `Annotations/0001.xml` is a valid image-label pair.
#
# In contrast, COCO format stores all labels in a single annotation file, e.g., all training annotations are
# stored in `instaces_train2017.json`, validation annotations are stored in `instances_val2017.json`.
#
# Other than identifying the valid format of desired dataset, there's not so much different in
# loading the dataset into gluoncv
from gluoncv.auto.tasks import ObjectDetection

##########################################################
# A subset of Pascal VOC
dataset = ObjectDetection.Dataset.from_voc('https://autogluon.s3.amazonaws.com/datasets/tiny_motorbike.zip')

##########################################################
# The dataset is once again a pandas DataFrame
print(dataset)

##########################################################
# The dataset supports random split as well
train, val, test = dataset.random_split(val_size=0.1, test_size=0.1)
print('train', len(train), 'val', len(val), 'test', len(test))

##########################################################
# For object detection, `rois` column is a list of bounding boxes
# in dict, 'image_attr' is optional attributes that can accelerate
# some image pre-processing functions, for example:
print(train.loc[0])

##########################################################
# Visualize Object Detection Dataset
# ----------------------------------
# you may plot the sample images as well as bounding boxes with `show_images`, like:
train.show_images(nsample=16, ncol=4, shuffle=True, fontsize=64)

##########################################################
# Next step
# ---------
# You have access to arbitrary datasets, e.g., kaggle competition datasets,
# you can start training by looking at these tutorials:
# - :ref:`sphx_glr_build_examples_auto_module_train_image_classifier_basic.py`
# - :ref:`sphx_glr_build_examples_auto_module_demo_auto_detection.py`
# You may also check out the`d8 dataset <http://preview.d2l.ai/d8/main/>`_ with built-in datasets.
# D8 datasets is fully compatible with gluoncv.auto, you can directly plug-in datasets loaded from d8 and
# train with `fit` functions.
