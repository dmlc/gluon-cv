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
# We can use `ImageClassification.Dataset` to load dataset from a folder,
# here root can be a local path or url, if it's a url, the archieve file
# will be downloaded and extracted automatically to `~/.gluoncv` by default,
# to change the default behavior, you may edit `~/.gluoncv/config.yml` but this is
# not recommended
train, val, test = ImageClassification.Dataset.from_folders(
    'https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip',
    train='train', val='val', test='test', exts=('.jpg', '.jpeg', '.png'))
print('train', train)
print('test', test)
# you may notice that the dataset is a pandas DataFrame, which are handy
# and it's okay that certain split is empty, as in this case, `validation` split is empty
print('validation', val)
# you may split the train set to `train` and `val` for training and validation
train, val, _ = train.random_split(val_size=0.1, test_size=0)
print(len(train), len(val))
# In some cases, you may get a raw folder without splits, you may use `from_folders` instead:
dataset = ImageClassification.Dataset.from_folder('https://s3.amazonaws.com/fast-ai-imageclas/oxford-iiit-pet.tgz')
