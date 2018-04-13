Prepare the ImageNet dataset
============================

The `ImageNet <http://www.image-net.org/>`_ project contains millions of images and thounds of objects for image classification. It is widely used in the research community to demonstrate if new proposed models are be able to achieve the state-of-the-art results. 

.. image:: https://www.fanyeong.com/wp-content/uploads/2018/01/v2-718f95df083b2d715ee29b018d9eb5c2_r.jpg
   :width: 500 px
   
The dataset are multiple versions. The commonly used one for image classification is the dataset provided in `ILSVRC 2012 <http://www.image-net.org/challenges/LSVRC/2012/>`_. This tutorial will go through the steps of preparing this dataset to be used by GluonVision. 

.. note:: 
   
   You need at least 300 GB disk space to download and extract the dataset. SSD (Solid-state disks) are prefered over HDD because of the better performance on reading and writing small objects (images). 
   
Download the Dataset
--------------------

First to go to the `download page <http://www.image-net.org/download-images>`_ (you may need to register an account), and then find the download link for ILSVRC2012. Next go to the download page to download the following two files:

======================== ======
Filename                 Size
======================== ======
ILSVRC2012_img_train.tar 138 GB
ILSVRC2012_img_val.tar   6.3 GB
======================== ======

Preprocess the Dataset
----------------------

Assume the two tar files are downloaded in the folder ``~/ILSVRC2012``, then use the following command to preprocess the dataset::

   python examples/datasets/setup_imagenet.py --path ~/ILSVRC2012


(explain a little bit more details here so that users know how to prepare their onw dataset)

