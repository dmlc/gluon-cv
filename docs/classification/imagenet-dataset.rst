Prepare the ImageNet dataset
============================

The `ImageNet <http://www.image-net.org/>`_ project contains millions of images and thounds of objects for image classification. It is widely used in the research community to demonstrate that new proposed models are be able to achieve the state-of-the-art results. 

.. image:: https://www.fanyeong.com/wp-content/uploads/2018/01/v2-718f95df083b2d715ee29b018d9eb5c2_r.jpg
   :width: 500 px
   
ImageNet contains multiple datasets. The commonly used version to benchmark image classification models is the dataset used in `ILSVRC 2012 <http://www.image-net.org/challenges/LSVRC/2012/>`_. This tutorial will go through how to prepare the ISVRC2012 dataset to be used by GluonVision. 

.. note:: 
   
   You need at least 300 GB disk space to download and extract the dataset. Solid-state disks (SSD) are prefered because of the better performance on read and write small objects (images).
   
Download the Dataset
--------------------

First to go to the `download page <http://www.image-net.org/download-images>`_ (you need to register an account if you don't have), and find the download link for ILSVRC2012. In the download page, we can 

 Training images (Task 1 & 2). 138GB. MD5: 1d675b47d978889d74fa0da5fadfb00e

 Training images (Task 3). 728MB. MD5: ccaf1013018ac1037801578038d370da

 Validation images (all tasks). 6.3GB. MD5: 29b22e2961454d5413ddabcf34fc5622

 Test images (all tasks). 13GB. MD5: fe64ceb247e473635708aed23ab6d839


Preprocess the Dataset
----------------------


