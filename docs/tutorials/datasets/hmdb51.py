"""Prepare the HMDB51 Dataset
=============================

`HMDB51 <http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/>`_  is an action recognition dataset,
collected from various sources, mostly from movies, and a small proportion from public databases such as the Prelinger archive,
YouTube and Google videos. The dataset contains 6,766 clips divided into 51 action categories, each containing a minimum of 100 clips.
This tutorial will go through the steps of preparing this dataset for GluonCV.

.. image:: http://serre-lab.clps.brown.edu/wp-content/uploads/2012/08/HMDB_snapshot1.png
   :width: 350 px

.. image:: http://serre-lab.clps.brown.edu/wp-content/uploads/2012/08/HMDB_snapshot2.png
   :width: 350 px

Setup
-----

We need the following two files from HMDB51: the dataset and the official train/test split.

============================================== ======
Filename                                        Size
============================================== ======
hmdb51_org.rar                                 2.1 GB
test_train_splits.rar                          200 KB
============================================== ======

The easiest way to download and unpack these files is to download helper script
:download:`hmdb51.py<../../../scripts/datasets/hmdb51.py>` and run the following command:

.. code-block:: bash

   python hmdb51.py

This script will help you download the dataset, unpack the data from compressed files,
decode the videos to frames, and generate the training files for you. All the files will be
stored at ``~/.mxnet/datasets/hmdb51`` by default. If you want to use more workers to speed up,
please specify ``--num-worker`` to a larger number.

.. note::

   You need at least 60 GB disk space to download and extract the dataset. SSD
   (Solid-state disks) is preferred over HDD because of faster speed.

   You may need to install ``unrar`` by ``sudo apt install unrar``.

   You may need to install ``rarfile``, ``Cython``, ``mmcv`` by ``pip install rarfile Cython mmcv``.

   The data preparation process may take a while. The total time to prepare the dataset depends on
   your Internet speed and disk performance. For example, it takes about 30min on an AWS EC2 instance with EBS.

"""
