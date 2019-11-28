"""Prepare the Kinetics400 dataset
==================================

`Kinetics400 <https://deepmind.com/research/open-source/kinetics>`_  is an action recognition dataset
of realistic action videos, collected from YouTube. With 306,245 short trimmed videos
from 400 action categories, it is one of the largest and most widely used dataset in the research
community for benchmarking state-of-the-art video action recognition models. This tutorial
will go through the steps of preparing this dataset for GluonCV.


Download
--------

Please refer to the `official website <https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics>`_ on how to download the videos.
Note that the downloaded videos will consume about 450G disk space, make sure there is enough space before downloading. The crawling process can take several days.

Once download is complete, please rename the folder names (since class names have white space) for ease of processing. Suppose the videos are
downloaded to ``~/.mxnet/datasets/kinetics400``, there will be three folders in it: ``annotations``, ``train`` and ``val``. You can use the following command to
rename the folder names:

.. code-block:: bash

   # sudo apt-get install detox
   detox -r train/
   detox -r val/

Decode into frames
------------------

The easiest way to prepare the videos in frames format is to download helper script
:download:`kinetics400.py<../../../scripts/datasets/kinetics400.py>` and run the following command:

.. code-block:: bash

   python kinetics400.py --src_dir ~/.mxnet/datasets/kinetics400/train --out_dir ~/.mxnet/datasets/kinetics400/rawframes_train --decode_video --new_width 450 --new_height 340
   python kinetics400.py --src_dir ~/.mxnet/datasets/kinetics400/val --out_dir ~/.mxnet/datasets/kinetics400/rawframes_val --decode_video --new_width 450 --new_height 340

This script will help you decode the videos to raw frames. We specify the width and height of frames for resizing because this will save lots of disk space without losing much accuracy.
All the resized frames will consume 2.9T disk space. If we don't specify the dimension, the original decoded frames will consume 6.8T disk space.
The data preparation process may take a while. The total time to prepare the dataset depends on your machine. For example, it takes about 8 hours on an AWS EC2 instance with EBS and using 56 workers.


Generate the training files
---------------------------

The last step is to generate training files for standard data loading. You can run the helper script as:

.. code-block:: bash

   python kinetics400.py --build_file_list --frame_path ~/.mxnet/datasets/kinetics400/rawframes_train --subset train --shuffle
   python kinetics400.py --build_file_list --frame_path  ~/.mxnet/datasets/kinetics400/rawframes_val --subset val --shuffle

Now you can start training your action recognition models on Kinetics400 dataset.

.. note::

   You need at least 4T disk space to download and extract the dataset. SSD
   (Solid-state disks) is preferred over HDD because of faster speed.

   You may need to install ``Cython`` and ``mmcv`` by ``pip install Cython mmcv``.
"""
