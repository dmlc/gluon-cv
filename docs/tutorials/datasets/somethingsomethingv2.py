"""Prepare the 20BN-something-something Dataset V2
==================================================

`Something-something-v2 <https://20bn.com/datasets/something-something>`_  is an action recognition dataset
of realistic action videos, collected from YouTube. With 220,847 short trimmed videos
from 174 action categories, it is one of the largest and most widely used dataset in the research
community for benchmarking state-of-the-art video action recognition models. This tutorial
will go through the steps of preparing this dataset for GluonCV.


Download
--------

Please refer to the `official website <https://20bn.com/datasets/something-something>`_ to download the videos.
The video data is provided as one large TGZ archive, split into parts of 1 GB max (there are 20 parts). The total download size is 19.4 GB.
The archive contains webm-files using the VP9 codec. Files are numbered from 1 to 220847.
Please use the provided md5sum to check if your downloaded parts are complete.

============================================== ======
Filename                                        Size
============================================== ======
20bn-something-something-v2-00                  1 GB
20bn-something-something-v2-01                  1 GB
20bn-something-something-v2-02                  1 GB
20bn-something-something-v2-03                  1 GB
20bn-something-something-v2-04                  1 GB
20bn-something-something-v2-05                  1 GB
20bn-something-something-v2-06                  1 GB
20bn-something-something-v2-07                  1 GB
20bn-something-something-v2-08                  1 GB
20bn-something-something-v2-09                  1 GB
20bn-something-something-v2-10                  1 GB
20bn-something-something-v2-11                  1 GB
20bn-something-something-v2-12                  1 GB
20bn-something-something-v2-13                  1 GB
20bn-something-something-v2-14                  1 GB
20bn-something-something-v2-15                  1 GB
20bn-something-something-v2-16                  1 GB
20bn-something-something-v2-17                  1 GB
20bn-something-something-v2-18                  1 GB
20bn-something-something-v2-19                 445 MB
============================================== ======

Once confirmed, you can use the following command to unzip the videos.

.. code-block:: bash

   cat 20bn-something-something-v2-?? | tar zx

Suppose by default the root directory for your data is ``ROOT=~/.mxnet/datasets/somethingsomethingv2``,
all the videos will be stored at ``ROOT/20bn-something-something-v2`` now.
Then, download the annotations and put them into folder ``ROOT/annotations``.

============================================== ======
Filename                                        Size
============================================== ======
something-something-v2-labels.json               9 KB
something-something-v2-train.json               26 MB
something-something-v2-validation.json         3.7 MB
something-something-v2-test.json               448 KB
============================================== ======


Preprocess
----------

The easiest way to prepare the dataset is to download helper script
:download:`somethingsomethingv2.py<../../../scripts/datasets/somethingsomethingv2.py>` and run the following command:

.. code-block:: bash

   python somethingsomethingv2.py

This script will help you decode the videos to raw frames and generate training files for standard data loading.
The video frames will be saved at ``ROOT/20bn-something-something-v2-frames``. The training files will be
saved at ``ROOT/annotations``. The data preparation process may take a while. The total time to prepare
the dataset depends on your machine. For example, it takes about 6 hours on an AWS EC2 instance with EBS.

Once the script is done, you can start training your action recognition models on something-something-v2 dataset.
"""
