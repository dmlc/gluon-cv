"""Prepare the UCF101 dataset
============================

`UCF101 <https://www.crcv.ucf.edu/data/UCF101.php>`_  is an action recognition dataset
of realistic action videos, collected from YouTube. With 13,320 short trimmed videos
from 101 action categories, it is one of the most widely used dataset in the research
community for benchmarking state-of-the-art video action recognition models. This tutorial
will go through the steps of preparing this dataset for GluonCV.

.. image:: https://www.crcv.ucf.edu/data/UCF101/UCF101.jpg
   :width: 500 px

.. note::

   You need at least 60 GB disk space to download and extract the dataset. SSD
   (Solid-state disks) is preferred over HDD because of faster speed.

Download
--------

First, go to the `UCF101 webpage <https://www.crcv.ucf.edu/data/UCF101.php>`_
, and find the links to download the dataset and the official train/test
split for action recognition.

============================================== ======
Filename                                        Size
============================================== ======
UCF101.rar                                     6.5 GB
UCF101TrainTestSplits-RecognitionTask.zip      114 KB
============================================== ======

If you prefer to use command lines to download the dataset, the instruction is as below:

.. code-block:: bash

   wget https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
   wget https://www.crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip

Setup
-----

First, extract the data from the compressed files,

.. code-block:: bash

   unrar x UCF101.rar
   unzip UCF101TrainTestSplits-RecognitionTask.zip

.. note::
   You may need to install `unrar` by `sudo apt install unrar`.

Then, download the helper script
:download:`ucf101.py<../../../scripts/datasets/ucf101.py>`. We can first use it to decode the videos to frames,

.. code-block:: bash

   python ucf101.py --decode_video --src_dir VIDEO_PATH --out_dir FRAME_PATH

.. note::
   VIDEO_PATH is where you store your downloaded videos, e.g., ./UCF-101
   FRAME_PATH is where you want to store the decoded video frames, e.g., ./rawframes
   You may need to install `Cython` and `mmcv` by `pip install Cython mmcv`.
   Extracting the images may take a while. For example, it takes
   about 15min on an AWS EC2 instance with EBS.

Once we have the video frames, we need to generate training files according to
the train/test split contained in the `ucfTrainTestlist` folder.

.. code-block:: bash

   python ucf101.py --build_file_list --anno_dir ANNOTATION_PATH --frame_path FRAME_PATH --out_list_path OUT_path --shuffle

.. note::
   ANNOTATION_PATH is where you store your annotation files, e.g., ./ucfTrainTestlist
   FRAME_PATH is where you store the decoded video frames, e.g., ./rawframes
   OUT_path is where you want to store your generated training files, e.g., ./ucfTrainTestlist

Take a quick example, the generated training file will look like this,
.. code-block:: bash

   Typing/v_Typing_g16_c02 251 94
   ApplyEyeMakeup/v_ApplyEyeMakeup_g25_c06 109 0
   IceDancing/v_IceDancing_g14_c02 256 43
   PlayingDhol/v_PlayingDhol_g15_c03 187 60
   TableTennisShot/v_TableTennisShot_g11_c03 135 89

.. note::
   First column indicates the path to the video.
   Second column indicates the number of frames in that video.
   Third column indicates the label of that video.


Read with GluonCV
-----------------

The prepared dataset can be loaded with utility class :py:class:`gluoncv.data.ucf101`
directly. Here is an example that randomly reads 25 images each time and
performs center cropping.
"""


from gluoncv.data import ucf101
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import transforms
from gluoncv.data.transforms import video

transform_train = transforms.Compose([
    video.VideoCenterCrop(size=224),
])

# Default location of the data is stored on ~/.mxnet/datasets/ucf101
# You need to specify ``setting`` and ``root`` for UCF101 if you decoded the video frames into a different folder.
train_dataset = ucf101.classification.UCF101(train=True, transform=transform_train)
train_data = DataLoader(train_dataset, batch_size=25, shuffle=True)

#########################################################################
for x, y in train_data:
    print(x.shape, y.shape)
    break

#########################################################################
# Plot some validation images
from gluoncv.utils import viz
viz.plot_image(train_dataset[0][0])  # index 0 is image, 1 is label
