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

Once download is complete, please rename the folder names (since class names have white space and parenthese) for ease of processing. Suppose the videos are
downloaded to ``~/.mxnet/datasets/kinetics400``, there will be three folders in it: ``annotations``, ``train`` and ``val``. You can use the following command to
rename the folder names:

.. code-block:: bash

   # sudo apt-get install detox
   detox -r train/
   detox -r val/

Decode into frames
------------------

If you decide to train your model using video data directly, you can skip this section.

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

#########################################################################
# Read with GluonCV
# -----------------
#
# The prepared dataset can be loaded with utility class :py:class:`gluoncv.data.Kinetics400` directly.
# In this tutorial, we provide three examples to read data from the dataset,
# (1) load one frame per video;
# (2) load one clip per video, the clip contains five consecutive frames;
# (3) load three clips evenly per video, each clip contains 12 frames.


#########################################################################
# We first show an example that randomly reads 5 videos each time, randomly selects one frame per video and
# performs center cropping.

import os
from gluoncv.data import Kinetics400
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import transforms
from gluoncv.data.transforms import video

transform_train = transforms.Compose([
    video.VideoCenterCrop(size=224),
    video.VideoToTensor()
])

# Default location of the data is stored on ~/.mxnet/datasets/kinetics400.
# You need to specify ``setting`` and ``root`` for Kinetics400 if you decoded the video frames into a different folder.
train_dataset = Kinetics400(train=True, transform=transform_train)
train_data = DataLoader(train_dataset, batch_size=5, shuffle=True)

#########################################################################
# We can see the shape of our loaded data as below. ``extra`` indicates if we select multiple crops or multiple segments
# from a video. Here, we only pick one frame per video, so the ``extra`` dimension is 1.
for x, y in train_data:
    print('Video frame size (batch, extra, channel, height, width):', x.shape)
    print('Video label:', y.shape)
    break

#########################################################################
# Here is the second example that randomly reads 25 videos each time, randomly selects one clip per video and
# performs center cropping. A clip can contain N consecutive frames, e.g., N=5.

train_dataset = Kinetics400(train=True, new_length=5, transform=transform_train)
train_data = DataLoader(train_dataset, batch_size=5, shuffle=True)

#########################################################################
# We can see the shape of our loaded data as below. Now we have another ``depth`` dimension which
# indicates how many frames in each clip (a.k.a, the temporal dimension).
for x, y in train_data:
    print('Video frame size (batch, extra, channel, depth, height, width):', x.shape)
    print('Video label:', y.shape)
    break

#######################################################################################
# Let's plot one training sample with 5 consecutive video frames. index 0 is image, 1 is label
from matplotlib import pyplot as plt
# subplot 1 for video frame 1
fig = plt.figure()
fig.add_subplot(1,5,1)
frame1 = train_dataset[150][0][0,:,0,:,:].transpose((1,2,0)).asnumpy()*255.0
plt.imshow(frame1.astype('uint8'))
# subplot 2 for video frame 2
fig.add_subplot(1,5,2)
frame2 = train_dataset[150][0][0,:,1,:,:].transpose((1,2,0)).asnumpy()*255.0
plt.imshow(frame2.astype('uint8'))
# subplot 3 for video frame 3
fig.add_subplot(1,5,3)
frame3 = train_dataset[150][0][0,:,2,:,:].transpose((1,2,0)).asnumpy()*255.0
plt.imshow(frame3.astype('uint8'))
# subplot 4 for video frame 4
fig.add_subplot(1,5,4)
frame4 = train_dataset[150][0][0,:,3,:,:].transpose((1,2,0)).asnumpy()*255.0
plt.imshow(frame4.astype('uint8'))
# subplot 5 for video frame 5
fig.add_subplot(1,5,5)
frame5 = train_dataset[150][0][0,:,4,:,:].transpose((1,2,0)).asnumpy()*255.0
plt.imshow(frame5.astype('uint8'))
# display
plt.show()

#########################################################################
# The last example is that we randomly read 5 videos each time, select 3 clips evenly per video and
# performs center cropping. A clip contains 12 consecutive frames.

train_dataset = Kinetics400(train=True, new_length=12, num_segments=3, transform=transform_train)
train_data = DataLoader(train_dataset, batch_size=5, shuffle=True)

#########################################################################
# We can see the shape of our loaded data as below. Now the ``extra`` dimension is 3, which indicates we
# have three segments for each video.
for x, y in train_data:
    print('Video frame size (batch, extra, channel, depth, height, width):', x.shape)
    print('Video label:', y.shape)
    break

#########################################################################
# Read with VideoLoader
# ---------------------
#
# In case you don't want to decode videos into frames, we provide a fast video loader, `Decord <https://github.com/zhreshold/decord>`_, to read the dataset.
# We still use the utility class :py:class:`gluoncv.data.Kinetics400`. The usage is similar to using image loader.

#########################################################################
# For example, if we want to randomly read 5 videos, randomly selects one frame per video and
# performs center cropping.

from gluoncv.utils.filesystem import import_try_install
import_try_install('decord')

# Since we are loading videos directly, we need to change the ``root`` location.
# ``tiny_train_videos`` contains a small subset of Kinetics400 dataset, which is used for demonstration only.
train_dataset = Kinetics400(root=os.path.expanduser('~/.mxnet/datasets/kinetics400/tiny_train_videos'), train=True,
                            transform=transform_train, video_loader=True, use_decord=True)
train_data = DataLoader(train_dataset, batch_size=5, shuffle=True)

#########################################################################
# We can see the shape of our loaded data as below.
for x, y in train_data:
    print('Video frame size (batch, extra, channel, height, width):', x.shape)
    print('Video label:', y.shape)
    break

#########################################################################
# Here is the second example that randomly reads 25 videos each time, randomly selects one clip per video and
# performs center cropping. A clip can contain N consecutive frames, e.g., N=5.

train_dataset = Kinetics400(root=os.path.expanduser('~/.mxnet/datasets/kinetics400/tiny_train_videos'),
                            train=True, new_length=5, transform=transform_train,
                            video_loader=True, use_decord=True)
train_data = DataLoader(train_dataset, batch_size=5, shuffle=True)

#########################################################################
# We can see the shape of our loaded data as below.
for x, y in train_data:
    print('Video frame size (batch, extra, channel, depth, height, width):', x.shape)
    print('Video label:', y.shape)
    break

#########################################################################
# The last example is that we randomly read 5 videos each time, select 3 clips evenly per video and
# performs center cropping. A clip contains 12 consecutive frames.

train_dataset = Kinetics400(root=os.path.expanduser('~/.mxnet/datasets/kinetics400/tiny_train_videos'), train=True,
                            new_length=12, num_segments=3, transform=transform_train,
                            video_loader=True, use_decord=True)
train_data = DataLoader(train_dataset, batch_size=5, shuffle=True)

#########################################################################
# We can see the shape of our loaded data as below.
for x, y in train_data:
    print('Video frame size (batch, extra, channel, depth, height, width):', x.shape)
    print('Video label:', y.shape)
    break

#########################################################################
# We also support other video loaders, e.g., OpenCV VideoReader, but Decord is significantly faster.
# We refer the users to read the documentations for more information.
