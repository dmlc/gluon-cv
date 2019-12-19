"""8. Extracting features from I3D models
=========================================

This is a video action recognition tutorial using Gluon CV toolkit, a step-by-step example.
The readers should have basic knowledge of deep learning and should be familiar with Gluon API.
New users may first go through `A 60-minute Gluon Crash Course <http://gluon-crash-course.mxnet.io/>`_.
You can `Start Training Now`_ or `Dive into Deep`_.

Feature extraction is a very useful tool to have when you don't have large annotated dataset or don't have the
computing resources to train a model for your use cases. In this tutorial, we provide a simple solution.
The only thing you need to prepare is a text file containing the video paths, we will take care of the rest.
You can extract all kinds of features from many popular pre-trained models using a single command line.

.. note::

    Feel free to skip the tutorial because the feature extraction script is self-complete and ready to launch.

    :download:`Download Full Python Script: feat_extract.py<../../../scripts/action-recognition/feat_extract.py>`

    For more command options, please run ``python feat_extract.py -h``
    Please checkout the `model_zoo <../model_zoo/index.html#action_recognition>`_ to select your preferred pretrained model.


"""

######################################################################
# Prepare Data
# ------------
#
# Your data can be stored in any hierarchy.
# The only thing you need to prepare is a text file, ``video.txt``, which should look like
#
# ::
#
#     /home/ubuntu/your_data/video_001.mp4
#     /home/ubuntu/your_data/video_001.mp4
#     /home/ubuntu/your_data/video_002.mp4
#     /home/ubuntu/your_data/video_003.mp4
#     /home/ubuntu/your_data/video_004.mp4
#     ......
#     /home/ubuntu/your_data/video_100.mp4
#
# Each line is the path to each video you want to extract features from.
#
# Or you can also use the format we used for training models in other tutorials,
# ::
#
#     /home/ubuntu/your_data/video_001.mp4 200 0
#     /home/ubuntu/your_data/video_001.mp4 300 1
#     /home/ubuntu/your_data/video_002.mp4 100 2
#     /home/ubuntu/your_data/video_003.mp4 400 2
#     /home/ubuntu/your_data/video_004.mp4 200 1
#     ......
#     /home/ubuntu/your_data/video_100.mp4.100 3
#
# Each line has three things, the path to each video, the number of video frames and the video label.
# However, the second and third things are not gonna used in the code, they are just a placeholder.
# So you can put any postive number in these two places.
#

######################################################################
# Once you prepare the ``video.txt``, you can start extracting feature by:
#
# ::
#
#     python feat_extract.py --data-list video.txt --model i3d_resnet50_v1_kinetics400 --save-dir ./features

######################################################################
# The extracted features will be saved to the ``features`` directory. Each video will have one feature file.
# For example, ``video_001.mp4`` will have a feature named ``i3d_resnet50_v1_kinetics400_video_001.mp4_feat.npy``.
# The feature is extracted from the center of the video by using a 32-frames clip.

######################################################################
# If you want a stronger feature by covering more temporal information. For example, you want to extract features from
# 10 segments of the video and combine them. You can do
#
# ::
#
#     python feat_extract.py --data-list video.txt --model i3d_resnet50_v1_kinetics400 --save-dir ./features --num-segments 10

######################################################################
# If you you want to extract features from 10 segments of the video, select 64-frame clip from each segment,
# and combine them. You can do
#
# ::
#
#     python feat_extract.py --data-list video.txt --model i3d_resnet50_v1_kinetics400 --save-dir ./features --num-segments 10 --new-length 64
#

######################################################################
# If you you want to extract features from 10 segments of the video, select 64-frame clip from each segment,
# perform three-cropping technology, and combine them. You can do
#
# ::
#
#     python feat_extract.py --data-list video.txt --model i3d_resnet50_v1_kinetics400 --save-dir ./features --num-segments 10 --new-length 64 --three-crop

######################################################################
# There are also many other options and other models you can choose, please check ``feat_extract.py`` for more usage information.
