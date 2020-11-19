"""2. Fine-tuning SOTA video models on your own dataset
=======================================================

Fine-tuning is an important way to obtain good video models on your own data when you don't have large annotated dataset or don't have the
computing resources to train a model from scratch for your use case.
In this tutorial, we provide a simple unified solution.
The only thing you need to prepare is a text file containing the information of your videos (e.g., the path to your videos),
we will take care of the rest.
You can start fine-tuning from many popular pre-trained models (e.g., I3D, R2+1D, SlowFast and TPN) using a single command line.


"""

######################################################################
# Custom DataLoader
# ------------------
#
# The first and only thing you need to prepare is the data annotation files ``train.txt`` and ``val.txt``.
# We provide a general dataloader for you to use on your own dataset.
# Your data can be stored in any hierarchy, and the ``train.txt`` should look like:
#
# ::
#
#     video_001.mp4 200 0
#     video_001.mp4 200 0
#     video_002.mp4 300 0
#     video_003.mp4 100 1
#     video_004.mp4 400 2
#     ......
#     video_100.mp4 200 10
#
# As you can see, there are three items in each line, separated by spaces.
# The first item is the path to your training videos, e.g., video_001.mp4.
# The second item is the number of frames in each video. But you can put any number here
# because our video loader will compute the number of frames again automatically during training.
# The third item is the label of that video, e.g., 0.
# ``val.txt`` looks the same as ``train.txt`` in terms of format.
#
#
# Once you prepare the ``train.txt`` and ``val.txt``, you are good to go.
# In this tutorial, we will use I3D model and Something-something-v2 dataset as an example.
# Suppose you have Something-something-v2 dataset and you don't want to train an I3D model from scratch.
# First, prepare the data anotation files as mentioned above.
# Second, follow this configuration file `i3d_resnet50_v1_custom.yaml <https://raw.githubusercontent.com/dmlc/gluon-cv/master/scripts/action-recognition/configuration/i3d_resnet50_v1_custom.yaml>`_.
# Specifically, you just need to change the data paths and number of classes in that yaml file.
#
# ::
#
#     TRAIN_ANNO_PATH: '/home/ubuntu/data/sthsthv2/sthsthv2_train.txt'
#     VAL_ANNO_PATH: '/home/ubuntu/data/sthsthv2/sthsthv2_val.txt'
#     TRAIN_DATA_PATH: '/home/ubuntu/data/sthsthv2/20bn-something-something-v2/'
#     VAL_DATA_PATH:  '/home/ubuntu/data/sthsthv2/20bn-something-something-v2/'
#     NUM_CLASSES: 174
#
# If you want to tune other parameters, it is also easy to do.
# Change the learning rate, batch size, clip lenght according to your use cases. Usually a small learning rate is preferred since the model initialization is decent.


######################################################################
# We also support finetuning on other models, e.g.,
# `resnet50_v1b_custom.yaml <https://raw.githubusercontent.com/dmlc/gluon-cv/master/scripts/action-recognition/configuration/resnet50_v1b_custom.yaml>`_,
# `slowfast_4x16_resnet50_custom.yaml <https://raw.githubusercontent.com/dmlc/gluon-cv/master/scripts/action-recognition/configuration/slowfast_4x16_resnet50_custom.yaml>`_,
# `tpn_resnet50_f32s2_custom.yaml <https://raw.githubusercontent.com/dmlc/gluon-cv/master/scripts/action-recognition/configuration/tpn_resnet50_f32s2_custom.yaml>`_,
# `r2plus1d_v1_resnet50_custom.yaml <https://raw.githubusercontent.com/dmlc/gluon-cv/master/scripts/action-recognition/configuration/r2plus1d_v1_resnet50_custom.yaml>`_,
# `i3d_slow_resnet50_f32s2_custom.yaml <https://raw.githubusercontent.com/dmlc/gluon-cv/master/scripts/action-recognition/configuration/i3d_slow_resnet50_f32s2_custom.yaml>`_.
# Try fine-tuning these SOTA video models on your own dataset and see how it goes.
#
# If you would like to extract good video features on your datasets,
# feel free to read the next `tutorial on feature extraction <extract_feat.html>`__.
