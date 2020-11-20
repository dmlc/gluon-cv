"""3. Extracting video features from pre-trained models
=======================================================

Feature extraction is a very useful tool when you don't have large annotated dataset or don't have the
computing resources to train a model from scratch for your use case. It's also useful to visualize what the model have learned.
In this tutorial, we provide a simple unified solution.
The only thing you need to prepare is a text file containing the information of your videos (e.g., the path to your videos),
we will take care of the rest.
You can extract strong video features from many popular pre-trained models in the GluonCV video model zoo using a single command line.

.. note::

    Feel free to skip the tutorial because the feature extraction script is self-complete and ready to launch.

    :download:`Download Full Python Script: feat_extract_pytorch.py<../../../scripts/action-recognition/feat_extract_pytorch.py>`

    Please checkout the `model_zoo <../model_zoo/index.html#action_recognition>`_ to select your preferred pretrained model.

    ``python feat_extract_pytorch.py --config-file CONFIG``


"""

######################################################################
# Prepare Data
# ------------
#
# Your data can be stored in any hierarchy.
# Just use the format we adopt for training models in the previous tutorial and save the data annotation file as ``video.txt``.
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
# Note that, at this moment, we only support extracting features from videos directly.

######################################################################
# Once you prepare the ``video.txt``, you can start extracting feature by:
#
# ::
#
#     python feat_extract_pytorch.py --config-file ./scripts/action-recognition/configuration/i3d_resnet50_v1_feat.yaml

######################################################################
# The extracted features will be saved to a directory defined in the config file. Each video will have one feature file.
# For example, ``video_001.mp4`` will have a feature named ``i3d_resnet50_v1_kinetics400_video_001_feat.npy``.
# The feature is extracted from the center of the video by using a 32-frames clip.


######################################################################
# There are many other options and other models you can choose,
# e.g., `resnet50_v1b_feat.yaml <https://raw.githubusercontent.com/dmlc/gluon-cv/master/scripts/action-recognition/configuration/resnet50_v1b_feat.yaml>`_,
# `slowfast_4x16_resnet50_feat.yaml <https://raw.githubusercontent.com/dmlc/gluon-cv/master/scripts/action-recognition/configuration/slowfast_4x16_resnet50_feat.yaml>`_,
# `tpn_resnet50_f32s2_feat.yaml <https://raw.githubusercontent.com/dmlc/gluon-cv/master/scripts/action-recognition/configuration/tpn_resnet50_f32s2_feat.yaml>`_,
# `r2plus1d_v1_resnet50_feat.yaml <https://raw.githubusercontent.com/dmlc/gluon-cv/master/scripts/action-recognition/configuration/r2plus1d_v1_resnet50_feat.yaml>`_,
# `i3d_slow_resnet50_f32s2_feat.yaml <https://raw.githubusercontent.com/dmlc/gluon-cv/master/scripts/action-recognition/configuration/i3d_slow_resnet50_f32s2_feat.yaml>`_.
# Try extracting features from these SOTA video models on your own dataset and see which one performs better.
