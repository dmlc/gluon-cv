"""9. Inference on your own videos using pre-trained models
===========================================================

In this tutorial, we provide a script for you to make human activity predictions on your own videos.
The only thing you need to prepare is a text file containing the information of your videos (e.g., the path to your videos),
we will take care of the rest.
You can use many popular pre-trained models (e.g., I3D, I3D-nonlocal, SlowFast) in a single command line.

.. note::

    Feel free to skip the tutorial because the inference script is self-complete and ready to launch.

    :download:`Download Full Python Script: inference.py<../../../scripts/action-recognition/inference.py>`

    For more command options, please run ``python inference.py -h``
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
# Each line is the path to each video you want to make predictions.
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
# Note that, at this moment, we only support inferencing on videos directly.

######################################################################
# Once you prepare the ``video.txt``, you can start inferencing on your videos.
# Let's first use I3D models as an example.
#
# ::
#
#     python inference.py --data-list video.txt --model i3d_resnet50_v1_kinetics400

######################################################################
# The predictions will be print out to the console and the log will be saved to the current directory.
# You can find the log as ``predictions.log``.
# If you want to save the logits (confidence score) to ``.npy`` and use it again later, you can
#
# ::
#
#     python inference.py --data-list video.txt --model i3d_resnet50_v1_kinetics400 --save-logits

######################################################################
# If you want to save both the logits and predictions to ``.npy`` and use them again later, you can
#
# ::
#
#     python inference.py --data-list video.txt --model i3d_resnet50_v1_kinetics400 --save-logits --save-preds

######################################################################
# If you want to use a strong network, like SlowFast. We support it as well.
# Just change the model name and pick which SlowFast configuration you want to use.
#
# ::
#
#     python inference.py --data-list video.txt --model slowfast_4x16_resnet50_kinetics400 --slowfast --slow-temporal-stride 16 --fast-temporal-stride 2 --new-length 64

######################################################################
# Here we choose the basic slowfast_4x16_resnet50 configuration.
# It requires the input to be a 64-frame video clip. We select 4 frames for the slow branch (temporal_stride = 16) and 32 frames for the fast branch (temporal_stride = 2).
#
# Similarly, you can specify num_segments, new_legnth, etc. as in previous tutorial to obtain more accurate predictions.
# There are many other options and other models you can choose, please check ``inference.py`` for more usage information.
