"""03. Multiple object tracking with pre-trained SMOT models
=============================================================

In this tutorial, we present a method,
called `Single-Shot Multi Object Tracking (SMOT) <https://arxiv.org/abs/2010.16031>`_, to perform multi-object tracking.
SMOT is a new tracking framework that converts any single-shot detector (SSD) model into an online multiple object tracker,
which emphasizes simultaneously detecting and tracking of the object paths.
As an example below, we directly use the SSD-Mobilenet object detector pretrained on COCO from :ref:`gluoncv-model-zoo`
and perform multiple object tracking on an arbitrary video.
We want to point out that, SMOT is very efficient, its runtime is close to the runtime of the chosen detector.

"""

######################################################################
# Predict with a SMOT model
# ----------------------------
#
# First, we download a video from MOT challenge website,

from gluoncv import utils
video_path = 'https://motchallenge.net/sequenceVideos/MOT17-02-FRCNN-raw.webm'
im_video = utils.download(video_path)

################################################################
# Then you can simply use our provided script under `/scripts/tracking/smot/demo.py` to obtain the multi-object tracking result.
#
# ::
#
#     python demo.py MOT17-02-FRCNN-raw.webm --network-name ssd_512_mobilenet1.0_coco --use-pretrained --custom-classes person --use-motion
#
#
################################################################
# You can see the tracking results below. Here, we only track persons,
# but you can track other objects as long as your detector is trained on that category.
#
# .. raw:: html
#
#     <div align="center">
#         <img src="../../_static/smot_demo.gif">
#     </div>
#
#     <br>

################################################################
# Our model is able to track multiple persons even when they are partially occluded.
# If you want to track multiple object categories at the same time,
# you can simply pass in the extra class names.
#
# For example, let's download a video from MOT challenge website,

from gluoncv import utils
video_path = 'https://motchallenge.net/sequenceVideos/MOT17-13-FRCNN-raw.webm'
im_video = utils.download(video_path)

################################################################
# Then you can simply use our provided script under `/scripts/tracking/smot/demo.py` to obtain the multi-object tracking result.
#
# ::
#
#     python demo.py MOT17-13-FRCNN-raw.webm --network-name ssd_512_resnet50_v1_coco --use-pretrained --custom-classes person car --detect-thresh 0.7 --use-motion
#
#
# Now we are tracking both person and cars,
#
# .. raw:: html
#
#     <div align="center">
#         <img src="../../_static/smot_multi_demo.gif">
#     </div>
#
#     <br>

################################################################
# Try SMOT on your own video and see the results!
