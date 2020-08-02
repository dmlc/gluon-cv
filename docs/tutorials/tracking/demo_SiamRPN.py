"""01. Single object tracking with pre-trained SiamRPN models
=============================================================

Object tracking is a long standing and useful computer vision task. Unlike image object detection,
it predicts the position of the object in the next frame according to its position
in the current (and sometimes previous) frame.

`SiamRPN <http://openaccess.thecvf.com/content_cvpr_2018/papers/
Li_High_Performance_Visual_CVPR_2018_paper.pdf>`_ (Siamese Region Proposal Network)
is a widely adopted single object tracking method.
It consists of a Siamese subnetwork for
feature extraction and a region proposal subnetwork
including the classification branch and regression branch for prediction.
In the inference phase, the proposed framework is formulated as a local one-shot detection task.
We can pre-compute the template branch of the Siamese subnetwork and formulate the
correlation layers as trivial convolution layers to perform online tracking.

In this tutorial, we will demonstrate how to load a pre-trained SiamRPN model
from :ref:`gluoncv-model-zoo`
and perform single object tracking on an arbitrary video.
"""

######################################################################
# Predict with a SiamRPN model
# ----------------------------
#
# You need to prepare two things to start a tracking demo, the video and its first frame object coordinates.
# The coordinates show the region of interest where to track, and in the format of
# (min_x, min_y, width, height).
#
# Here we download a video and set the region of interest in the first frame as [298, 160, 48, 80].

from gluoncv import utils
video_path = 'https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/tracking/Coke.mp4'
im_video = utils.download(video_path)
gt_bbox = [298, 160, 48, 80]

################################################################
# Then you can simply use our provided script to obtain the object tracking result,
#
# ::
#
#     python demo.py --video-path ./Coke.mp4 --gt-bbox 298 160 48 80
#
#
################################################################
# You can see the tracking results below.
#
# .. raw:: html
#
#     <div align="center">
#         <img src="../../_static/tracking_demo.gif">
#     </div>
#
#     <br>

################################################################
# Our model is very stable.
# It can track the object even when it moves at high speed and is partially occluded.
# Try it on your own video and see the results!
