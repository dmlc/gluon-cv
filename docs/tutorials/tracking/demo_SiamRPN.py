"""01. Predict with pre-trained SiamRPN models
Object tracking is often used in video object detection, but unlike image object detection,
it predicts the position of the next frame according to the position of the object of previous frame

`SiamRPN <http://openaccess.thecvf.com/content_cvpr_2018/papers/Li_High_Performance_Visual_CVPR_2018_paper.pdf>`_ 
(Siamese Region Proposal Network) is a widely adopted Object tracking method.
it consists of Siamese subnetwork for
feature extraction and region proposal subnetwork
including the classification branch and regression branch.
In the inference phase, the proposed framework is formulated as a local one-shot detection task.
We can pre-compute the template branch of the Siamese subnetwork and formulate the
correlation layers as trivial convolution layers to perform online tracking.

In this tutorial, we will demonstrate how to load a pre-trained SiamRPN model from :ref:`gluoncv-model-zoo`
and predict video object location from the Internet according to first frame video object location.

==========================================
This article shows how to play with pre-trained SiamRPN models with only a few
lines of code.
First let's import some necessary libraries:
"""

from gluoncv import model_zoo, data, utils
from gluoncv.utils.siamrpn_tracker import SiamRPNTracker as build_tracker
from gluoncv.utils.siamrpn_tracker import get_axis_aligned_bbox
from gluoncv.utils.filesystem import try_import_cv2
import matplotlib.pyplot as plt
import numpy as np
import mxnet as mx
import os
cv2 = try_import_cv2()

######################################################################
# Load a pretrained model
# -------------------------
#
# Let's get an SiamRPN model trained. We pick the one using Alexnet as the base model.
# By specifying ``pretrained=True``, it will automatically download the model from the model
# zoo if necessary. For more pretrained models, please refer to
# :doc:`../../model_zoo/index`.
net = model_zoo.get_model('siamrpn_alexnet_v2_otb15', ctx = mx.cpu(0), pretrained=True)

######################################################################
# Build a tracker model
# -------------------------
tracker = build_tracker(net)

######################################################################
# Pre-process an video 
# --------------------
#
# Next we need a video and first frame object coordinates
# download an video，it as input to the network
# gt_bbox is first frame object coordinates，and it is bbox(center_x,center_y,weight,height)
im_video = utils.download('https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/tracking/Coke.mp4')
gt_bbox = [298,160,48,80]
cap = cv2.VideoCapture(im_video)
frames_total_num = cap.get(7)
video_frames = []

while(True):
    ret, img = cap.read()
    if not ret:
        break
    video_frames.append(img)

######################################################################
# --------------------
#  Then, show the example image and bbox:
plt.imshow(video_frames[0])
plt.show()

######################################################################
# Predict with a SiamRPN and make inference
# --------------------
#
# this function returns a dictionaries result. which has two keys.one is
# this function returns a dictionaries result. which has two keys. one is bbox,
# which represents the coordinates of the predicted frame,
# the other is best_score, which records everyframe best_score.
scores = []
pred_bboxes = []
for ind , img in enumerate(video_frames):
    if ind == 0:
        cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
        gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
        tracker.init(img, gt_bbox_)
        pred_bbox = gt_bbox_
        scores.append(None)
        pred_bboxes.append(pred_bbox)
    else:
        outputs = tracker.track(img)
        pred_bbox = outputs['bbox']
        pred_bboxes.append(pred_bbox)
        scores.append(outputs['best_score'])
    pred_bbox = list(map(int, pred_bbox))
    cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                            (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]),
                            (0, 255, 255), 3)
    cv2.imwrite('%04d.jpg'%(ind+1), img)

# this is example of our tracking.We can find Our model is very stable.
# It can still track this object when moving at high speed and partially occluded. Welcome to use.
# .. raw:: html
#
#     <div align="center">
#         <img src="../../_static/tracking_demo.gif">
#     </div>
#
#     <br>
