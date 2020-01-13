"""01. Predict with pre-trained SiamRPN models
==========================================
This article shows how to play with pre-trained SiamRPN models with only a few
lines of code.
First let's import some necessary libraries:
"""

from gluoncv import model_zoo, data, utils
from gluoncv.utils.siamrpn_tracker import SiamRPNTracker as build_tracker
from gluoncv.utils.siamrpn_tracker import get_axis_aligned_bbox
import cv2
import numpy as np
import mxnet as mx
import os

######################################################################
# Load a pretrained model
# -------------------------
#
# Let's get an siamrpn model trained. We pick the one using Alexnet as the base model.
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
# Next we download an video，it as input to the network
# gt_bbox is first frame object coordinates，and it is bbox(center_x,center_y,center_w,center_h)
# this model has two module，one is tracker.init. it has Picture and coordinates of the previous frame as input.
# get_axis_aligned_bbox converts region to (cx, cy, w, h) that represent by axis aligned box
# the other is tracker.track. it has Picture of the predict frame needed as input.
# And returns is Dictionaries. keys are bbox and best_score. Represents the coordinates and scores of the predicted frame.
# scores list record everyframe best_score and pred_bboxes list record everyframe predict bbox coordinate.
# result_path is the path where you save the result ，which draw pictures and save tracking result.
# this is example of our tracking
# .. raw:: html
#
#     <div align="center">
#         <img src="../../_static/tracking_demo.gif">
#     </div>
#
#     <br>

im_video = utils.download('https://raw.githubusercontent.com/FrankYoungchen/siam_data/master/Coke.mp4')
gt_bbox = [298,160,48,80]

result_path = './result'
cap = cv2.VideoCapture(im_video)
frames_total_num = cap.get(7)
index = 1
scores = []
pred_bboxes = []

while(True):
    ret, img = cap.read()
    if not ret:
        break
    if index == 1:
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
    cv2.imwrite(os.path.join(result_path, '%04d.jpg'%index), img)
    index = index+1 
    if index>frames_total_num:        
        break