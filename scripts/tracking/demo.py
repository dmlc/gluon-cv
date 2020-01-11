"""01. Predict with pre-trained SiamRPN models
==========================================
This article shows how to play with pre-trained SiamRPN models with only a few
lines of code.
First let's import some necessary libraries:
"""

from gluoncv import model_zoo, data, utils
from gluoncv.utils.siamrpn_tracker import SiamRPNTracker as build_tracker
import cv2
import numpy as np
import mxnet as mx
import os
import imageio
from mxnet import nd

######################################################################
# Load a pretrained model
# -------------------------
#
# Let's get an siamrpn model trained 
# dataset with Alexnet as the base model. By specifying
# ``pretrained=True``, it will automatically download the model from the model
# zoo if necessary. For more pretrained models, please refer to
# :doc:`../../model_zoo/index`.
net = model_zoo.get_model('siamrpn_alexnet_v2_otb15', ctx = mx.cpu(0), pretrained=True)

######################################################################
# Build a pretrained model
# -------------------------
tracker = build_tracker(net)
######################################################################
# Pre-process an video 
# --------------------
#
# Next we download an video，it as input to the network
# gt_bbox is first frame object coordinates，and it is bbox(center_x,center_y,center_w,center_h)
# this model has two module，one is tracker.init. it has Picture and coordinates of the previous frame as input.
# the other is tracker.track. it has Picture of the predict frame needed as input.
# And returns is Dictionaries. keys are gt_bbox and score.Represents the coordinates and scores of the predicted frame.

im_video = utils.download('https://raw.githubusercontent.com/FrankYoungchen/siam_data/master/Dog.mp4')
gif_path = im_video.split('.')[0] + '.gif'
gt_bbox = [74, 86, 56, 48]

cap = cv2.VideoCapture(im_video)
frames_total_num = cap.get(7)
index = 1
scores = []
pred_bboxes = []
gif_frame = []

while(True):
    ret, img = cap.read()
    gt_bbox_ = np.array(gt_bbox)
    tracker.init(img, gt_bbox_)
    if index == 1:
        gt_bbox = gt_bbox_
        pred_bbox = gt_bbox_
        scores.append(None)
        pred_bboxes.append(pred_bbox)
    else:
        outputs = tracker.track(img)
        pred_bbox = outputs['bbox']
        pred_bboxes.append(pred_bbox)
        scores.append(outputs['best_score'])
        gt_bbox_ = pred_bbox
    
    gt_bbox = list(map(int, gt_bbox))
    pred_bbox = list(map(int, pred_bbox))
    cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                            (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]),
                            (0, 255, 255), 3)
    gif_frame.append(img)
    index = index+1 
    if index>frames_total_num:        
        break
# Finally, Generate GIF based on prediction results
imageio.mimsave(gif_path, gif_frame, 'GIF', duration=0.1)