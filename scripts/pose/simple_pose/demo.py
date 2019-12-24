from __future__ import division
import argparse

import numpy as np
import mxnet as mx
from mxnet import gluon, nd, image

import matplotlib.pyplot as plt

import gluoncv as gcv
gcv.utils.check_version('0.6.0')
from gluoncv import data
from gluoncv.model_zoo import get_model
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
from gluoncv.utils.viz import plot_keypoints


parser = argparse.ArgumentParser(description='Predict ImageNet classes from a given image')
parser.add_argument('--detector', type=str, default='yolo3_mobilenet1.0_coco',
                    help='name of the detection model to use')
parser.add_argument('--pose-model', type=str, default='simple_pose_resnet50_v1b',
                    help='name of the pose estimation model to use')
parser.add_argument('--input-pic', type=str, required=True,
                    help='path to the input picture')
opt = parser.parse_args()

def keypoint_detection(img_path, detector, pose_net):
    x, img = data.transforms.presets.yolo.load_test(img_path, short=512)
    class_IDs, scores, bounding_boxs = detector(x)

    pose_input, upscale_bbox = detector_to_simple_pose(img, class_IDs, scores, bounding_boxs)
    predicted_heatmap = pose_net(pose_input)
    pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)

    ax = plot_keypoints(img, pred_coords, confidence, class_IDs, bounding_boxs, scores,
                        box_thresh=0.5, keypoint_thresh=0.2)
    plt.show()

if __name__ == '__main__':
    detector = get_model(opt.detector, pretrained=True)
    detector.reset_class(["person"], reuse_weights=['person'])
    net = get_model(opt.pose_model, pretrained=True)

    keypoint_detection(opt.input_pic, detector, net)
