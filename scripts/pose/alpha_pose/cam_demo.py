from __future__ import division
import argparse, time, logging, os, math, tqdm, cv2

import numpy as np
import mxnet as mx
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms

import matplotlib.pyplot as plt

import gluoncv as gcv
gcv.utils.check_version('0.6.0')
from gluoncv import data
from gluoncv.data import mscoco
from gluoncv.model_zoo import get_model
from gluoncv.data.transforms.pose import detector_to_alpha_pose, heatmap_to_coord_alpha_pose
from gluoncv.utils.viz import plot_image, plot_keypoints

parser = argparse.ArgumentParser(description='Predict ImageNet classes from a given image')
parser.add_argument('--detector', type=str, default='yolo3_mobilenet1.0_coco',
                    help='name of the detection model to use')
parser.add_argument('--pose-model', type=str, default='alpha_pose_resnet101_v1b_coco',
                    help='name of the pose estimation model to use')
parser.add_argument('--num-frames', type=int, default=100,
                    help='Number of frames to capture')
opt = parser.parse_args()

def keypoint_detection(img, detector, pose_net, ctx=mx.cpu(), axes=None):
    x, img = gcv.data.transforms.presets.yolo.transform_test(img, short=512, max_size=350)
    x = x.as_in_context(ctx)
    class_IDs, scores, bounding_boxs = detector(x)

    plt.cla()
    pose_input, upscale_bbox = detector_to_alpha_pose(img, class_IDs, scores, bounding_boxs,
                                                       output_shape=(128, 96), ctx=ctx)
    if len(upscale_bbox) > 0:
        predicted_heatmap = pose_net(pose_input)
        pred_coords, confidence = heatmap_to_coord_alpha_pose(predicted_heatmap, upscale_bbox)

        axes = plot_keypoints(img, pred_coords, confidence, class_IDs, bounding_boxs, scores,
                              box_thresh=0.5, keypoint_thresh=0.2, ax=axes)
        plt.draw()
        plt.pause(0.001)
    else:
        axes = plot_image(frame, ax=axes)
        plt.draw()
        plt.pause(0.001)

    return axes

if __name__ == '__main__':
    ctx = mx.cpu()
    detector_name = "ssd_512_mobilenet1.0_coco"
    detector = get_model(detector_name, pretrained=True, ctx=ctx)
    detector.reset_class(classes=['person'], reuse_weights={'person':'person'})
    net = get_model('simple_pose_resnet18_v1b', pretrained='ccd24037', ctx=ctx)

    cap = cv2.VideoCapture(0)
    time.sleep(1)  ### letting the camera autofocus
    axes = None

    for i in range(opt.num_frames):
        ret, frame = cap.read()
        frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')
        axes = keypoint_detection(frame, detector, net, ctx, axes=axes)
