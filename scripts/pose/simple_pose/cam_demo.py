from __future__ import division
import argparse, time, logging, os, math, tqdm, cv2

import numpy as np
import mxnet as mx
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms

import gluoncv as gcv
gcv.utils.check_version('0.6.0')
from gluoncv import data
from gluoncv.data import mscoco
from gluoncv.model_zoo import get_model
from gluoncv.data.transforms.pose import detector_to_simple_pose, heatmap_to_coord
from gluoncv.utils.viz import cv_plot_image, cv_plot_keypoints

parser = argparse.ArgumentParser(description='Predict ImageNet classes from a given image')
parser.add_argument('--detector', type=str, default='yolo3_mobilenet1.0_coco',
                    help='name of the detection model to use')
parser.add_argument('--pose-model', type=str, default='simple_pose_resnet50_v1b',
                    help='name of the pose estimation model to use')
parser.add_argument('--num-frames', type=int, default=100,
                    help='Number of frames to capture')
opt = parser.parse_args()

def keypoint_detection(img, detector, pose_net, ctx=mx.cpu()):
    x, scaled_img = gcv.data.transforms.presets.yolo.transform_test(img, short=480, max_size=1024)
    x = x.as_in_context(ctx)
    class_IDs, scores, bounding_boxs = detector(x)

    pose_input, upscale_bbox = detector_to_simple_pose(scaled_img, class_IDs, scores, bounding_boxs,
                                                       output_shape=(128, 96), ctx=ctx)
    if len(upscale_bbox) > 0:
        predicted_heatmap = pose_net(pose_input)
        pred_coords, confidence = heatmap_to_coord(predicted_heatmap, upscale_bbox)

        scale = 1.0 * img.shape[0] / scaled_img.shape[0]
        img = cv_plot_keypoints(img.asnumpy(), pred_coords, confidence, class_IDs, bounding_boxs, scores,
                                box_thresh=1, keypoint_thresh=0.3, scale=scale)
    return img

if __name__ == '__main__':
    ctx = mx.cpu()
    detector_name = "ssd_512_mobilenet1.0_coco"
    detector = get_model(detector_name, pretrained=True, ctx=ctx)
    detector.reset_class(classes=['person'], reuse_weights={'person':'person'})
    net = get_model('simple_pose_resnet18_v1b', pretrained='ccd24037', ctx=ctx)

    cap = cv2.VideoCapture(0)
    time.sleep(1)  ### letting the camera autofocus

    for i in range(opt.num_frames):
        ret, frame = cap.read()
        frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')
        img = keypoint_detection(frame, detector, net, ctx=ctx)
        cv_plot_image(img)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()
