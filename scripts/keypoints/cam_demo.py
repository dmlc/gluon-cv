from __future__ import division
import argparse, time, logging, os, math, tqdm, cv2

import numpy as np
import mxnet as mx
from mxnet import gluon, nd, image
from mxnet.gluon.data.vision import transforms

import matplotlib.pyplot as plt

import gluoncv as gcv
from gluoncv import data
from gluoncv.data import mscoco
from gluoncv.model_zoo import get_model
from gluoncv.data.transforms.pose import get_final_preds

num_joints = 17

def upscale_bbox_fn(bbox, img, scale=1.25):
    new_bbox = []
    x0 = bbox[0]
    y0 = bbox[1]
    x1 = bbox[2]
    y1 = bbox[3]
    w = (x1 - x0) / 2
    h = (y1 - y0) / 2
    center = [x0 + w, y0 + h]
    new_x0 = max(center[0] - w * scale, 0)
    new_y0 = max(center[1] - h * scale, 0)
    new_x1 = min(center[0] + w * scale, img.shape[1])
    new_y1 = min(center[1] + h * scale, img.shape[0])
    new_bbox = [new_x0, new_y0, new_x1, new_y1]
    return new_bbox

def crop_resize_normalize(img, bbox_list, output_size):
    output_list = []
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    for bbox in bbox_list:
        x0 = max(int(bbox[0]), 0)
        y0 = max(int(bbox[1]), 0)
        x1 = min(int(bbox[2]), int(img.shape[1]))
        y1 = min(int(bbox[3]), int(img.shape[0]))
        w = x1 - x0
        h = y1 - y0
        # nd.image.resize is in mxnet 1.5.0
        # res_img = nd.image.resize(img[y0:y1, x0:x1], (256, 192))
        res_img = image.fixed_crop(nd.array(img), x0, y0, w, h, (output_size[1], output_size[0]))
        res_img = transform_test(res_img)
        output_list.append(res_img)
    output_array = nd.stack(*output_list)
    return output_array

def get_final_preds(batch_heatmaps, center, scale):
    from gluoncv.data.transforms.pose import get_max_pred
    coords, maxvals = get_max_pred(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    preds = nd.zeros_like(coords)

    # Transform back
    for i in range(coords.shape[0]):
        w_ratio = coords[i][:, 0] / heatmap_width
        h_ratio = coords[i][:, 1] / heatmap_height
        preds[i][:, 0] = scale[i][0] * 2 * w_ratio + center[i][0] - scale[i][0]
        preds[i][:, 1] = scale[i][1] * 2 * h_ratio + center[i][1] - scale[i][1]

    return preds, maxvals

def heatmap_to_coord(heatmaps, bbox_list):
    center_list = []
    scale_list = []
    for i, bbox in enumerate(bbox_list):
        x0 = bbox[0]
        y0 = bbox[1]
        x1 = bbox[2]
        y1 = bbox[3]
        w = (x1 - x0) / 2
        h = (y1 - y0) / 2
        center_list.append(np.array([x0 + w, y0 + h]))
        scale_list.append(np.array([w, h]))

    coords, maxvals = get_final_preds(heatmaps, center_list, scale_list)
    return coords, maxvals

def keypoint_detection(img=None, img_path=None, detector=None, pose_net=None):
    detector_tic = time.time()
    if img is not None:
        x, img = gcv.data.transforms.presets.yolo.transform_test(img, short=512, max_size=350)
    elif img_path is not None:
        x, img = data.transforms.presets.yolo.load_test(img_path, short=512)
    else:
        raise("Need img or img_path.")
    # pretrained images
    class_IDs, scores, bounding_boxs = detector(x)
    nd.waitall()
    detector_tic = time.time() - detector_tic

    transform_tic = time.time()
    L = class_IDs.shape[1]
    thr = 0.5
    upscale_bbox = []
    for i in range(L):
        if class_IDs[0][i].asscalar() != 0:
            continue
        if scores[0][i].asscalar() < thr:
            continue
        bbox = bounding_boxs[0][i]
        upscale_bbox.append(upscale_bbox_fn(bbox.asnumpy().tolist(), img, scale=1.25))

    if len(upscale_bbox) > 0:
        pose_input = crop_resize_normalize(nd.array(img), upscale_bbox, (256, 192))
        nd.waitall()
        transform_tic = time.time() - transform_tic

        pose_tic = time.time()
        predicted_heatmap = pose_net(pose_input)
        nd.waitall()
        pose_tic = time.time() - pose_tic

        post_proc_tic = time.time()
        pred_coords, _ = heatmap_to_coord(predicted_heatmap, upscale_bbox)
        nd.waitall()
        post_proc_tic = time.time() - post_proc_tic

        person_ind = class_IDs[0].asnumpy() == 0
        ax = gcv.utils.viz.plot_bbox(img, bounding_boxs[0].asnumpy()[person_ind[:,0]],
                                     scores[0].asnumpy()[person_ind[:,0]], thresh=0.5)
        plt.xlim([0, img.shape[1]-1])
        plt.ylim([0, img.shape[0]-1])
        plt.gca().invert_yaxis()
        for i in range(pred_coords.shape[0]):
            pts = pred_coords[i].asnumpy()
            plt.scatter(pts[:,0], pts[:,1], s=20)
            joint_pairs = [[0,1], [1,3], [0,2], [2,4],
                           [5,6], [5,7], [7,9], [6,8], [8,10],
                           [5,11], [6,12], [11,12],
                           [11,13], [12,14], [13,15], [14,16]]
            colormap_index = np.linspace(0, 1, len(joint_pairs))
            for cm_ind, jp in zip(colormap_index, joint_pairs):
                plt.plot(pts[jp, 0], pts[jp, 1],
                         linewidth=5.0, alpha=0.7, color=plt.cm.cool(cm_ind))

        plt.draw()
        plt.pause(0.1)
        print(detector_tic, transform_tic, pose_tic, post_proc_tic)
    else:
        plt.imshow(img)
        plt.draw()
        plt.pause(0.1)
        print("Nobody detected.")

if __name__ == '__main__':
    detector_name = "ssd_512_mobilenet1.0_coco"
    # detector_name = "yolo3_mobilenet1.0_coco"
    detector = get_model(detector_name, ctx=mx.cpu(), pretrained=True)
    detector.reset_class(classes=['person'], reuse_weights={'person':'person'})
    net = get_model('simple_pose_resnet18_v1b', pretrained=True, ctx=mx.cpu())

    cap = cv2.VideoCapture(0)
    time.sleep(1)  ### letting the camera autofocus

    for i in range(100):
        ret, frame = cap.read()
        frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')
        keypoint_detection(frame, None, detector, net)
