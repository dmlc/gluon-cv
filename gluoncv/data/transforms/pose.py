# pylint: disable=all
"""Pose related transformation functions

Adapted from https://github.com/Microsoft/human-pose-estimation.pytorch

---------------------------------------------
Copyright (c) Microsoft
Licensed under the MIT License.
Written by Bin Xiao (Bin.Xiao@microsoft.com)
---------------------------------------------
"""
from __future__ import absolute_import
from __future__ import division

import math
import numpy as np
import mxnet as mx
from mxnet import nd, image
from mxnet.gluon.data.vision import transforms
from ...utils.filesystem import try_import_cv2

def flip_heatmap(heatmap, joint_pairs, shift=False):
    """Flip pose heatmap according to joint pairs.

    Parameters
    ----------
    heatmap : numpy.ndarray
        Heatmap of joints.
    joint_pairs : list
        List of joint pairs
    shift : bool
        Whether to shift the output

    Returns
    -------
    numpy.ndarray
        Flipped heatmap

    """
    assert heatmap.ndim == 4, "heatmap should have shape (batch_size, num_joints, height, width)"
    out = heatmap[:, :, :, ::-1]

    for pair in joint_pairs:
        tmp = out[:, pair[0], :, :].copy()
        out[:, pair[0], :, :] = out[:, pair[1], :, :]
        out[:, pair[1], :, :] = tmp

    if shift:
        out[:, :, :, 1:] = out[:, :, :, 0:-1]
    return out

def flip_joints_3d(joints_3d, width, joint_pairs):
    """Flip 3d joints.

    Parameters
    ----------
    joints_3d : numpy.ndarray
        Joints in shape (num_joints, 3, 2)
    width : int
        Image width.
    joint_pairs : list
        List of joint pairs.

    Returns
    -------
    numpy.ndarray
        Flipped 3d joints with shape (num_joints, 3, 2)

    """
    joints = joints_3d.copy()
    # flip horizontally
    joints[:, 0, 0] = width - joints[:, 0, 0] - 1
    # change left-right parts
    for pair in joint_pairs:
        joints[pair[0], :, 0], joints[pair[1], :, 0] = \
            joints[pair[1], :, 0], joints[pair[0], :, 0].copy()
        joints[pair[0], :, 1], joints[pair[1], :, 1] = \
            joints[pair[1], :, 1], joints[pair[0], :, 1].copy()

    joints[:, :, 0] *= joints[:, :, 1]
    return joints

def transform_predictions(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    cv2 = try_import_cv2()
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def crop(img, center, scale, output_size, rot=0):
    cv2 = try_import_cv2()
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(img,
                             trans,
                             (int(output_size[0]), int(output_size[1])),
                             flags=cv2.INTER_LINEAR)

    return dst_img

def transform_preds(coords, center, scale, output_size):
    target_coords = nd.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2].asnumpy(), trans)
    return target_coords


def get_max_pred(batch_heatmaps):
    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = nd.argmax(heatmaps_reshaped, 2)
    maxvals = nd.max(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = nd.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = nd.floor((preds[:, :, 1]) / width)

    pred_mask = nd.tile(nd.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_final_preds(batch_heatmaps, center, scale):
    coords, maxvals = get_max_pred(batch_heatmaps)

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    # post-processing
    for n in range(coords.shape[0]):
        for p in range(coords.shape[1]):
            hm = batch_heatmaps[n][p]
            px = int(nd.floor(coords[n][p][0] + 0.5).asscalar())
            py = int(nd.floor(coords[n][p][1] + 0.5).asscalar())
            if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                diff = nd.concat(hm[py][px+1] - hm[py][px-1],
                                 hm[py+1][px] - hm[py-1][px],
                                 dim=0)
                coords[n][p] += nd.sign(diff) * .25

    preds = nd.zeros_like(coords)

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(coords[i], center[i], scale[i],
                                   [heatmap_width, heatmap_height])

    return preds, maxvals

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

def crop_resize_normalize(img, bbox_list, output_size,
                          mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    output_list = []
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    for bbox in bbox_list:
        x0 = max(int(bbox[0]), 0)
        y0 = max(int(bbox[1]), 0)
        x1 = min(int(bbox[2]), int(img.shape[1]))
        y1 = min(int(bbox[3]), int(img.shape[0]))
        w = x1 - x0
        h = y1 - y0
        res_img = image.fixed_crop(nd.array(img), x0, y0, w, h, (output_size[1], output_size[0]))
        res_img = transform_test(res_img)
        output_list.append(res_img)
    output_array = nd.stack(*output_list)
    return output_array

def detector_to_simple_pose(img, class_ids, scores, bounding_boxs,
                            output_shape=(256, 192), scale=1.25, ctx=mx.cpu(),
                            thr=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    L = class_ids.shape[1]
    upscale_bbox = []
    for i in range(L):
        if class_ids[0][i].asscalar() != 0:
            continue
        if scores[0][i].asscalar() < thr:
            continue
        bbox = bounding_boxs[0][i]
        upscale_bbox.append(upscale_bbox_fn(bbox.asnumpy().tolist(), img, scale=scale))
    if len(upscale_bbox) > 0:
        pose_input = crop_resize_normalize(img, upscale_bbox, output_shape, mean=mean, std=std)
        pose_input = pose_input.as_in_context(ctx)
    else:
        pose_input = None
    return pose_input, upscale_bbox

def heatmap_to_coord(heatmaps, bbox_list):
    heatmap_height = heatmaps.shape[2]
    heatmap_width = heatmaps.shape[3]
    coords, maxvals = get_max_pred(heatmaps)
    preds = nd.zeros_like(coords)

    for i, bbox in enumerate(bbox_list):
        x0 = bbox[0]
        y0 = bbox[1]
        x1 = bbox[2]
        y1 = bbox[3]
        w = (x1 - x0) / 2
        h = (y1 - y0) / 2
        center = np.array([x0 + w, y0 + h])
        scale = np.array([w, h])

        w_ratio = coords[i][:, 0] / heatmap_width
        h_ratio = coords[i][:, 1] / heatmap_height
        preds[i][:, 0] = scale[0] * 2 * w_ratio + center[0] - scale[0]
        preds[i][:, 1] = scale[1] * 2 * h_ratio + center[1] - scale[1]
    return preds, maxvals


'''AlphaPose'''
def alpha_pose_detection_processor(img, boxes, class_idxs, scores, thr=0.5):
    if len(boxes.shape) == 3:
        boxes = boxes.squeeze(axis=0)

    # cilp coordinates
    boxes[:, [0, 2]] = mx.nd.clip(boxes[:, [0, 2]], 0., img.shape[1] - 1)
    boxes[:, [1, 3]] = mx.nd.clip(boxes[:, [1, 3]], 0., img.shape[0] - 1)

    # select boxes
    mask1 = (class_idxs == 0).asnumpy()
    mask2 = (scores > thr).asnumpy()
    picked_idxs = np.where((mask1 + mask2) > 1)[1]
    if picked_idxs.shape[0] == 0:
        return None, None
    else:
        return boxes[picked_idxs], scores[picked_idxs]

def alpha_pose_image_cropper(img, boxes, scores, output_shape=(256, 192),
                             mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225)):
    if boxes is None:
        return None, boxes

    # crop person poses
    img_width, img_height = img.shape[1], img.shape[0]

    tensors = mx.nd.zeros([boxes.shape[0], 3, output_shape[0], output_shape[1]])
    out_boxes = np.zeros([boxes.shape[0], 4])

    img = mx.nd.image.to_tensor(nd.array(img))
    img = mx.nd.image.normalize(img, mean=mean, std=std)
    img = img.transpose(axes=[1, 2, 0])

    for i, box in enumerate(boxes.asnumpy()):
        box_width = box[2] - box[0]
        box_height = box[3] - box[1]
        if box_width > 100:
            scale_rate = 0.2
        else:
            scale_rate = 0.3

        # crop image
        left = int(max(0, box[0] - box_width * scale_rate / 2))
        up = int(max(0, box[1] - box_height * scale_rate / 2))
        right = int(min(img_width - 1,
                        max(left + 5, box[2] + box_width * scale_rate / 2)))
        bottom = int(min(img_height - 1,
                         max(up + 5, box[3] + box_height * scale_rate / 2)))
        crop_width = right - left
        if crop_width < 1:
            continue
        crop_height = bottom - up
        if crop_height < 1:
            continue
        cropped_img = mx.image.fixed_crop(img, left, up, crop_width, crop_height)

        # resize image
        resize_factor = min(output_shape[1] / crop_width, output_shape[0] / crop_height)
        new_width = int(crop_width * resize_factor)
        new_height = int(crop_height * resize_factor)
        tensor = mx.image.imresize(cropped_img, new_width, new_height)
        tensor = tensor.transpose(axes=[2, 0, 1])
        tensor = tensor.reshape(1, 3, new_height, new_width)

        # pad tensor
        pad_h = output_shape[0] - new_height
        pad_w = output_shape[1] - new_width
        pad_shape = (0, 0, 0, 0, pad_h // 2, (pad_h + 1) // 2, pad_w // 2, (pad_w + 1) // 2)
        tensor = mx.nd.pad(tensor, mode='constant',
                           constant_value=0.5, pad_width=pad_shape)
        tensors[i] = tensor.reshape(3, output_shape[0], output_shape[1])
        out_boxes[i] = (left, up, right, bottom)


    return tensors, out_boxes

def heatmap_to_coord_alpha_pose(hms, boxes):
    hm_h = hms.shape[2]
    hm_w = hms.shape[3]
    aspect_ratio = float(hm_h) / hm_w
    pt1 = mx.nd.array(boxes[:, (0, 2)])
    pt2 = mx.nd.array(boxes[:, (1, 3)])

    # get keypoint coordinates
    idxs = mx.nd.argmax(hms.reshape(hms.shape[0], hms.shape[1], -1), 2, keepdims=True)
    maxval = mx.nd.max(hms.reshape(hms.shape[0], hms.shape[1], -1), 2, keepdims=True)
    preds = idxs.tile(reps=[1, 1, 2])
    preds[:, :, 0] %= hms.shape[3]
    preds[:, :, 1] /= hms.shape[3]

    # get pred masks
    pred_mask = (maxval > 0).tile(reps=[1, 1, 2])
    preds *= pred_mask

    # coordinate transformation
    box_size = pt2 - pt1
    len_h = mx.nd.maximum(box_size[:, 1:2], box_size[:, 0:1] * aspect_ratio)
    len_w = len_h / aspect_ratio
    canvas_size = mx.nd.concatenate([len_w, len_h], axis=1)
    offsets = pt1 - mx.nd.maximum(0, canvas_size / 2 - box_size / 2)
    preds_tf = preds * len_h / hm_h + offsets

    return preds_tf, maxval

def detector_to_alpha_pose(img, class_ids, scores, bounding_boxs,
                            output_shape=(256, 192), scale=1.25, ctx=mx.cpu(),
                            thr=0.5, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    boxes, scores = alpha_pose_detection_processor(
        img, bounding_boxs, class_ids, scores, thr=thr)
    pose_input, upscale_bbox = alpha_pose_image_cropper(
        img, boxes, scores, output_shape=output_shape, mean=mean, std=std)
    return pose_input, upscale_bbox
