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
import random
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
def refine_bound(ul, br):
    """Adjust bound"""
    ul[0] = min(ul[0], br[0] - 5)
    ul[1] = min(ul[1], br[1] - 5)
    br[0] = max(br[0], ul[0] + 5)
    br[1] = max(br[1], ul[1] + 5)
    return ul, br

def random_crop_bbox(ul, br):
    """Random crop bbox"""
    switch = random.uniform(0, 1)
    if switch > 0.96:
        br[0] = (ul[0] + br[0]) / 2
        br[1] = (ul[1] + br[1]) / 2
    elif switch > 0.92:
        ul[0] = (ul[0] + br[0]) / 2
        br[1] = (ul[1] + br[1]) / 2
    elif switch > 0.88:
        ul[1] = (ul[1] + br[1]) / 2
        br[0] = (ul[0] + br[0]) / 2
    elif switch > 0.84:
        ul[0] = (ul[0] + br[0]) / 2
        ul[1] = (ul[1] + br[1]) / 2
    elif switch > 0.80:
        br[0] = (ul[0] + br[0]) / 2
    elif switch > 0.76:
        ul[0] = (ul[0] + br[0]) / 2
    elif switch > 0.72:
        br[1] = (ul[1] + br[1]) / 2
    elif switch > 0.68:
        ul[1] = (ul[1] + br[1]) / 2
    return ul, br

def random_sample_bbox(ul, br, w, h, im_width, im_height):
    """Take random sample"""
    patch_scale = random.uniform(0, 1)
    if patch_scale > 0.85:
        ratio = float(h) / w
        if w < h:
            patch_w = patch_scale * w
            patch_h = patch_w * ratio
        else:
            patch_h = patch_scale * h
            patch_w = patch_h / ratio
        xmin = ul[0] + random.uniform(0, 1) * (w - patch_w)
        ymin = ul[1] + random.uniform(0, 1) * (h - patch_h)
        xmax = xmin + patch_w + 1
        ymax = ymin + patch_h + 1
    else:
        xmin = max(1, min(ul[0] + np.random.normal(-0.0142, 0.1158) * w, im_width - 3))
        ymin = max(1, min(ul[1] + np.random.normal(0.0043, 0.068) * h, im_height - 3))
        xmax = min(max(xmin + 2, br[0] + np.random.normal(0.0154, 0.1337) * w), im_width - 3)
        ymax = min(max(ymin + 2, br[1] + np.random.normal(-0.0013, 0.0711) * h), im_height - 3)

    ul[0] = xmin
    ul[1] = ymin
    br[0] = xmax
    br[1] = ymax
    return ul, br

def count_visible(ul, br, joints_3d):
    """Count number of visible joints given bound ul, br"""
    vis = np.logical_and.reduce((
        joints_3d[:, 0, 0] > 0,
        joints_3d[:, 0, 0] > ul[0],
        joints_3d[:, 0, 0] < br[0],
        joints_3d[:, 1, 0] > 0,
        joints_3d[:, 1, 0] > ul[1],
        joints_3d[:, 1, 0] < br[1],
        joints_3d[:, 0, 1] > 0,
        joints_3d[:, 1, 1] > 0
        ))
    return np.sum(vis), vis

def cv_cropBox(img, ul, br, resH, resW, pad_val=0):
    cv2 = try_import_cv2()
    ul = ul
    br = (br - 1)
    # br = br.int()
    lenH = max((br[1] - ul[1]).item(), (br[0] - ul[0]).item() * resH / resW)
    lenW = lenH * resW / resH
    if img.ndim == 2:
        img = img[:, np.newaxis]

    box_shape = [br[1] - ul[1], br[0] - ul[0]]
    pad_size = [(lenH - box_shape[0]) // 2, (lenW - box_shape[1]) // 2]
    # Padding Zeros
    img[:ul[1], :, :], img[:, :ul[0], :] = pad_val, pad_val
    img[br[1] + 1:, :, :], img[:, br[0] + 1:, :] = pad_val, pad_val

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    src[0, :] = np.array(
        [ul[0] - pad_size[1], ul[1] - pad_size[0]], np.float32)
    src[1, :] = np.array(
        [br[0] + pad_size[1], br[1] + pad_size[0]], np.float32)
    dst[0, :] = 0
    dst[1, :] = np.array([resW - 1, resH - 1], np.float32)

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    dst_img = cv2.warpAffine(img, trans,
                             (resW, resH), flags=cv2.INTER_LINEAR)

    return dst_img

def cv_rotate(img, rot, resW, resH):
    cv2 = try_import_cv2()
    center = np.array((resW - 1, resH - 1)) / 2
    rot_rad = np.pi * rot / 180

    src_dir = get_dir([0, (resH - 1) * -0.5], rot_rad)
    dst_dir = np.array([0, (resH - 1) * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    src[0, :] = center
    src[1, :] = center + src_dir
    dst[0, :] = [(resW - 1) * 0.5, (resH - 1) * 0.5]
    dst[1, :] = np.array([(resW - 1) * 0.5, (resH - 1) * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    dst_img = cv2.warpAffine(img, trans,
                             (resW, resH), flags=cv2.INTER_LINEAR)
    return dst_img

def transformBox(pt, ul, br, inpH, inpW, resH, resW):
    center = np.zeros(2)
    center[0] = (br[0] - 1 - ul[0]) / 2
    center[1] = (br[1] - 1 - ul[1]) / 2

    lenH = max(br[1] - ul[1], (br[0] - ul[0]) * inpH / inpW)
    lenW = lenH * inpW / inpH

    _pt = np.zeros(2)
    _pt[0] = pt[0] - ul[0]
    _pt[1] = pt[1] - ul[1]
    # Move to center
    _pt[0] = _pt[0] + max(0, (lenW - 1) / 2 - center[0])
    _pt[1] = _pt[1] + max(0, (lenH - 1) / 2 - center[1])
    pt = (_pt * resH) / lenH
    pt[0] = round(float(pt[0]))
    pt[1] = round(float(pt[1]))
    return pt

def drawGaussian(img, pt, sigma, sig=1):
    tmpSize = 3 * sigma
    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - tmpSize), int(pt[1] - tmpSize)]
    br = [int(pt[0] + tmpSize + 1), int(pt[1] + tmpSize + 1)]

    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return img

    # Generate gaussian
    size = 2 * tmpSize + 1
    x = np.arange(0, size, 1, np.float32)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    sigma = size / 4.0
    # The gaussian is not normalized, we want the center value to equal 1
    g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * (sigma ** 2)))

    if sig < 0:
        g *= opt.spRate
    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return img

def alpha_pose_detection_processor(img, boxes, class_idxs, scores, thr=0.5):
    if len(boxes.shape) == 3:
        boxes = boxes.squeeze(axis=0)
    if len(class_idxs.shape) == 3:
        class_idxs = class_idxs.squeeze(axis=0)
    if len(scores.shape) == 3:
        scores = scores.squeeze(axis=0)

    # cilp coordinates
    boxes[:, [0, 2]] = mx.nd.clip(boxes[:, [0, 2]], 0., img.shape[1] - 1)
    boxes[:, [1, 3]] = mx.nd.clip(boxes[:, [1, 3]], 0., img.shape[0] - 1)

    # select boxes
    mask1 = (class_idxs == 0).asnumpy()
    mask2 = (scores > thr).asnumpy()
    picked_idxs = np.where((mask1 + mask2) > 1)[0]
    if picked_idxs.shape[0] == 0:
        return None, None
    else:
        return boxes[picked_idxs], scores[picked_idxs]

def alpha_pose_image_cropper(source_img, boxes, scores, output_shape=(256, 192)):
    if boxes is None:
        return None, boxes

    # crop person poses
    img_width, img_height = source_img.shape[1], source_img.shape[0]

    tensors = mx.nd.zeros([boxes.shape[0], 3, output_shape[0], output_shape[1]])
    out_boxes = np.zeros([boxes.shape[0], 4])

    for i, box in enumerate(boxes.asnumpy()):
        img = source_img.copy()
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
        ul = np.array((left, up))
        br = np.array((right, bottom))
        img = cv_cropBox(img, ul, br, output_shape[0], output_shape[1])

        img = mx.nd.image.to_tensor(mx.nd.array(img))
        # img = img.transpose((2, 0, 1))
        img[0] = img[0] - 0.406
        img[1] = img[1] - 0.457
        img[2] = img[2] - 0.480
        assert img.shape[0] == 3
        tensors[i] = img
        out_boxes[i] = (left, up, right, bottom)

    return tensors, out_boxes

def heatmap_to_coord_alpha_pose(hms, boxes):
    hm_h = hms.shape[2]
    hm_w = hms.shape[3]
    coords, maxvals = get_max_pred(hms)
    if boxes.shape[1] == 1:
        pt1 = mx.nd.array(boxes[:, 0, (0, 1)], dtype=hms.dtype)
        pt2 = mx.nd.array(boxes[:, 0, (2, 3)], dtype=hms.dtype)
    else:
        assert boxes.shape[1] == 4
        pt1 = mx.nd.array(boxes[:, (0, 1)], dtype=hms.dtype)
        pt2 = mx.nd.array(boxes[:, (2, 3)], dtype=hms.dtype)

    # post-processing
    for n in range(coords.shape[0]):
        for p in range(coords.shape[1]):
            hm = hms[n][p]
            px = int(nd.floor(coords[n][p][0] + 0.5).asscalar())
            py = int(nd.floor(coords[n][p][1] + 0.5).asscalar())
            if 1 < px < hm_w - 1 and 1 < py < hm_h - 1:
                diff = nd.concat(hm[py][px + 1] - hm[py][px - 1],
                                 hm[py + 1][px] - hm[py - 1][px],
                                 dim=0)
                coords[n][p] += nd.sign(diff) * .25

    preds = nd.zeros_like(coords)
    for i in range(hms.shape[0]):
        for j in range(hms.shape[1]):
            preds[i][j] = transformBoxInvert(coords[i][j], pt1[i], pt2[i], hm_h, hm_w)

    return preds, maxvals


def transformBoxInvert(pt, ul, br, resH, resW):
    # type: (Tensor, Tensor, Tensor, float, float, float, float) -> Tensor

    center = mx.nd.zeros(2)
    center[0] = (br[0] - 1 - ul[0]) / 2
    center[1] = (br[1] - 1 - ul[1]) / 2

    lenH = max(br[1] - ul[1], (br[0] - ul[0]) * resH / resW)
    lenW = lenH * resW / resH

    _pt = (pt * lenH) / resH

    if bool(((lenW - 1) / 2 - center[0]) > 0):
        _pt[0] = _pt[0] - ((lenW - 1) / 2 - center[0]).asscalar()
    if bool(((lenH - 1) / 2 - center[1]) > 0):
        _pt[1] = _pt[1] - ((lenH - 1) / 2 - center[1]).asscalar()

    new_point = mx.nd.zeros(2)
    new_point[0] = _pt[0] + ul[0]
    new_point[1] = _pt[1] + ul[1]
    return new_point


def detector_to_alpha_pose(img, class_ids, scores, bounding_boxs,
                           output_shape=(256, 192), ctx=mx.cpu(),
                           thr=0.5):
    boxes, scores = alpha_pose_detection_processor(
        img, bounding_boxs, class_ids, scores, thr=thr)
    pose_input, upscale_bbox = alpha_pose_image_cropper(
        img, boxes, scores, output_shape=output_shape)
    return pose_input, upscale_bbox
