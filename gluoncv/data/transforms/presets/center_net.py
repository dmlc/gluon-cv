"""Transforms described in https://arxiv.org/abs/1904.07850."""
from __future__ import absolute_import
import numpy as np
import mxnet as mx
from .. import bbox as tbbox
from .. import image as timage
from .. import experimental
from ...utils.filesystem import try_import_cv2

__all__ = ['CenterNetDefaultTrainTransform', 'CenterNetDefaultValTransform']


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def _get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    cv2 = try_import_cv2()
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

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
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def _affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

class CenterNetDefaultTrainTransform(object):
    """Default SSD training transform which includes tons of image augmentations.

    Parameters
    ----------
    width : int
        Image width.
    height : int
        Image height.
    num_class : int
        Number of categories
    scale_factor : int, default is 4
        The downsampling scale factor between input image and output heatmap
    mean : array-like of size 3
        Mean pixel values to be subtracted from image tensor. Default is [0.485, 0.456, 0.406].
    std : array-like of size 3
        Standard deviation to be divided from image. Default is [0.229, 0.224, 0.225].
    """
    def __init__(self, width, height, num_class, scale_factor=4, mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225), **kwargs):
        self._width = width
        self._height = height
        self._num_class = num_class
        self._scale_factor = scale_factor
        self._mean = mean
        self._std = std

        from ....model_zoo.center_net.target_generator import CenterNetTargetGenerator
        self._target_generator = CenterNetTargetGenerator(
            num_class, width // scale_factor, height // scale_factor)

    def __call__(self, src, label):
        """Apply transform to training image/label."""
        # random color jittering
        # img = experimental.image.random_color_distort(src)
        img = src
        bbox = label

        # random expansion with prob 0.5
        # if np.random.uniform(0, 1) > 0.5:
        #     img, expand = timage.random_expand(img, fill=[m * 255 for m in self._mean])
        #     bbox = tbbox.translate(label, x_offset=expand[0], y_offset=expand[1])
        # else:
        #     img, bbox = img, label

        # random cropping
        # h, w, _ = img.shape
        # bbox, crop = experimental.bbox.random_crop_with_constraints(bbox, (w, h))
        # x0, y0, w, h = crop
        # img = mx.image.fixed_crop(img, x0, y0, w, h)

        # random horizontal flip
        h, w, _ = img.shape
        img, flips = timage.random_flip(img, px=0.5)
        bbox = tbbox.flip(bbox, (w, h), flip_x=flips[0])

        # resize with random interpolation
        h, w, _ = img.shape
        # interp = np.random.randint(0, 5)
        # img = timage.imresize(img, self._width, self._height, interp=interp)
        # bbox = tbbox.resize(bbox, (w, h), (self._width, self._height))
        cv2 = try_import_cv2()
        input_h, input_w = self._height, self._width
        s = max(h, w) * 1.0
        c = np.array([w / 2., h / 2.], dtype=np.float32)
        sf = 0.4
        cf = 0.1
        c[0] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        c[1] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
        trans_input = _get_affine_transform(c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img.asnumpy(), trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)
        output_w = input_w // self._scale_factor
        output_h = input_h // self._scale_factor
        trans_output = _get_affine_transform(c, s, 0, [output_w, output_h])
        for i in range(bbox.shape[0]):
            bbox[i, :2] = _affine_transform(bbox[i, :2], trans_output)
            bbox[i, 2:4] = _affine_transform(bbox[i, 2:4], trans_output)
        bbox[:, :2] = np.clip(bbox[:, :2], 0, output_w - 1)
        bbox[:, 2:4] = np.clip(bbox[:, 2:4], 0, output_h - 1)
        img = mx.nd.array(inp)

        # to tensor
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)

        # generate training target so cpu workers can help reduce the workload on gpu
        gt_bboxes = bbox[:, :4]
        gt_ids = bbox[:, 4:5]
        heatmap, wh_target, wh_mask, center_reg, center_reg_mask = self._target_generator(
            img.shape[2], img.shape[1], gt_bboxes, gt_ids)
        return img, heatmap, wh_target, wh_mask, center_reg, center_reg_mask


class CenterNetDefaultValTransform(object):
    """Default SSD validation transform.

    Parameters
    ----------
    width : int
        Image width.
    height : int
        Image height.
    mean : array-like of size 3
        Mean pixel values to be subtracted from image tensor. Default is [0.485, 0.456, 0.406].
    std : array-like of size 3
        Standard deviation to be divided from image. Default is [0.229, 0.224, 0.225].

    """
    def __init__(self, width, height, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self._width = width
        self._height = height
        self._mean = mean
        self._std = std

    def __call__(self, src, label):
        """Apply transform to validation image/label."""
        # resize
        h, w, _ = src.shape
        img = timage.imresize(src, self._width, self._height, interp=9)
        bbox = tbbox.resize(label, in_size=(w, h), out_size=(self._width, self._height))

        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)
        return img, bbox.astype(img.dtype)