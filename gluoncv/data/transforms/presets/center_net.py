"""Transforms described in https://arxiv.org/abs/1904.07850."""
from __future__ import absolute_import
import random
import numpy as np
import mxnet as mx
from .. import bbox as tbbox
from .. import image as timage
from .. import experimental
from ....utils.filesystem import try_import_cv2

__all__ = ['CenterNetDefaultTrainTransform', 'CenterNetDefaultValTransform']

def _get_border(border, size):
    i = 1
    while size - border // i <= border // i:
        i *= 2
    return border // i

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

def grayscale(image):
    cv2 = try_import_cv2()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def lighting_(data_rng, image, alphastd, eigval, eigvec):
    alpha = data_rng.normal(scale=alphastd, size=(3, ))
    image += np.dot(eigvec, eigval * alpha)

def blend_(alpha, image1, image2):
    image1 *= alpha
    image2 *= (1 - alpha)
    image1 += image2

def saturation_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs[:, :, None])

def brightness_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    image *= alpha

def contrast_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs_mean)

def color_aug(data_rng, image, eig_val, eig_vec):
    functions = [brightness_, contrast_, saturation_]
    random.shuffle(functions)

    gs = grayscale(image)
    gs_mean = gs.mean()
    for f in functions:
        f(data_rng, image, gs, gs_mean, 0.4)
    lighting_(data_rng, image, 0.1, eig_val, eig_vec)

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
        self._mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self._std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)

        from ....model_zoo.center_net.target_generator import CenterNetTargetGenerator
        self._target_generator = CenterNetTargetGenerator(
            num_class, width // scale_factor, height // scale_factor)

    def __call__(self, src, label):
        """Apply transform to training image/label."""
        # random color jittering
        img = src
        bbox = label

        # random horizontal flip
        h, w, _ = img.shape
        img, flips = timage.random_flip(img, px=0.5)
        bbox = tbbox.flip(bbox, (w, h), flip_x=flips[0])

        cv2 = try_import_cv2()
        input_h, input_w = self._height, self._width
        s = max(h, w) * 1.0
        c = np.array([w / 2., h / 2.], dtype=np.float32)
        sf = 0.4
        cf = 0.1
        w_border = _get_border(128, img.shape[1])
        h_border = _get_border(128, img.shape[0])
        c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
        c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
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
        img = inp

        # to tensor
        img = img.astype(np.float32) / 255.
        color_aug(self._data_rng, img, self._eig_val, self._eig_vec)
        img = (img - self._mean) / self._std
        img = img.transpose(2, 0, 1).astype(np.float32)
        img = mx.nd.array(img)

        # generate training target so cpu workers can help reduce the workload on gpu
        gt_bboxes = bbox[:, :4]
        gt_ids = bbox[:, 4:5]
        heatmap, wh_target, wh_mask, center_reg, center_reg_mask = self._target_generator(
            img.shape[2], img.shape[1], gt_bboxes, gt_ids)
        return img, heatmap, wh_target, wh_mask, center_reg, center_reg_mask

class CenterNetDefaultTrainTransformDebug(object):
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
        self._mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self._std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)

        from ....model_zoo.center_net.target_generator import CenterNetTargetGeneratorDebug
        self._target_generator = CenterNetTargetGeneratorDebug(
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
        # h, w, _ = img.shape
        # interp = np.random.randint(0, 5)
        # img = timage.imresize(img, self._width, self._height, interp=interp)
        # bbox = tbbox.resize(bbox, (w, h), (self._width, self._height))
        cv2 = try_import_cv2()
        input_h, input_w = self._height, self._width
        s = max(h, w) * 1.0
        c = np.array([w / 2., h / 2.], dtype=np.float32)
        sf = 0.4
        cf = 0.1
        w_border = _get_border(128, img.shape[1])
        h_border = _get_border(128, img.shape[0])
        c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
        c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
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
        img = inp[:, :, (2, 1, 0)]    # To BGR?

        # to tensor
        img = img.astype(np.float32) / 255.
        color_aug(self._data_rng, img, self._eig_val, self._eig_vec)
        img = (img - self._mean) / self._std
        img = img.transpose(2, 0, 1).astype(np.float32)
        # img = mx.nd.image.to_tensor(img)
        # img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)
        # img = img.asnumpy()

        # generate training target so cpu workers can help reduce the workload on gpu
        gt_bboxes = bbox[:, :4]
        gt_ids = bbox[:, 4:5]
        ret = self._target_generator(img.shape[2], img.shape[1], gt_bboxes, gt_ids)
        ret.update({'input': img})
        return ret


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
        self._mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self._std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    def __call__(self, src, label):
        """Apply transform to validation image/label."""
        # resize
        img, bbox = src, label
        cv2 = try_import_cv2()
        input_h, input_w = self._height, self._width
        h, w, _ = src.shape
        s = max(h, w) * 1.0
        c = np.array([w / 2., h / 2.], dtype=np.float32)
        trans_input = _get_affine_transform(c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img.asnumpy(), trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)
        output_w = input_w
        output_h = input_h
        trans_output = _get_affine_transform(c, s, 0, [output_w, output_h])
        for i in range(bbox.shape[0]):
            bbox[i, :2] = _affine_transform(bbox[i, :2], trans_output)
            bbox[i, 2:4] = _affine_transform(bbox[i, 2:4], trans_output)
        bbox[:, :2] = np.clip(bbox[:, :2], 0, output_w - 1)
        bbox[:, 2:4] = np.clip(bbox[:, 2:4], 0, output_h - 1)
        img = inp

        # to tensor
        img = img.astype(np.float32) / 255.
        img = (img - self._mean) / self._std
        img = img.transpose(2, 0, 1).astype(np.float32)
        img = mx.nd.array(img)
        return img, bbox.astype(img.dtype)
