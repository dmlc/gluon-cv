""" siamrpn_tracker """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import argparse
import math
from collections import namedtuple
import numpy as np
import mxnet as mx
from mxnet import nd, cpu, gpu
from gluoncv.utils.filesystem import try_import_cv2

def parse_args():
    """siamrpn_tracker test."""
    parser = argparse.ArgumentParser(description='siamrpn tracker')
    parser.add_argument('--PENALTY_K', default=0.16, type=float,
                        help='datasets')
    parser.add_argument('--WINDOW_INFLUENCE', default=0.40, type=float,
                        help='config file')
    parser.add_argument('--LR', default=0.30, type=float,
                        help='snapshot of models to eval')
    parser.add_argument('--EXEMPLAR_SIZE', default=127, type=int,
                        help='EXEMPLAR_SIZE')
    parser.add_argument('--INSTANCE_SIZE', default=287, type=int,
                        help='INSTANCE_SIZE')
    parser.add_argument('--BASE_SIZE', default=0, type=int,
                        help='eval one special video')
    parser.add_argument('--CONTEXT_AMOUNT', default=0.5, type=float,
                        help='eval one special video')
    parser.add_argument('--CUDA', default=True, type=str,
                        help='eval one special video')
    parser.add_argument('--STRIDE', default=8, type=int,
                        help='eval one special video')
    parser.add_argument('--RATIOS', default=[0.33, 0.5, 1, 2, 3], type=list,
                        help='eval one special video')
    parser.add_argument('--SCALES', default=[8], type=list,
                        help='eval one special video')
    parser.add_argument('--ANCHOR_NUM', default=5, type=int,
                        help='eval one special video')
    parser.add_argument('--num-gpus', default=0, type=int,
                        help='number of gpus to use.')
    opt = parser.parse_args()
    return opt

Corner = namedtuple('Corner', 'x1 y1 x2 y2')
BBox = Corner
Center = namedtuple('Center', 'x y w h')

def get_axis_aligned_bbox(region):
    """ convert region to (center_x, center_y, bbox_w, bbox_h)
        that represent by axis aligned box
    Args:
        conrner: region or np.array (4*N)
    Return:
        Center or np.array (4 * N)"""
    region_x = region[0]
    region_y = region[1]
    bbox_w = region[2]
    bbox_h = region[3]
    center_x = region_x+bbox_w/2
    center_y = region_y+bbox_h/2
    return center_x, center_y, bbox_w, bbox_h

def corner2center(corner):
    """ convert (x1, y1, x2, y2) to (cx, cy, w, h)
    Args:
        conrner: Corner or np.array (4*N)
    Return:
        Center or np.array (4 * N)
    """
    if isinstance(corner, Corner):
        x_min, y_min, x_max, y_max = corner
        return Center((x_min + x_max) * 0.5, (y_min + y_max) * 0.5,
                      (x_max - x_min), (y_max - y_min))
    else:
        x_min, y_min, x_max, y_max = corner[0], corner[1], corner[2], corner[3]
        center_x = (x_min + x_max) * 0.5
        center_y = (y_min + y_max) * 0.5
        bbox_w = x_max - x_min
        bbox_h = y_max - y_min
        return center_x, center_y, bbox_w, bbox_h


def center2corner(center):
    """ convert (cx, cy, w, h) to (x1, y1, x2, y2)
    Args:
        center: Center or np.array (4 * N)
    Return:
        center or np.array (4 * N)
    """
    if isinstance(center, Center):
        center_x, center_y, bbox_w, bbox_h = center
        return Corner(center_x - bbox_w * 0.5, center_y - bbox_h * 0.5,
                      center_x + bbox_w * 0.5, center_y + bbox_h * 0.5)
    else:
        center_x, center_y, bbox_w, bbox_h = center[0], center[1], center[2], center[3]
        x_min = center_x - bbox_w * 0.5
        y_min = center_y - bbox_h * 0.5
        x_max = center_x + bbox_w * 0.5
        y_max = center_y + bbox_h * 0.5
        return x_min, y_min, x_max, y_max

class Anchors():
    """This class generate anchors."""
    def __init__(self, stride, ratios, scales, image_center=0, size=0):
        self.stride = stride
        self.ratios = ratios
        self.scales = scales
        self.image_center = image_center
        self.size = size
        self.anchor_num = len(self.scales) * len(self.ratios)
        self.anchors = None
        self.generate_anchors()

    def generate_anchors(self):
        """generate anchors based on predefined configuration"""
        self.anchors = np.zeros((self.anchor_num, 4), dtype=np.float32)
        size = self.stride * self.stride
        count = 0
        for per_r in self.ratios:
            w_s = int(math.sqrt(size*1. / per_r))
            h_s = int(w_s * per_r)
            for per_s in self.scales:
                a_w = w_s * per_s
                a_h = h_s * per_s
                self.anchors[count][:] = [-a_w*0.5, -a_h*0.5, a_w*0.5, a_h*0.5][:]
                count += 1


class BaseTracker(object):
    """ Base tracker of single objec tracking """
    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox(list): [x, y, width, height]
                        x, y need to be 0-based
        """
        raise NotImplementedError

    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        raise NotImplementedError

class SiamRPNTracker(BaseTracker):
    """ SiamRPNTracker"""
    def __init__(self, model):
        super(SiamRPNTracker, self).__init__()
        self.opt = parse_args()
        self.score_size = (self.opt.INSTANCE_SIZE - self.opt.EXEMPLAR_SIZE) // \
            self.opt.STRIDE + 1 + self.opt.BASE_SIZE
        self.anchor_num = len(self.opt.RATIOS) * len(self.opt.SCALES)
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.anchors = self.generate_anchor(self.score_size)
        self.model = model
        self.channel_average = None
        self.size = None
        self.center_pos = None

    def generate_anchor(self, score_size):
        """generate_anchor"""
        anchors = Anchors(self.opt.STRIDE,
                          self.opt.RATIOS,
                          self.opt.SCALES)
        anchor = anchors.anchors
        x_min, y_min, x_max, y_max = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x_min+x_max)*0.5, (y_min+y_max)*0.5, x_max-x_min, y_max-y_min], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        x_x, y_y = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                               [ori + total_stride * dy for dy in range(score_size)])
        x_x, y_y = np.tile(x_x.flatten(), (anchor_num, 1)).flatten(), \
            np.tile(y_y.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = x_x.astype(np.float32), y_y.astype(np.float32)
        return anchor

    def _convert_bbox(self, delta, anchor):
        """from loc to predict postion"""
        delta = nd.transpose(delta, axes=(1, 2, 3, 0))
        delta = nd.reshape(delta, shape=(4, -1))
        delta = delta.asnumpy()
        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * anchor[:, 3]
        return delta

    def _convert_score(self, score):
        """from cls to score"""
        score = nd.transpose(score, axes=(1, 2, 3, 0))
        score = nd.reshape(score, shape=(2, -1))
        score = nd.transpose(score, axes=(1, 0))
        score = nd.softmax(score, axis=1).asnumpy()[:, 1]
        return score

    def _bbox_clip(self, center_x, center_y, width, height, boundary):
        """get bbox in image """
        center_x = max(0, min(center_x, boundary[1]))
        center_y = max(0, min(center_y, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return center_x, center_y, width, height

    def get_subwindow(self, img, pos, model_sz, original_sz, avg_chans):
        """
        function
        ----------
            Adjust the position of the frame to prevent the boundary from being exceeded.
            If the boundary is exceeded,
            the average value of each channel of the image is used to replace the exceeded value.
        args:
            im: BGR based image
            pos: center position
            model_sz: exemplar size,x is 127, z is 287 in ours
            s_z: original size
            avg_chans: channel average
        """
        cv2 = try_import_cv2()
        if isinstance(pos, float):
            pos = [pos, pos]
        im_sz = img.shape
        original_c = (original_sz + 1) / 2
        context_xmin = np.floor(pos[0] - original_c + 0.5)
        context_xmax = context_xmin + original_sz - 1
        context_ymin = np.floor(pos[1] - original_c + 0.5)
        context_ymax = context_ymin + original_sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad
        im_h, im_w, im_c = img.shape

        if any([top_pad, bottom_pad, left_pad, right_pad]):
            #If there is a pad, use the average channels to complete
            size = (im_h + top_pad + bottom_pad, im_w + left_pad + right_pad, im_c)
            te_im = np.zeros(size, np.uint8)
            te_im[top_pad:top_pad + im_h, left_pad:left_pad + im_w, :] = img
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + im_w, :] = avg_chans
            if bottom_pad:
                te_im[im_h + top_pad:, left_pad:left_pad + im_w, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, im_w + left_pad:, :] = avg_chans
            im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                             int(context_xmin):int(context_xmax + 1), :]
        else:
            #If there is no pad, crop Directly
            im_patch = img[int(context_ymin):int(context_ymax + 1),
                           int(context_xmin):int(context_xmax + 1), :]
        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))
        im_patch = im_patch.transpose(2, 0, 1)
        im_patch = im_patch[np.newaxis, :, :, :]
        im_patch = im_patch.astype(np.float32)
        if self.opt.num_gpus > 0:
            im_patch = mx.nd.array(im_patch, ctx=gpu())
        else:
            im_patch = mx.nd.array(im_patch, ctx=cpu())
        return im_patch

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        # center postion(x,y)
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        # size is (w,h)
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + self.opt.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + self.opt.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    self.opt.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        self.model.template(z_crop)

    def track(self, img):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        # calculate z crop size
        w_z = self.size[0] + self.opt.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + self.opt.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = self.opt.EXEMPLAR_SIZE / s_z
        s_x = s_z * (self.opt.INSTANCE_SIZE / self.opt.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    self.opt.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)
        outputs = self.model.track(x_crop)
        #get cls
        score = self._convert_score(outputs['cls'])
        #get coordinate
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)
        def change(hw_r):
            return np.maximum(hw_r, 1. / hw_r)

        def get_scale(bbox_w, bbox_h):
            pad = (bbox_w + bbox_h) * 0.5
            return np.sqrt((bbox_w + pad) * (bbox_h + pad))

        # scale penalty，
        # compare the overall scale of the proposal and last frame
        s_c = change(get_scale(pred_bbox[2, :], pred_bbox[3, :]) /
                     (get_scale(self.size[0]*scale_z, self.size[1]*scale_z)))

        # aspect ratio penalty. ratio is ratio of height and width.
        # compare proposal’s ratio and last frame
        r_c = change((self.size[0]/self.size[1]) /
                     (pred_bbox[2, :]/pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * self.opt.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - self.opt.WINDOW_INFLUENCE) + \
            self.window * self.opt.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / scale_z
        penalty_lr = penalty[best_idx] * score[best_idx] * self.opt.LR

        center_x = bbox[0] + self.center_pos[0]
        center_y = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - penalty_lr) + bbox[2] * penalty_lr
        height = self.size[1] * (1 - penalty_lr) + bbox[3] * penalty_lr

        # clip boundary
        center_x, center_y, width, height = self._bbox_clip(center_x, center_y, width,
                                                            height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([center_x, center_y])
        self.size = np.array([width, height])

        bbox = [center_x - width / 2,
                center_y - height / 2,
                width,
                height]
        best_score = score[best_idx]
        return {'bbox': bbox,
                'best_score': best_score}

