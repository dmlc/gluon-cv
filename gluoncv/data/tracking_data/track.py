""" siamrpn dataloader,include youtube-bb,VID,DET,COCO dataset
Code adapted from https://github.com/STVIR/pysot """
# coding: utf-8
# pylint: disable=missing-docstring,unused-argument,arguments-differ
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import logging
import os
import numpy as np

from mxnet.gluon.data import dataset
from gluoncv.utils.filesystem import try_import_cv2
from gluoncv.model_zoo.siamrpn.siamrpn_tracker import corner2center, center2corner
from gluoncv.model_zoo.siamrpn.siamrpn_tracker import Corner, Center, Anchors

def IoU(rect1, rect2):
    """
    caculate interection over union

    Parameters
    ----------
    rect1: list or np.array, rectangle1
    rect2: list or np.array, rectangle2

    Returns
    -------
    iou
    """
    x1, y1, x2, y2 = rect1[0], rect1[1], rect1[2], rect1[3]
    tx1, ty1, tx2, ty2 = rect2[0], rect2[1], rect2[2], rect2[3]

    xx1 = np.maximum(tx1, x1)
    yy1 = np.maximum(ty1, y1)
    xx2 = np.minimum(tx2, x2)
    yy2 = np.minimum(ty2, y2)

    ww = np.maximum(0, xx2 - xx1)
    hh = np.maximum(0, yy2 - yy1)

    area = (x2-x1) * (y2-y1)
    target_a = (tx2-tx1) * (ty2 - ty1)
    inter = ww * hh
    iou = inter / (area + target_a - inter)
    return iou

class SubDataset(object):
    """Load the dataset for tracking.

    Parameters
    ----------
    name : str
        dataset name.
    root : str
        Path to the folder stored the dataset.
    anno : str
        Path to the Json stored detaset annotation
    frame_range : int
        dataset frame range that get positive pair.
    num_use : int
        if num_use = -1 show all dataset,show fisrt num_use dataset
    start_idx : int
        dataset start idx

    """
    def __init__(self, name, root, anno, frame_range, num_use, start_idx):
        self.name = name
        self.root = root
        self.anno = anno
        self.frame_range = frame_range
        self.num_use = num_use
        self.start_idx = start_idx
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("global")
        self.logger.info("loading %s", self.name)
        # load json
        with open(self.anno, 'r') as f:
            meta_data = json.load(f)
            meta_data = self._filter_zero(meta_data)

        for video in list(meta_data.keys()):
            for track in meta_data[video]:
                frames = meta_data[video][track]
                frames = list(map(int,
                                  filter(lambda x: x.isdigit(), frames.keys())))
                frames.sort()
                meta_data[video][track]['frames'] = frames
                if len(frames) <= 0:
                    self.logger.warning("%d/%d has no frames", int(video), int(track))
                    del meta_data[video][track]

        for video in list(meta_data.keys()):
            if len(meta_data[video]) <= 0:
                self.logger.warning("%d has no tracks", int(video))
                del meta_data[video]

        self.labels = meta_data
        self.num = len(self.labels)
        self.num_use = self.num if self.num_use == -1 else self.num_use
        self.videos = list(meta_data.keys())
        self.logger.info("%s loaded", self.name)
        self.path_format = '{}.{}.{}.jpg'
        self.pick = self.shuffle()

    def _filter_zero(self, meta_data):
        """
        Collate annotation and filter data

        Parameters
        ----------
            path
                meta_data

        Returns
            data after collate annotation and filter
        """
        meta_data_new = {}
        for video, tracks in meta_data.items():
            new_tracks = {}
            for trk, frames in tracks.items():
                new_frames = {}
                for frm, bbox in frames.items():
                    if not isinstance(bbox, dict):
                        if len(bbox) == 4:
                            x1, y1, x2, y2 = bbox
                            w, h = x2 - x1, y2 - y1
                        else:
                            w, h = bbox
                        if w <= 0 or h <= 0:
                            continue
                    new_frames[frm] = bbox
                if len(new_frames) > 0:
                    new_tracks[trk] = new_frames
            if len(new_tracks) > 0:
                meta_data_new[video] = new_tracks
        return meta_data_new

    def log(self):
        self.logger.info("%s start-index %d select [%d/%d] path_format %s",
                         self.name, self.start_idx, self.num_use, self.num, self.path_format)

    def shuffle(self):
        """shuffle data"""
        lists = list(range(self.start_idx, self.start_idx + self.num))
        pick = []
        while len(pick) < self.num_use:
            np.random.shuffle(lists)
            pick += lists
        return pick[:self.num_use]

    def get_image_anno(self, video, track, frame):
        """get image annotation

        Parameters
        ----------
            video : str
                video name
            track : str
                track number
            frame : str
                frame number
        Returns
            image_path and image bbox
        """
        frame = "{:06d}".format(frame)
        image_path = os.path.join(self.root, video,
                                  self.path_format.format(frame, track, 'x'))
        image_anno = self.labels[video][track][frame]
        return image_path, image_anno

    def get_positive_pair(self, index):
        """get positive pair every frame range

        Parameters
        ----------
            index : int
                per video_index

        Returns
            get positive pair template_frame and search_frame
        """
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(video.keys()))
        track_info = video[track]

        frames = track_info['frames']
        template_frame = np.random.randint(0, len(frames))
        left = max(template_frame - self.frame_range, 0)
        right = min(template_frame + self.frame_range, len(frames)-1) + 1
        search_range = frames[left:right]
        template_frame = frames[template_frame]
        search_frame = np.random.choice(search_range)
        return self.get_image_anno(video_name, track, template_frame), \
            self.get_image_anno(video_name, track, search_frame)

    def get_random_target(self, index=-1):
        """
        if neg, get image and annotation random target

        Returns
            get random frame
        """
        if index == -1:
            index = np.random.randint(0, self.num)
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(video.keys()))
        track_info = video[track]
        frames = track_info['frames']
        frame = np.random.choice(frames)
        return self.get_image_anno(video_name, track, frame)

    def __len__(self):
        return self.num

class TrkDataset(dataset.Dataset):
    """Load the dataset for tracking.

    Parameters
    ----------

    data_path : str
        Path to the folder stored the per dataset.
    dataset_names : list
        name to the folder stored the per dataset.
    detaset_root : list
        Path to the subfolder stored the per dataset.
    detaset_anno : list
        Path to the subanno stored the per dataset.
    dataset_frame_range : list
        dataset frame range that get positive pair.
    dataset_num_use : list
        if dataset_num_use = -1 show all dataset,show fisrt dataset_num_use dataset
    train_search_size : int
        train search size
    train_exemplar_size : int
        train exemplar size
    anchor_stride : int
        anchor stride
    anchor_ratios : int
        anchor ratios
    train_base_size : int
        train base size
    train_output_size : int
        train output size
    template_shift : int
        length of template augmentation shift
    template_scale : float
        template augmentation scale ratio
    template_blur : float
        template augmentation blur ratio
    template_flip : float
        template augmentation flip ratio
    template_color : float
        template augmentation color ratio
    search_shift : int
        length  of search augmentation shift ratio
    search_scale : float
        search augmentation shift ratio
    search_blur : float
        search augmentation blur ratio
    search_flip : float
        search augmentation filp ratio
    search_color : float
        search augmentation color ratio
    videos_per_epoch : int
        videos number per epoch
    train_epoch : int
        train epoch
    gray : float
        if gray=1, image graying
    neg : float
        negative ratio
    """
    def __init__(self,
                 data_path=os.path.expanduser('~/.mxnet/datasets'),
                 dataset_names=('vid', 'coco', 'det', 'yt_bb'),
                 detaset_root=('vid/crop511', 'coco/crop511', 'det/crop511', 'yt_bb/crop511'),
                 detaset_anno=('vid/train.json', 'coco/train2017.json', 'det/train.json',
                               'yt_bb/train.json'),
                 dataset_frame_range=(100, 1, 1, 3),
                 dataset_num_use=(100000, -1, -1, -1),
                 train_search_size=255,
                 train_exemplar_size=127,
                 anchor_stride=8,
                 anchor_ratios=(0.33, 0.5, 1, 2, 3),
                 train_base_size=8,
                 train_output_size=25,
                 template_shift=4,
                 template_scale=0.05,
                 template_blur=0,
                 template_flip=0,
                 template_color=1.0,
                 search_shift=64,
                 search_scale=0.18,
                 search_blur=0,
                 search_flip=0,
                 search_color=1.0,
                 videos_per_epoch=600000,
                 train_epoch=20,
                 gray=0.0,
                 neg=0.2,
                ):
        super(TrkDataset, self).__init__()
        self.train_search_size = train_search_size
        self.train_exemplar_size = train_exemplar_size
        self.anchor_stride = anchor_stride
        self.anchor_ratios = list(anchor_ratios)
        self.train_base_size = train_base_size
        self.train_output_size = train_output_size
        self.data_path = data_path
        self.dataset_names = list(dataset_names)
        self.detaset_root = list(detaset_root)
        self.detaset_anno = list(detaset_anno)
        self.dataset_frame_range = list(dataset_frame_range)
        self.dataset_num_use = list(dataset_num_use)
        self.template_shift = template_shift
        self.template_scale = template_scale
        self.template_blur = template_blur
        self.template_flip = template_flip
        self.template_color = template_color
        self.search_shift = search_shift
        self.search_scale = search_scale
        self.search_blur = search_blur
        self.search_flip = search_flip
        self.search_color = search_color
        self.videos_per_epoch = videos_per_epoch
        self.train_epoch = train_epoch
        self.gray = gray
        self.neg = neg
        self.cv2 = try_import_cv2()
        self.logger = logging.getLogger()
        desired_size = (self.train_search_size - self.train_exemplar_size) / \
            self.anchor_stride + 1 + self.train_base_size
        if desired_size != self.train_output_size:
            raise Exception('size not match!')

        # create anchor target
        self.anchor_target = AnchorTarget(anchor_stride=self.anchor_stride,
                                          anchor_ratios=self.anchor_ratios,
                                          train_search_size=self.train_search_size,
                                          train_output_size=self.train_output_size)
        print(self.anchor_target)
        # create sub dataset
        self.all_dataset = []
        start = 0
        self.num = 0

        for idx in range(len(self.dataset_names)):
            sub_dataset = SubDataset(self.dataset_names[idx],
                                     os.path.join(self.data_path, self.detaset_root[idx]),
                                     os.path.join(self.data_path, self.detaset_anno[idx]),
                                     self.dataset_frame_range[idx],
                                     self.dataset_num_use[idx],
                                     start)
            start += sub_dataset.num
            self.num += sub_dataset.num_use

            sub_dataset.log()
            self.all_dataset.append(sub_dataset)

        # data augmentation
        self.template_aug = Augmentation(self.template_shift,
                                         self.template_scale,
                                         self.template_blur,
                                         self.template_flip,
                                         self.template_color)
        self.search_aug = Augmentation(self.search_shift,
                                       self.search_scale,
                                       self.search_blur,
                                       self.search_flip,
                                       self.search_color)
        videos_per_epoch = self.videos_per_epoch
        self.num = videos_per_epoch if videos_per_epoch > 0 else self.num
        self.num *= self.train_epoch
        self.pick = self.shuffle()

    def shuffle(self):
        pick = []
        m = 0
        while m < self.num:
            p = []
            for sub_dataset in self.all_dataset:
                sub_p = sub_dataset.pick
                p += sub_p
            np.random.shuffle(p)
            pick += p
            m = len(pick)
        self.logger.info("shuffle done!")
        self.logger.info("dataset length %d", self.num)
        return pick[:self.num]

    def _find_dataset(self, index):
        for per_dataset in self.all_dataset:
            if per_dataset.start_idx + per_dataset.num > index:
                return per_dataset, index - per_dataset.start_idx
        return None, None

    def _get_bbox(self, image, shape):
        imh, imw = image.shape[:2]
        if len(shape) == 4:
            w, h = shape[2]-shape[0], shape[3]-shape[1]
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = self.train_exemplar_size
        wc_z = w + context_amount * (w+h)
        hc_z = h + context_amount * (w+h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w*scale_z
        h = h*scale_z
        cx, cy = imw//2, imh//2
        bbox = center2corner(Center(cx, cy, w, h))
        return bbox

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        index = self.pick[index]
        data, index = self._find_dataset(index)

        gray = self.gray and self.gray > np.random.random()
        neg = self.neg and self.neg > np.random.random()

        # get one dataset
        if neg:
            template = data.get_random_target(index)
            search = np.random.choice(self.all_dataset).get_random_target()
        else:
            template, search = data.get_positive_pair(index)

        # get image
        template_image = self.cv2.imread(template[0])
        search_image = self.cv2.imread(search[0])

        # get bounding box
        template_box = self._get_bbox(template_image, template[1])
        search_box = self._get_bbox(search_image, search[1])

        # augmentation
        template, _ = self.template_aug(template_image,
                                        template_box,
                                        self.train_exemplar_size,
                                        gray=gray)

        search, bbox = self.search_aug(search_image,
                                       search_box,
                                       self.train_search_size,
                                       gray=gray)

        # get labels
        cls, delta, delta_weight, _ = self.anchor_target(bbox, self.train_output_size, neg)
        template = template.transpose((2, 0, 1)).astype(np.float32)
        search = search.transpose((2, 0, 1)).astype(np.float32)

        return template, search, cls, delta, delta_weight, np.array(bbox)

class Augmentation:
    """dataset Augmentation for tracking.

    Parameters
    ----------
    shift : int
        length of template augmentation shift
    scale : float
        template augmentation scale ratio
    blur : float
        template augmentation blur ratio
    flip : float
        template augmentation flip ratio
    color : float
        template augmentation color ratio
    """
    def __init__(self, shift, scale, blur, flip, color):
        self.shift = shift
        self.scale = scale
        self.blur = blur
        self.flip = flip
        self.color = color
        self.rgbVar = np.array(
            [[-0.55919361, 0.98062831, - 0.41940627],
             [1.72091413, 0.19879334, - 1.82968581],
             [4.64467907, 4.73710203, 4.88324118]], dtype=np.float32)
        self.cv2 = try_import_cv2()

    @staticmethod
    def random():
        return np.random.random() * 2 - 1.0

    def _crop_roi(self, image, bbox, out_sz, padding=(0, 0, 0)):
        """crop image roi size.

        Parameters
        ----------
        image : np.array
            image
        bbox : list or np.array
            bbox coordinate，like (xmin,ymin,xmax,ymax)
        out_sz : int
            size after crop

        Return:
            image after crop
        """
        bbox = [float(x) for x in bbox]
        a = (out_sz-1) / (bbox[2]-bbox[0])
        b = (out_sz-1) / (bbox[3]-bbox[1])
        c = -a * bbox[0]
        d = -b * bbox[1]
        mapping = np.array([[a, 0, c],
                            [0, b, d]]).astype(np.float)
        crop = self.cv2.warpAffine(image, mapping, (out_sz, out_sz),
                                   borderMode=self.cv2.BORDER_CONSTANT,
                                   borderValue=padding)
        return crop

    def _blur_aug(self, image):
        """blur filter to smooth image

        Parameters
        ----------
        image : np.array
            image

        Return:
            image after blur
        """
        def rand_kernel():
            sizes = np.arange(5, 46, 2)
            size = np.random.choice(sizes)
            kernel = np.zeros((size, size))
            c = int(size/2)
            wx = np.random.random()
            kernel[:, c] += 1. / size * wx
            kernel[c, :] += 1. / size * (1-wx)
            return kernel
        kernel = rand_kernel()
        image = self.cv2.filter2D(image, -1, kernel)
        return image

    def _color_aug(self, image):
        """Random increase of image channel

        Parameters
        ----------
        image : np.array
            image

        Return:
            image after Random increase of channel
        """
        offset = np.dot(self.rgbVar, np.random.randn(3, 1))
        offset = offset[::-1]  # bgr 2 rgb
        offset = offset.reshape(3)
        image = image - offset
        return image

    def _gray_aug(self, image):
        """image Grayscale

        Parameters
        ----------
        image : np.array
            image

        Return:
            image after Grayscale
        """
        grayed = self.cv2.cvtColor(image, self.cv2.COLOR_BGR2GRAY)
        image = self.cv2.cvtColor(grayed, self.cv2.COLOR_GRAY2BGR)
        return image

    def _shift_scale_aug(self, image, bbox, crop_bbox, size):
        """shift scale augmentation

        Parameters
        ----------
        image : np.array
            image
        bbox : list or np.array
            bbox
        crop_bbox :
            crop size image from center

        Return
            image ,bbox after shift and scale
        """
        im_h, im_w = image.shape[:2]

        # adjust crop bounding box
        crop_bbox_center = corner2center(crop_bbox)

        if self.scale:
            scale_x = (1.0 + Augmentation.random() * self.scale)
            scale_y = (1.0 + Augmentation.random() * self.scale)
            h, w = crop_bbox_center.h, crop_bbox_center.w
            scale_x = min(scale_x, float(im_w) / w)
            scale_y = min(scale_y, float(im_h) / h)
            crop_bbox_center = Center(crop_bbox_center.x,
                                      crop_bbox_center.y,
                                      crop_bbox_center.w * scale_x,
                                      crop_bbox_center.h * scale_y)

        crop_bbox = center2corner(crop_bbox_center)
        if self.shift:
            sx = Augmentation.random() * self.shift
            sy = Augmentation.random() * self.shift

            x1, y1, x2, y2 = crop_bbox

            sx = max(-x1, min(im_w - 1 - x2, sx))
            sy = max(-y1, min(im_h - 1 - y2, sy))

            crop_bbox = Corner(x1 + sx, y1 + sy, x2 + sx, y2 + sy)

        # adjust target bounding box
        x1, y1 = crop_bbox.x1, crop_bbox.y1
        bbox = Corner(bbox.x1 - x1, bbox.y1 - y1,
                      bbox.x2 - x1, bbox.y2 - y1)

        if self.scale:
            bbox = Corner(bbox.x1 / scale_x, bbox.y1 / scale_y,
                          bbox.x2 / scale_x, bbox.y2 / scale_y)

        image = self._crop_roi(image, crop_bbox, size)
        return image, bbox

    def _flip_aug(self, image, bbox):
        """flip augmentation

        Parameters
        ----------
        image : np.array
            image
        bbox : list or np.array
            bbox

        Return
            image and bbox after filp
        """
        image = self.cv2.flip(image, 1)
        width = image.shape[1]
        bbox = Corner(width - 1 - bbox.x2, bbox.y1,
                      width - 1 - bbox.x1, bbox.y2)
        return image, bbox

    def __call__(self, image, bbox, size, gray=False):
        shape = image.shape
        crop_bbox = center2corner(Center(shape[0]//2, shape[1]//2,
                                         size-1, size-1))
        # gray augmentation
        if gray:
            image = self._gray_aug(image)

        # shift scale augmentation
        image, bbox = self._shift_scale_aug(image, bbox, crop_bbox, size)

        # color augmentation
        if self.color > np.random.random():
            image = self._color_aug(image)

        # blur augmentation
        if self.blur > np.random.random():
            image = self._blur_aug(image)

        # flip augmentation
        if self.flip and self.flip > np.random.random():
            image, bbox = self._flip_aug(image, bbox)
        return image, bbox

class AnchorTarget:
    def __init__(self, anchor_stride, anchor_ratios, train_search_size, train_output_size):
        """flip augmentation

        Parameters
        ----------
        image : np.array
            image
        bbox : list or np.array
            bbox

        Return
            image and bbox after filp
        """
        self.anchor_stride = anchor_stride
        self.anchor_ratios = anchor_ratios
        self.anchor_scales = [8]
        self.anchors = Anchors(self.anchor_stride,
                               self.anchor_ratios,
                               self.anchor_scales)
        self.train_search_size = train_search_size
        self.train_output_size = train_output_size

        self.anchors.generate_all_anchors(im_c=self.train_search_size//2,
                                          size=self.train_output_size)
        # Positive anchor threshold
        self.train_thr_high = 0.6
        # Negative anchor threshold
        self.thain_thr_low = 0.3
        # Number of negative
        self.train_pos_num = 16
        self.train_neg_num = 16
        self.train_total_num = 64

    def __call__(self, target, size, neg=False):
        anchor_num = len(self.anchor_ratios) * len(self.anchor_scales)

        # -1 ignore 0 negative 1 positive
        cls = -1 * np.ones((anchor_num, size, size), dtype=np.int64)
        delta = np.zeros((4, anchor_num, size, size), dtype=np.float32)
        delta_weight = np.zeros((anchor_num, size, size), dtype=np.float32)

        def select(position, keep_num=16):
            num = position[0].shape[0]
            if num <= keep_num:
                return position, num
            slt = np.arange(num)
            np.random.shuffle(slt)
            slt = slt[:keep_num]
            return tuple(p[slt] for p in position), keep_num

        tcx, tcy, tw, th = corner2center(target)

        if neg:
            # l = size // 2 - 3
            # r = size // 2 + 3 + 1
            # cls[:, l:r, l:r] = 0

            cx = size // 2
            cy = size // 2
            cx += int(np.ceil((tcx - self.train_search_size // 2) /
                              self.anchor_stride + 0.5))
            cy += int(np.ceil((tcy - self.train_search_size // 2) /
                              self.anchor_stride + 0.5))
            l = max(0, cx - 3)
            r = min(size, cx + 4)
            u = max(0, cy - 3)
            d = min(size, cy + 4)
            cls[:, u:d, l:r] = 0

            neg, _ = select(np.where(cls == 0), self.train_neg_num)
            cls[:] = -1
            cls[neg] = 0

            overlap = np.zeros((anchor_num, size, size), dtype=np.float32)
            return cls, delta, delta_weight, overlap

        anchor_box = self.anchors.all_anchors[0]
        anchor_center = self.anchors.all_anchors[1]

        x1, y1, x2, y2 = anchor_box[0], anchor_box[1], \
            anchor_box[2], anchor_box[3]
        cx, cy, w, h = anchor_center[0], anchor_center[1], \
            anchor_center[2], anchor_center[3]

        delta[0] = (tcx - cx) / w
        delta[1] = (tcy - cy) / h
        delta[2] = np.log(tw / w)
        delta[3] = np.log(th / h)

        overlap = IoU([x1, y1, x2, y2], target)

        pos = np.where(overlap > self.train_thr_high)
        neg = np.where(overlap < self.thain_thr_low)

        pos, pos_num = select(pos, self.train_pos_num)
        neg, _ = select(neg, self.train_total_num - self.train_pos_num)

        cls[pos] = 1
        delta_weight[pos] = 1. / (pos_num + 1e-6)

        cls[neg] = 0
        return cls, delta, delta_weight, overlap
