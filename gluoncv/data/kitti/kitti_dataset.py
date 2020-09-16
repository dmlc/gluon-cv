"""KITTI Dataset. (KITTI Raw, KITTI Odom, KITTI Depth)
Vision meets Robotics: The KITTI Dataset, IJRR 2013
http://www.cvlibs.net/datasets/kitti/raw_data.php
Code partially borrowed from
https://github.com/nianticlabs/monodepth2/blob/master/datasets/kitti_dataset.py
"""
# pylint: disable=abstract-method, unused-import
# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import numpy as np
import PIL.Image as pil

from ...utils.filesystem import try_import_skimage
from .kitti_utils import generate_depth_map
from .mono_dataset import MonoDataset


class KITTIDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """

    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)

        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3,
                         "l": 2, "r": 3}

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class KITTIRAWDataset(KITTIDataset):
    """KITTI Raw Dataset.
    Parameters
    ----------
    data_path : string
        Path to KITTI RAW dataset folder. Default is '$(HOME)/.mxnet/datasets/kitti/kitti_data'

    Examples
    --------
    >>> from gluoncv.data.kitti.kitti_utils import dict_batchify_fn, readlines
    >>> train_filenames = os.path.join(
    >>>     os.path.expanduser("~"), '/.mxnet/datasets/kitti/splits/eigen_full/train_files.txt')
    >>> train_filenames = readlines(train_filenames)
    >>> # Create Dataset
    >>> trainset = gluoncv.data.KITTIRAWDataset(
    >>>         filenames=train_filenames, height=192, width=640,
    >>>         frame_idxs=[0], num_scales=4, is_train=True, img_ext='.png')
    >>> # Create Training Loader
    >>> train_data = gluon.data.DataLoader(
    >>>     trainset, batch_size=12, shuffle=True,
    >>>     batchify_fn=dict_batchify_fn, num_workers=12,
    >>>     pin_memory=True, last_batch='discard')
    """

    # pylint: disable=keyword-arg-before-vararg
    def __init__(self, data_path=os.path.join(
            os.path.expanduser("~"), '.mxnet/datasets/kitti/kitti_data'), *args, **kwargs):
        super(KITTIRAWDataset, self).__init__(data_path, *args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        skimage = try_import_skimage()
        from skimage import transform
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt


class KITTIOdomDataset(KITTIDataset):
    """KITTI dataset for odometry training and testing
    """

    # pylint: disable=keyword-arg-before-vararg
    def __init__(self, data_path=os.path.join(
            os.path.expanduser("~"), '.mxnet/datasets/kitti/kitti_odom'), *args, **kwargs):
        super(KITTIOdomDataset, self).__init__(data_path, *args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            "sequences/{:02d}".format(int(folder)),
            "image_{}".format(self.side_map[side]),
            f_str)
        return image_path


class KITTIDepthDataset(KITTIDataset):
    """KITTI dataset which uses the updated ground truth depth maps
    """

    def __init__(self, *args, **kwargs):
        super(KITTIDepthDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            folder,
            "image_0{}/data".format(self.side_map[side]),
            f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:010d}.png".format(frame_index)
        depth_path = os.path.join(
            self.data_path,
            folder,
            "proj_depth/groundtruth/image_0{}".format(self.side_map[side]),
            f_str)

        depth_gt = pil.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt
