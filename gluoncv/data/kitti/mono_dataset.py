"""Monocular Depth Estimation Dataset.
Digging into Self-Supervised Monocular Depth Prediction, ICCV 2019
https://arxiv.org/abs/1806.01260
Code partially borrowed from
https://github.com/nianticlabs/monodepth2/blob/master/datasets/mono_dataset.py
"""
import random
import copy
import numpy as np
from PIL import Image  # using pillow-simd for increased speed

import mxnet as mx
from mxnet.gluon.data import dataset
from mxnet.gluon.data.vision import transforms


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class MonoDataset(dataset.Dataset):
    """Superclass for monocular dataloaders
    Parameters
    ----------
    data_path : string
        Path to dataset folder.
    filenames : string
        Path to split file.
        For example: '$(HOME)/.mxnet/datasets/kitti/splits/eigen_full/train_files.txt'
    height : int
        The height for input images.
    width : int
        The height for input images.
    frame_idxs : list
        The frames to load.
        an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or "s" for the opposite image in the stereo pair.
    num_scales : int
        The number of scales of the image relative to the full-size image.
    is_train : bool
        Whether use Data Augmentation. Default is: False
    img_ext : string
        The extension name of input image. Default is '.jpg'
    """

    def __init__(self, data_path, filenames, height, width, frame_idxs,
                 num_scales, is_train=False, img_ext='.jpg'):
        super(MonoDataset, self).__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        self.brightness = 0.2
        self.contrast = 0.2
        self.saturation = 0.2
        self.hue = 0.1

        self.load_depth = self.check_depth()

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    s = 2 ** i
                    size = (self.height // s, self.width // s)
                    inputs[(n, im, i)] = copy.deepcopy(
                        inputs[(n, im, i - 1)].resize(size[::-1], self.interp))

        for k in list(inputs):
            f = mx.nd.array(inputs[k])
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to mxnet NDArray.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the full-size image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = False  # self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        folder = line[0]

        if len(line) == 3:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None

        for i in self.frame_idxs:
            if i == "s":
                other_side = {"r": "l", "l": "r"}[side]
                inputs[("color", i, -1)] = self.get_color(
                    folder, frame_index, other_side, do_flip)
            else:
                inputs[("color", i, -1)] = self.get_color(
                    folder, frame_index + i, side, do_flip)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = mx.nd.array(K)

            inputs[("inv_K", scale)] = mx.nd.array(inv_K)

        if do_color_aug:
            color_aug = transforms.RandomColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = mx.nd.array(inputs["depth_gt"].astype(np.float32))

        if "s" in self.frame_idxs:
            stereo_T = np.eye(4, dtype=np.float32)
            baseline_sign = -1 if do_flip else 1
            side_sign = -1 if side == "l" else 1
            stereo_T[0, 3] = side_sign * baseline_sign * 0.1

            inputs["stereo_T"] = mx.nd.array(stereo_T)

        return inputs

    def get_color(self, folder, frame_index, side, do_flip):
        raise NotImplementedError

    def check_depth(self):
        raise NotImplementedError

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
