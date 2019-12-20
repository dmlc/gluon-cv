# pylint: disable=missing-docstring,arguments-differ
"""Extended image transformations to video transformations.
Code partially borrowed from https://github.com/bryanyzhu/two-stream-pytorch"""
from __future__ import division
import random
import numbers
import numpy as np
from mxnet.gluon import Block

__all__ = ['VideoToTensor', 'VideoNormalize', 'VideoRandomHorizontalFlip', 'VideoMultiScaleCrop',
           'VideoCenterCrop', 'VideoTenCrop', 'VideoGroupTrainTransform', 'VideoGroupValTransform']

class VideoGroupValTransform(Block):
    """Combination of transforms for validation.
        (1) center crop
        (2) to tensor
        (3) normalize
    """

    def __init__(self, size, mean, std, max_intensity=255.0):
        super(VideoGroupValTransform, self).__init__()
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.mean = np.asarray(mean).reshape((len(mean), 1, 1))
        self.std = np.asarray(std).reshape((len(std), 1, 1))
        self.max_intensity = max_intensity

    def forward(self, clips):
        h, w, _ = clips[0].shape
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))

        new_clips = []
        for cur_img in clips:
            crop_img = cur_img[y1:y1+th, x1:x1+tw, :]
            tensor_img = np.transpose(crop_img, axes=(2, 0, 1)) / self.max_intensity
            new_clips.append((tensor_img - self.mean) / self.std)
        return new_clips

class VideoGroupTrainTransform(Block):
    """Combination of transforms for training.
        (1) multiscale crop
        (2) scale
        (3) random horizontal flip
        (4) to tensor
        (5) normalize
    """
    def __init__(self, size, scale_ratios, mean, std, fix_crop=True,
                 more_fix_crop=True, max_distort=1, prob=0.5, max_intensity=255.0):
        super(VideoGroupTrainTransform, self).__init__()

        from ...utils.filesystem import try_import_cv2
        self.cv2 = try_import_cv2()
        self.height = size[0]
        self.width = size[1]
        self.scale_ratios = scale_ratios
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.max_distort = max_distort
        self.prob = prob
        self.max_intensity = max_intensity
        self.mean = np.asarray(mean).reshape((len(mean), 1, 1))
        self.std = np.asarray(std).reshape((len(std), 1, 1))

    def fillFixOffset(self, datum_height, datum_width):
        h_off = int((datum_height - self.height) / 4)
        w_off = int((datum_width - self.width) / 4)

        offsets = []
        offsets.append((0, 0))          # upper left
        offsets.append((0, 4*w_off))    # upper right
        offsets.append((4*h_off, 0))    # lower left
        offsets.append((4*h_off, 4*w_off))  # lower right
        offsets.append((2*h_off, 2*w_off))  # center

        if self.more_fix_crop:
            offsets.append((0, 2*w_off))        # top center
            offsets.append((4*h_off, 2*w_off))  # bottom center
            offsets.append((2*h_off, 0))        # left center
            offsets.append((2*h_off, 4*w_off))  # right center

            offsets.append((1*h_off, 1*w_off))  # upper left quarter
            offsets.append((1*h_off, 3*w_off))  # upper right quarter
            offsets.append((3*h_off, 1*w_off))  # lower left quarter
            offsets.append((3*h_off, 3*w_off))  # lower right quarter

        return offsets

    def fillCropSize(self, input_height, input_width):
        crop_sizes = []
        base_size = np.min((input_height, input_width))
        scale_rates = self.scale_ratios
        for h, scale_rate_h in enumerate(scale_rates):
            crop_h = int(base_size * scale_rate_h)
            for w, scale_rate_w in enumerate(scale_rates):
                crop_w = int(base_size * scale_rate_w)
                if (np.absolute(h - w) <= self.max_distort):
                    crop_sizes.append((crop_h, crop_w))

        return crop_sizes

    def forward(self, clips):
        h, w, _ = clips[0].shape

        crop_size_pairs = self.fillCropSize(h, w)
        size_sel = random.randint(0, len(crop_size_pairs)-1)
        crop_height = crop_size_pairs[size_sel][0]
        crop_width = crop_size_pairs[size_sel][1]

        is_flip = random.random() < self.prob
        if self.fix_crop:
            offsets = self.fillFixOffset(h, w)
            off_sel = random.randint(0, len(offsets)-1)
            h_off = offsets[off_sel][0]
            w_off = offsets[off_sel][1]
        else:
            h_off = random.randint(0, h - self.height)
            w_off = random.randint(0, w - self.width)

        new_clips = []
        for cur_img in clips:
            crop_img = cur_img[h_off:h_off+crop_height, w_off:w_off+crop_width, :]
            scale_img = self.cv2.resize(crop_img, (self.width, self.height))
            if is_flip:
                flip_img = np.flip(scale_img, axis=1)
            else:
                flip_img = scale_img
            tensor_img = np.transpose(flip_img, axes=(2, 0, 1)) / self.max_intensity
            new_clips.append((tensor_img - self.mean) / self.std)
        return new_clips

class VideoToTensor(Block):
    """Convert images to tensor.

    Convert a list of images of shape (H x W x C) in the range
    [0, 255] to a float32 tensor of shape (C x H x W) in
    the range [0, 1).

    Parameters
    ----------
    max_intensity : float
        The maximum intensity value to be divided in order to fit the output tensor
        in the range [0, 1).

    Inputs:
        - **data**: a list of frames with shape [H x W x C] and uint8 type

    Outputs:
        - **out**: a list of frames with shape [C x H x W] and float32 type
    """
    def __init__(self, max_intensity=255.0):
        super(VideoToTensor, self).__init__()
        self.max_intensity = max_intensity

    def forward(self, clips):
        new_clips = []
        for cur_img in clips:
            new_clips.append(np.transpose(cur_img, axes=(2, 0, 1)) / self.max_intensity)
        return new_clips

class VideoNormalize(Block):
    """Normalize images with mean and standard deviation.

    Given mean `(m1, ..., mn)` and std `(s1, ..., sn)` for `n` channels,
    this transform normalizes each channel of the input tensor with::

        output[i] = (input[i] - mi) / si

    If mean or std is scalar, the same value will be applied to all channels.

    Parameters
    ----------
    mean : float or tuple of floats
        The mean values.
    std : float or tuple of floats
        The standard deviation values.


    Inputs:
        - **data**: a list of frames with shape [C x H x W]

    Outputs:
        - **out**: a list of normalized frames with shape [C x H x W]
    """

    def __init__(self, mean, std):
        super(VideoNormalize, self).__init__()
        self.mean = np.asarray(mean).reshape((len(mean), 1, 1))
        self.std = np.asarray(std).reshape((len(std), 1, 1))

    def forward(self, clips):
        new_clips = []
        for cur_img in clips:
            new_clips.append((cur_img - self.mean) / self.std)
        return new_clips

class VideoRandomHorizontalFlip(Block):
    """Randomly flip the images left to right with a probability.

    Parameters
    ----------
    prob : float
        The probability value to flip the images.

    Inputs:
        - **data**: a list of frames with shape [H x W x C]

    Outputs:
        - **out**: a list of flipped frames with shape [H x W x C]
    """

    def __init__(self, prob=0.5):
        super(VideoRandomHorizontalFlip, self).__init__()
        self.prob = prob

    def forward(self, clips):
        new_clips = []
        if random.random() < self.prob:
            for cur_img in clips:
                new_clips.append(np.flip(cur_img, axis=1))
        else:
            new_clips = clips
        return new_clips

class VideoMultiScaleCrop(Block):
    """Corner cropping and multi-scale cropping.
    	Two data augmentation techniques introduced in:
        Towards Good Practices for Very Deep Two-Stream ConvNets,
        http://arxiv.org/abs/1507.02159
        Limin Wang, Yuanjun Xiong, Zhe Wang and Yu Qiao

    Parameters:
    ----------
    size : int
    	height and width required by network input, e.g., (224, 224)
    scale_ratios : list
    	efficient scale jittering, e.g., [1.0, 0.875, 0.75, 0.66]
    fix_crop : bool
    	use corner cropping or not. Default: True
    more_fix_crop : bool
    	use more corners or not. Default: True
    max_distort : float
    	maximum distortion. Default: 1

    Inputs:
    	- **data**: a list of frames with shape [H x W x C]

    Outputs:
        - **out**: a list of cropped frames with shape [size x size x C]

    """

    def __init__(self, size, scale_ratios, fix_crop=True,
                 more_fix_crop=True, max_distort=1):
        super(VideoMultiScaleCrop, self).__init__()
        from ...utils.filesystem import try_import_cv2
        self.cv2 = try_import_cv2()
        self.height = size[0]
        self.width = size[1]
        self.scale_ratios = scale_ratios
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.max_distort = max_distort

    def fillFixOffset(self, datum_height, datum_width):
        """Fixed cropping strategy

        Inputs:
            - **data**: height and width of input tensor

        Outputs:
            - **out**: a list of locations to crop the image

        """
        h_off = int((datum_height - self.height) / 4)
        w_off = int((datum_width - self.width) / 4)

        offsets = []
        offsets.append((0, 0))          # upper left
        offsets.append((0, 4*w_off))    # upper right
        offsets.append((4*h_off, 0))    # lower left
        offsets.append((4*h_off, 4*w_off))  # lower right
        offsets.append((2*h_off, 2*w_off))  # center

        if self.more_fix_crop:
            offsets.append((0, 2*w_off))        # top center
            offsets.append((4*h_off, 2*w_off))  # bottom center
            offsets.append((2*h_off, 0))        # left center
            offsets.append((2*h_off, 4*w_off))  # right center

            offsets.append((1*h_off, 1*w_off))  # upper left quarter
            offsets.append((1*h_off, 3*w_off))  # upper right quarter
            offsets.append((3*h_off, 1*w_off))  # lower left quarter
            offsets.append((3*h_off, 3*w_off))  # lower right quarter

        return offsets

    def fillCropSize(self, input_height, input_width):
        """Fixed cropping strategy

        Inputs:
            - **data**: height and width of input tensor

        Outputs:
            - **out**: a list of crop sizes to crop the image

        """
        crop_sizes = []
        base_size = np.min((input_height, input_width))
        scale_rates = self.scale_ratios
        for h, scale_rate_h in enumerate(scale_rates):
            crop_h = int(base_size * scale_rate_h)
            for w, scale_rate_w in enumerate(scale_rates):
                crop_w = int(base_size * scale_rate_w)
                if (np.absolute(h - w) <= self.max_distort):
                    crop_sizes.append((crop_h, crop_w))

        return crop_sizes

    def forward(self, clips):
        h, w, _ = clips[0].shape

        crop_size_pairs = self.fillCropSize(h, w)
        size_sel = random.randint(0, len(crop_size_pairs)-1)
        crop_height = crop_size_pairs[size_sel][0]
        crop_width = crop_size_pairs[size_sel][1]

        if self.fix_crop:
            offsets = self.fillFixOffset(h, w)
            off_sel = random.randint(0, len(offsets)-1)
            h_off = offsets[off_sel][0]
            w_off = offsets[off_sel][1]
        else:
            h_off = random.randint(0, h - self.height)
            w_off = random.randint(0, w - self.width)

        new_clips = []
        for cur_img in clips:
            crop_img = cur_img[h_off:h_off+crop_height, w_off:w_off+crop_width, :]
            new_clips.append(self.cv2.resize(crop_img, (self.width, self.height)))
        return new_clips


class VideoCenterCrop(Block):
    """Crop images at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)

    Parameters:
    ----------
    size : int
    	height and width required by network input, e.g., (224, 224)

    Inputs:
    	- **data**: a list of frames with shape [H x W x C]

    Outputs:
        - **out**: a list of cropped frames with shape [size x size x C]
    """

    def __init__(self, size):
        super(VideoCenterCrop, self).__init__()
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def forward(self, clips):
        h, w, _ = clips[0].shape
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))

        new_clips = []
        for cur_img in clips:
            new_clips.append(cur_img[y1:y1+th, x1:x1+tw, :])
        return new_clips

class VideoThreeCrop(Block):
    """This method crops 3 regions. All regions will be in shape
    :obj`size`. Depending on the situation, these regions may consist of:
        (1) 1 center, 1 top and 1 bottom
        (2) 1 center, 1 left and 1 right

    Parameters:
    ----------
    size : int
        height and width required by network input, e.g., (224, 224)

    Inputs:
        - **data**: a list of N frames with shape [H x W x C]

    Outputs:
        - **out**: a list of 3xN cropped frames with shape [size x size x C]
    """

    def __init__(self, size):
        super(VideoThreeCrop, self).__init__()
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def forward(self, clips):
        h, w, _ = clips[0].shape
        th, tw = self.size
        assert th == h or tw == w

        if th == h:
            w_step = (w - tw) // 2
            offsets = []
            offsets.append((0, 0))  # left
            offsets.append((2 * w_step, 0))  # right
            offsets.append((w_step, 0))  # middle
        elif tw == w:
            h_step = (h - th) // 2
            offsets = []
            offsets.append((0, 0))  # top
            offsets.append((0, 2 * h_step))  # down
            offsets.append((0, h_step))  # middle

        new_clips = []
        for ow, oh in offsets:
            for cur_img in clips:
                crop_img = cur_img[oh:oh+th, ow:ow+tw, :]
                new_clips.append(crop_img)
        return new_clips



class VideoTenCrop(Block):
    """Crop 10 regions from images.
    This is performed same as:
    http://chainercv.readthedocs.io/en/stable/reference/transforms.html#ten-crop

    This method crops 10 regions. All regions will be in shape
    :obj`size`. These regions consist of 1 center crop and 4 corner
    crops and horizontal flips of them.
    The crops are ordered in this order.
    * center crop
    * top-left crop
    * bottom-left crop
    * top-right crop
    * bottom-right crop
    * center crop (flipped horizontally)
    * top-left crop (flipped horizontally)
    * bottom-left crop (flipped horizontally)
    * top-right crop (flipped horizontally)
    * bottom-right crop (flipped horizontally)

    Parameters:
    ----------
    size : int
    	height and width required by network input, e.g., (224, 224)

    Inputs:
    	- **data**: a list of N frames with shape [H x W x C]

    Outputs:
        - **out**: a list of 10xN frames with shape [size x size x C]

    """
    def __init__(self, size):
        super(VideoTenCrop, self).__init__()
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def forward(self, clips):
        h, w, _ = clips[0].shape
        oh, ow = self.size
        if h < oh or w < ow:
            raise ValueError("Cannot crop area {} from image with size \
            	({}, {})".format(str(self.size), h, w))

        new_clips = []
        for cur_img in clips:
            center = cur_img[(h - oh) // 2:(h + oh) // 2, (w - ow) // 2:(w + ow) // 2, :]
            tl = cur_img[0:oh, 0:ow, :]
            bl = cur_img[h - oh:h, 0:ow, :]
            tr = cur_img[0:oh, w - ow:w, :]
            br = cur_img[h - oh:h, w - ow:w, :]
            new_clips.append(center)
            new_clips.append(tl)
            new_clips.append(bl)
            new_clips.append(tr)
            new_clips.append(br)
            new_clips.append(np.flip(center, axis=1))
            new_clips.append(np.flip(tl, axis=1))
            new_clips.append(np.flip(bl, axis=1))
            new_clips.append(np.flip(tr, axis=1))
            new_clips.append(np.flip(br, axis=1))
        return new_clips
