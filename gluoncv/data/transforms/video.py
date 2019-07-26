# pylint: disable= missing-docstring,arguments-differ
"""Extended image transformations to video transformations."""
from __future__ import division
import random
import numbers
import numpy as np
from mxnet import nd
from mxnet.gluon import Block

__all__ = ['VideoToTensor', 'VideoNormalize', 'VideoRandomHorizontalFlip', 'VideoMultiScaleCrop',
           'VideoCenterCrop', 'VideoTenCrop']

class VideoToTensor(Block):
    """Converts a video clip NDArray to a tensor NDArray.

    Converts a video clip NDArray of shape (H x W x C) in the range
    [0, 255] to a float32 tensor NDArray of shape (C x H x W) in
    the range [0, 1).

    Parameters
    ----------
    max_intensity : float
        The maximum intensity value to be divided.

    Inputs:
        - **data**: input tensor with (H x W x C) shape and uint8 type.

    Outputs:
        - **out**: output tensor with (C x H x W) shape and float32 type.
    """
    def __init__(self, max_intensity=255.0):
        super(VideoToTensor, self).__init__()
        self.max_intensity = max_intensity

    def forward(self, clips):
        return nd.transpose(clips, axes=(2, 0, 1)) / self.max_intensity

class VideoNormalize(Block):
    """Normalize an tensor of shape (C x H x W) with mean and standard deviation.

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
        - **data**: input tensor with (C x H x W) shape.

    Outputs:
        - **out**: output tensor with the shape as `data`.
    """

    def __init__(self, mean, std):
        super(VideoNormalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, clips):
        c, _, _ = clips.shape
        num_images = int(c / 3)
        clip_mean = self.mean * num_images
        clip_std = self.std * num_images
        clip_mean = nd.array(np.asarray(clip_mean).reshape((c, 1, 1)))
        clip_std = nd.array(np.asarray(clip_std).reshape((c, 1, 1)))

        return (clips - clip_mean) / clip_std

class VideoRandomHorizontalFlip(Block):
    """Randomly flip the input video clip left to right with a probability of 0.5.

    Parameters
    ----------
    px : float
        The probability value to flip the input tensor.

    Inputs:
        - **data**: input tensor with (H x W x C) shape.

    Outputs:
        - **out**: output tensor with same shape as `data`.
    """

    def __init__(self, px=0):
        super(VideoRandomHorizontalFlip, self).__init__()
        self.px = px

    def forward(self, clips):
        if random.random() < 0.5:
            clips = nd.flip(clips, axis=1)
        return clips

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
    	- **data**: input tensor with (H x W x C) shape.

    Outputs:
        - **out**: output tensor with desired size as 'size'

    """

    def __init__(self, size, scale_ratios, fix_crop=True,
                 more_fix_crop=True, max_distort=1):
        super(VideoMultiScaleCrop, self).__init__()
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
                # append this cropping size into the list
                if (np.absolute(h-w) <= self.max_distort):
                    crop_sizes.append((crop_h, crop_w))

        return crop_sizes

    def forward(self, clips):

        from ...utils.filesystem import try_import_cv2
        cv2 = try_import_cv2()

        clips = clips.asnumpy()
        h, w, c = clips.shape
        is_color = False
        if c % 3 == 0:
            is_color = True

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

        scaled_clips = np.zeros((self.height, self.width, c))
        if is_color:
            num_imgs = int(c / 3)
            for frame_id in range(num_imgs):
                cur_img = clips[:, :, frame_id*3:frame_id*3+3]
                crop_img = cur_img[h_off:h_off+crop_height, w_off:w_off+crop_width, :]
                scaled_clips[:, :, frame_id*3:frame_id*3+3] = \
                	cv2.resize(crop_img, (self.width, self.height), cv2.INTER_LINEAR)
        else:
            num_imgs = int(c / 1)
            for frame_id in range(num_imgs):
                cur_img = clips[:, :, frame_id:frame_id+1]
                crop_img = cur_img[h_off:h_off+crop_height, w_off:w_off+crop_width, :]
                scaled_clips[:, :, frame_id:frame_id+1] = np.expand_dims(\
                	cv2.resize(crop_img, (self.width, self.height), cv2.INTER_LINEAR), axis=2)

        return nd.array(scaled_clips)


class VideoCenterCrop(Block):
    """Crops the given numpy array at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)

    Parameters:
    ----------
    size : int
    	height and width required by network input, e.g., (224, 224)

    Inputs:
    	- **data**: input tensor with (H x W x C) shape.

    Outputs:
        - **out**: output tensor with desired size as 'size'
    """

    def __init__(self, size):
        super(VideoCenterCrop, self).__init__()
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def forward(self, clips):
        h, w, c = clips.shape
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))

        is_color = False
        if c % 3 == 0:
            is_color = True

        scaled_clips = nd.zeros((th, tw, c))
        if is_color:
            num_imgs = int(c / 3)
            for frame_id in range(num_imgs):
                cur_img = clips[:, :, frame_id*3:frame_id*3+3]
                crop_img = cur_img[y1:y1+th, x1:x1+tw, :]
                assert(crop_img.shape == (th, tw, 3))
                scaled_clips[:, :, frame_id*3:frame_id*3+3] = crop_img
        else:
            num_imgs = int(c / 1)
            for frame_id in range(num_imgs):
                cur_img = clips[:, :, frame_id:frame_id+1]
                crop_img = cur_img[y1:y1+th, x1:x1+tw, :]
                assert(crop_img.shape == (th, tw, 1))
                scaled_clips[:, :, frame_id:frame_id+1] = crop_img
        return scaled_clips


class VideoTenCrop(Block):
    """Crop 10 regions from an array.
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
    	- **data**: input tensor with (H x W x C) shape.

    Outputs:
        - **out**: output tensor with (H x W x 10C) shape.

    """
    def __init__(self, size):
        super(VideoTenCrop, self).__init__()
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def forward(self, clips):
        h, w, _ = clips.shape
        oh, ow = self.size
        if h < oh or w < ow:
            raise ValueError("Cannot crop area {} from image with size \
            	({}, {})".format(str(self.size), h, w))

        center = clips[(h - oh) // 2:(h + oh) // 2, (w - ow) // 2:(w + ow) // 2, :]
        tl = clips[0:oh, 0:ow, :]
        bl = clips[h - oh:h, 0:ow, :]
        tr = clips[0:oh, w - ow:w, :]
        br = clips[h - oh:h, w - ow:w, :]
        crops = nd.concat(*[center, tl, bl, tr, br], dim=2)
        crops = nd.concat(*[crops, nd.flip(crops, axis=1)], dim=2)
        return crops
