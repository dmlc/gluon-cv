"""Extended image transformations to `mxnet.image`."""
from __future__ import division
import cv2
import random
import numbers
import numpy as np
import mxnet as mx
from mxnet import nd, image
from mxnet.base import numeric_types
from mxnet.gluon import Block, HybridBlock
from mxnet.gluon.nn import Sequential, HybridSequential

__all__ = ['VideoToTensor', 'VideoNormalize', 'VideoRandomHorizontalFlip', 'VideoMultiScaleCrop',
           'VideoCenterCrop', 'VideoTenCrop']

class Compose(Sequential):
    """Sequentially composes multiple transforms.

    Parameters
    ----------
    transforms : list of transform Blocks.
        The list of transforms to be composed.


    Inputs:
        - **data**: input tensor with shape of the first transform Block requires.

    Outputs:
        - **out**: output tensor with shape of the last transform Block produces.

    Examples
    --------
    >>> transformer = transforms.Compose([transforms.Resize(300),
    ...                                   transforms.CenterCrop(256),
    ...                                   transforms.ToTensor()])
    >>> image = mx.nd.random.uniform(0, 255, (224, 224, 3)).astype(dtype=np.uint8)
    >>> transformer(image)
    
    """
    def __init__(self, transforms):
        super(Compose, self).__init__()
        transforms.append(None)
        hybrid = []
        for i in transforms:
            if isinstance(i, HybridBlock):
                hybrid.append(i)
                continue
            elif len(hybrid) == 1:
                self.add(hybrid[0])
                hybrid = []
            elif len(hybrid) > 1:
                hblock = HybridSequential()
                for j in hybrid:
                    hblock.add(j)
                hblock.hybridize()
                self.add(hblock)
                hybrid = []

            if i is not None:
                self.add(i)

class VideoToTensor(Block):

    def __init__(self, max_intensity=255.0):
        super(VideoToTensor, self).__init__()
        self.max_intensity = max_intensity

    def forward(self, clips):
        return nd.transpose(clips, axes=(2,0,1)) / self.max_intensity

class VideoNormalize(Block):

    def __init__(self, mean, std):
        super(VideoNormalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, clips):
        clips = clips.asnumpy()
        c, _, _ = clips.shape
        num_images = int(c / 3)
        clip_mean = self.mean * num_images
        clip_std = self.std * num_images
        clip_mean = np.asarray(clip_mean).reshape((c, 1, 1))
        clip_std = np.asarray(clip_std).reshape((c, 1, 1))

        return nd.array((clips - clip_mean) / clip_std)

class VideoRandomHorizontalFlip(Block):

    def __init__(self, px=0):
        super(VideoRandomHorizontalFlip, self).__init__()
        self.px = px

    def forward(self, clips):
        if random.random() < 0.5:
            clips = nd.flip(clips, axis=1)
        return clips

class VideoMultiScaleCrop(Block):
    """
    Description: Corner cropping and multi-scale cropping. Two data augmentation techniques introduced in:
        Towards Good Practices for Very Deep Two-Stream ConvNets,
        http://arxiv.org/abs/1507.02159
        Limin Wang, Yuanjun Xiong, Zhe Wang and Yu Qiao
    Parameters:
        size: height and width required by network input, e.g., (224, 224)
        scale_ratios: efficient scale jittering, e.g., [1.0, 0.875, 0.75, 0.66]
        fix_crop: use corner cropping or not. Default: True
        more_fix_crop: use more corners or not. Default: True
        max_distort: maximum distortion. Default: 1
        interpolation: Default: cv2.INTER_LINEAR
    """

    def __init__(self, size, scale_ratios, fix_crop=True, more_fix_crop=True, max_distort=1, interpolation=cv2.INTER_LINEAR):
        super(VideoMultiScaleCrop, self).__init__()
        self.height = size[0]
        self.width = size[1]
        self.scale_ratios = scale_ratios
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.max_distort = max_distort
        self.interpolation = interpolation

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
        for h in range(len(scale_rates)):
            crop_h = int(base_size * scale_rates[h])
            for w in range(len(scale_rates)):
                crop_w = int(base_size * scale_rates[w])
                # append this cropping size into the list
                if (np.absolute(h-w) <= self.max_distort):
                    crop_sizes.append((crop_h, crop_w))

        return crop_sizes

    def forward(self, clips):
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

        scaled_clips = np.zeros((self.height,self.width,c))
        if is_color:
            num_imgs = int(c / 3)
            for frame_id in range(num_imgs):
                cur_img = clips[:,:,frame_id*3:frame_id*3+3]
                crop_img = cur_img[h_off:h_off+crop_height, w_off:w_off+crop_width, :]
                scaled_clips[:,:,frame_id*3:frame_id*3+3] = cv2.resize(crop_img, (self.width, self.height), self.interpolation)
            return nd.array(scaled_clips)
        else:
            num_imgs = int(c / 1)
            for frame_id in range(num_imgs):
                cur_img = clips[:,:,frame_id:frame_id+1]
                crop_img = cur_img[h_off:h_off+crop_height, w_off:w_off+crop_width, :]
                scaled_clips[:,:,frame_id:frame_id+1] = np.expand_dims(cv2.resize(crop_img, (self.width, self.height), self.interpolation), axis=2)
            return nd.array(scaled_clips)


class VideoCenterCrop(Block):
    """Crops the given numpy array at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, clips):
        clips = clips.asnumpy()
        h, w, c = clips.shape
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))

        is_color = False
        if c % 3 == 0:
            is_color = True

        if is_color:
            num_imgs = int(c / 3)
            scaled_clips = np.zeros((th,tw,c))
            for frame_id in range(num_imgs):
                cur_img = clips[:,:,frame_id*3:frame_id*3+3]
                crop_img = cur_img[y1:y1+th, x1:x1+tw, :]
                assert(crop_img.shape == (th, tw, 3))
                scaled_clips[:,:,frame_id*3:frame_id*3+3] = crop_img
            return nd.array(scaled_clips)
        else:
            num_imgs = int(c / 1)
            scaled_clips = np.zeros((th,tw,c))
            for frame_id in range(num_imgs):
                cur_img = clips[:,:,frame_id:frame_id+1]
                crop_img = cur_img[y1:y1+th, x1:x1+tw, :]
                assert(crop_img.shape == (th, tw, 1))
                scaled_clips[:,:,frame_id:frame_id+1] = crop_img
            return nd.array(scaled_clips)


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

    Parameters
    ----------
    src : mxnet.nd.NDArray
        Input image sequences
    size : int or tuple
        Tuple of length 2, as (width, height) of the cropped areas.

    Returns
    -------
    mxnet.nd.NDArray
        The cropped images with shape (size[1], size[0], C x 10)

    """
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, clips):
        h, w, c = clips.shape
        oh, ow = self.size
        if h < oh or w < ow:
            raise ValueError("Cannot crop area {} from image with size ({}, {})".format(str(self.size), h, w))

        center = clips[(h - oh) // 2:(h + oh) // 2, (w - ow) // 2:(w + ow) // 2, :]
        tl = clips[0:oh, 0:ow, :]
        bl = clips[h - oh:h, 0:ow, :]
        tr = clips[0:oh, w - ow:w, :]
        br = clips[h - oh:h, w - ow:w, :]
        # crops = nd.stack(*[center, tl, bl, tr, br], axis=0)
        crops = nd.concat(*[center, tl, bl, tr, br], dim=2)
        crops = nd.concat(*[crops, nd.flip(crops, axis=1)], dim=2)
        return crops

