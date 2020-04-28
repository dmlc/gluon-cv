# pylint: disable=missing-docstring,arguments-differ,line-too-long
"""Extended image transformations to video transformations.
Code adapted from:
https://github.com/bryanyzhu/two-stream-pytorch
https://github.com/facebookresearch/SlowFast
"""
from __future__ import division
import random
import math
import numbers
import numpy as np
from mxnet.gluon import Block

__all__ = ['VideoToTensor', 'VideoNormalize', 'VideoRandomHorizontalFlip', 'VideoMultiScaleCrop',
           'VideoCenterCrop', 'VideoTenCrop', 'VideoGroupTrainTransform', 'VideoGroupValTransform',
           'VideoGroupTrainTransformV2', 'VideoGroupValTransformV2', 'ShortSideRescale',
           'VideoGroupTrainTransformV3', 'RandomResizedCrop', 'VideoGroupTrainTransformV4']


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
        self.multiScaleCrop = VideoMultiScaleCrop(size=size,
                                                  scale_ratios=scale_ratios,
                                                  fix_crop=fix_crop,
                                                  more_fix_crop=more_fix_crop,
                                                  max_distort=max_distort)
        self.scale_ratios = scale_ratios
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.max_distort = max_distort
        self.prob = prob
        self.max_intensity = max_intensity
        self.mean = np.asarray(mean).reshape((len(mean), 1, 1))
        self.std = np.asarray(std).reshape((len(std), 1, 1))


    def forward(self, clips):
        h, w, _ = clips[0].shape

        crop_size_pairs = self.multiScaleCrop.fillCropSize(self.scale_ratios, self.max_distort, h, w)
        size_sel = random.choice(crop_size_pairs)
        crop_height = size_sel[0]
        crop_width = size_sel[1]

        is_flip = random.random() < self.prob
        if self.fix_crop:
            offsets = self.multiScaleCrop.fillFixOffset(self.more_fix_crop, h, w, crop_height, crop_width)
            off_sel = random.choice(offsets)
            h_off = off_sel[0]
            w_off = off_sel[1]
        else:
            h_off = random.randint(0, h - crop_height)
            w_off = random.randint(0, w - crop_width)

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


class ShortSideRescale(Block):
    """Short side rescale, keeping aspect ratio.

    Parameters
    ----------
    short_side : int
        A number that at least one side of the image has to be equal to.

    Inputs:
        - **data**: a list of frames with shape [H x W x C]

    Outputs:
        - **out**: a list of rescaled frames with shape [short_side x (W*ratio) x C]
                   or [(H*ratio) x short_side x C]. Ratio is the input's original aspect ratio.
    """

    def __init__(self, short_side):
        super(ShortSideRescale, self).__init__()

        from ...utils.filesystem import try_import_cv2
        self.cv2 = try_import_cv2()
        self.short_side = short_side

    def forward(self, clips):
        h, w, _ = clips[0].shape

        new_w = self.short_side
        new_h = self.short_side
        if w < h:
            new_h = int(math.floor((float(h) / w) * self.short_side))
        else:
            new_w = int(math.floor((float(w) / h) * self.short_side))

        new_clips = []
        for cur_img in clips:
            new_clips.append(self.cv2.resize(cur_img, (new_w, new_h)))
        return new_clips


class RandomResizedCrop(Block):
    """Random crop and resize.
    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.
    This is popularly used to train the Inception networks.

    Parameters
    ----------
    size : tuple of int
        Expected output size of each edge
    scale : tuple of float
        Range of size of the origin size cropped
    ratio : tuple of float
        Range of aspect ratio of the origin aspect ratio cropped

    Inputs:
        - **data**: a list of frames with shape [H x W x C]

    Outputs:
        - **out**: a list of cropped and resized frames with shape [size x size x C]
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)):
        super(RandomResizedCrop, self).__init__()

        from ...utils.filesystem import try_import_cv2
        self.cv2 = try_import_cv2()
        self.height = size[0]
        self.width = size[1]
        self.scale = scale
        self.ratio = ratio

    def get_params(self, img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Parameters:
            img (cv2 Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        height, width, _ = img.shape
        area = height * width

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if 0 < w <= width and 0 < h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                return i, j, h, w

        # Fallback to center crop
        in_ratio = float(width) / float(height)
        if (in_ratio < min(ratio)):
            w = width
            h = int(round(w / min(ratio)))
        elif (in_ratio > max(ratio)):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height

        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def forward(self, clips):
        h_off, w_off, crop_height, crop_width = self.get_params(clips[0], self.scale, self.ratio)

        new_clips = []
        for cur_img in clips:
            crop_img = cur_img[h_off:h_off+crop_height, w_off:w_off+crop_width, :]
            new_clips.append(self.cv2.resize(crop_img, (self.width, self.height)))
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
    	maximum aspect ratio distortion, used together with scale_ratios. Default: 1

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

    def fillFixOffset(self, more_fix_crop, image_h, image_w, crop_h, crop_w):
        """Fixed cropping strategy. Only crop the 4 corners and the center.
        If more_fix_crop is turned on, more corners will be counted.

        Inputs:
            - **data**: height and width of input tensor

        Outputs:
            - **out**: a list of locations to crop the image

        """
        h_off = (image_h - crop_h) // 4
        w_off = (image_w - crop_w) // 4

        offsets = []
        offsets.append((0, 0))          # upper left
        offsets.append((0, 4*w_off))    # upper right
        offsets.append((4*h_off, 0))    # lower left
        offsets.append((4*h_off, 4*w_off))  # lower right
        offsets.append((2*h_off, 2*w_off))  # center

        if more_fix_crop:
            offsets.append((0, 2*w_off))        # top center
            offsets.append((4*h_off, 2*w_off))  # bottom center
            offsets.append((2*h_off, 0))        # left center
            offsets.append((2*h_off, 4*w_off))  # right center

            offsets.append((1*h_off, 1*w_off))  # upper left quarter
            offsets.append((1*h_off, 3*w_off))  # upper right quarter
            offsets.append((3*h_off, 1*w_off))  # lower left quarter
            offsets.append((3*h_off, 3*w_off))  # lower right quarter

        return offsets

    def fillCropSize(self, scale_ratios, max_distort, image_h, image_w):
        """Fixed cropping strategy. Select crop size from
        pre-defined list (computed by scale_ratios).

        Inputs:
            - **data**: height and width of input tensor

        Outputs:
            - **out**: a list of crop sizes to crop the image

        """
        crop_sizes = []
        base_size = np.min((image_h, image_w))
        for h_index, scale_rate_h in enumerate(scale_ratios):
            crop_h = int(base_size * scale_rate_h)
            for w_index, scale_rate_w in enumerate(scale_ratios):
                crop_w = int(base_size * scale_rate_w)
                if (np.absolute(h_index - w_index) <= max_distort):
                    # To control the aspect ratio distortion
                    crop_sizes.append((crop_h, crop_w))

        return crop_sizes

    def forward(self, clips):
        h, w, _ = clips[0].shape

        crop_size_pairs = self.fillCropSize(self.scale_ratios, self.max_distort, h, w)
        size_sel = random.choice(crop_size_pairs)
        crop_height = size_sel[0]
        crop_width = size_sel[1]

        if self.fix_crop:
            offsets = self.fillFixOffset(self.more_fix_crop, h, w, crop_height, crop_width)
            off_sel = random.choice(offsets)
            h_off = off_sel[0]
            w_off = off_sel[1]
        else:
            h_off = random.randint(0, h - crop_height)
            w_off = random.randint(0, w - crop_width)

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


class VideoGroupTrainTransformV2(Block):
    """Combination of transforms for training.
    Follow the style of https://github.com/open-mmlab/mmaction
        (1) short side keep aspect ratio resize
        (2) multiscale crop
        (3) scale
        (4) random horizontal flip
        (5) to tensor
        (6) normalize
    """
    def __init__(self, size, short_side, scale_ratios, mean, std, fix_crop=True,
                 more_fix_crop=True, max_distort=1, prob=0.5, max_intensity=255.0):
        super(VideoGroupTrainTransformV2, self).__init__()

        from ...utils.filesystem import try_import_cv2
        self.cv2 = try_import_cv2()
        self.height = size[0]
        self.width = size[1]
        self.short_side = short_side
        self.multiScaleCrop = VideoMultiScaleCrop(size=size,
                                                  scale_ratios=scale_ratios,
                                                  fix_crop=fix_crop,
                                                  more_fix_crop=more_fix_crop,
                                                  max_distort=max_distort)
        self.scale_ratios = scale_ratios
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.max_distort = max_distort
        self.prob = prob
        self.max_intensity = max_intensity
        self.mean = np.asarray(mean).reshape((len(mean), 1, 1))
        self.std = np.asarray(std).reshape((len(std), 1, 1))

    def forward(self, clips):
        h, w, _ = clips[0].shape

        # step 1: short side keep aspect ratio resize
        new_w = self.short_side
        new_h = self.short_side
        if w < h:
            new_h = int(math.floor((float(h) / w) * self.short_side))
        else:
            new_w = int(math.floor((float(w) / h) * self.short_side))

        # step 2: multiscale crop
        crop_size_pairs = self.multiScaleCrop.fillCropSize(self.scale_ratios, self.max_distort, new_h, new_w)
        size_sel = random.choice(crop_size_pairs)
        crop_height = size_sel[0]
        crop_width = size_sel[1]

        is_flip = random.random() < self.prob
        if self.fix_crop:
            offsets = self.multiScaleCrop.fillFixOffset(self.more_fix_crop, new_h, new_w, crop_height, crop_width)
            off_sel = random.choice(offsets)
            h_off = off_sel[0]
            w_off = off_sel[1]
        else:
            h_off = random.randint(0, new_h - crop_height)
            w_off = random.randint(0, new_w - crop_width)

        new_clips = []
        for cur_img in clips:
            scale_img = self.cv2.resize(cur_img, (new_w, new_h))
            crop_img = scale_img[h_off:h_off+crop_height, w_off:w_off+crop_width, :]
            resize_img = self.cv2.resize(crop_img, (self.width, self.height))
            if is_flip:
                flip_img = np.flip(resize_img, axis=1)
            else:
                flip_img = resize_img
            tensor_img = np.transpose(flip_img, axes=(2, 0, 1)) / self.max_intensity
            new_clips.append((tensor_img - self.mean) / self.std)
        return new_clips


class VideoGroupValTransformV2(Block):
    """Combination of transforms for validation.
    Follow the style of https://github.com/facebookresearch/SlowFast/
        (1) short side keep aspect ratio resize
        (2) center crop
        (3) to tensor
        (4) normalize
    """

    def __init__(self, crop_size, short_side, mean, std, max_intensity=255.0):
        super(VideoGroupValTransformV2, self).__init__()

        from ...utils.filesystem import try_import_cv2
        self.cv2 = try_import_cv2()
        self.height = crop_size[0]
        self.width = crop_size[1]
        self.short_side = short_side
        self.mean = np.asarray(mean).reshape((len(mean), 1, 1))
        self.std = np.asarray(std).reshape((len(std), 1, 1))
        self.max_intensity = max_intensity

    def forward(self, clips):
        h, w, _ = clips[0].shape

        # step 1: short side keep aspect ratio resize
        new_w = self.short_side
        new_h = self.short_side
        if w < h:
            new_h = int(math.floor((float(h) / w) * self.short_side))
        else:
            new_w = int(math.floor((float(w) / h) * self.short_side))

        # step 2: center crop
        h_off = int(math.ceil((new_h - self.height) / 2))
        w_off = int(math.ceil((new_w - self.width) / 2))

        new_clips = []
        for cur_img in clips:
            scale_img = self.cv2.resize(cur_img, (new_w, new_h))
            crop_img = scale_img[h_off:h_off+self.height, w_off:w_off+self.width, :]
            tensor_img = np.transpose(crop_img, axes=(2, 0, 1)) / self.max_intensity
            new_clips.append((tensor_img - self.mean) / self.std)
        return new_clips


class VideoGroupTrainTransformV3(Block):
    """Combination of transforms for training.
    Follow the style of https://github.com/facebookresearch/SlowFast/
        (1) random short side scale jittering
        (2) random crop
        (3) random horizontal flip
        (4) to tensor
        (5) normalize
    """
    def __init__(self, crop_size, min_size, max_size,
                 mean, std, prob=0.5, max_intensity=255.0):
        super(VideoGroupTrainTransformV3, self).__init__()

        from ...utils.filesystem import try_import_cv2
        self.cv2 = try_import_cv2()
        self.height = crop_size[0]
        self.width = crop_size[1]
        self.min_size = min_size
        self.max_size = max_size
        self.mean = np.asarray(mean).reshape((len(mean), 1, 1))
        self.std = np.asarray(std).reshape((len(std), 1, 1))
        self.prob = prob
        self.max_intensity = max_intensity

    def forward(self, clips):
        h, w, _ = clips[0].shape

        # step 1: random short side scale jittering
        size = int(round(np.random.uniform(self.min_size, self.max_size)))
        new_w = size
        new_h = size
        if w < h:
            new_h = int(math.floor((float(h) / w) * size))
        else:
            new_w = int(math.floor((float(w) / h) * size))

        # step 2: random crop
        h_off = 0
        if new_h > self.height:
            h_off = int(np.random.randint(0, new_h - self.height))
        w_off = 0
        if new_w > self.width:
            w_off = int(np.random.randint(0, new_w - self.width))

        # step 3: random horizontal flip
        is_flip = random.random() < self.prob

        new_clips = []
        for cur_img in clips:
            scale_img = self.cv2.resize(cur_img, (new_w, new_h))
            crop_img = scale_img[h_off:h_off+self.height, w_off:w_off+self.width, :]
            if is_flip:
                flip_img = np.flip(crop_img, axis=1)
            else:
                flip_img = crop_img
            tensor_img = np.transpose(flip_img, axes=(2, 0, 1)) / self.max_intensity
            new_clips.append((tensor_img - self.mean) / self.std)
        return new_clips


class VideoGroupTrainTransformV4(Block):
    """Combination of transforms for training.
    Follow the style of https://github.com/open-mmlab/mmaction.
    This is only for tranining SlowFast family networks in mmaction.
        (1) random crop and resize
        (2) random horizontal flip
        (3) to tensor
        (4) normalize
    """
    def __init__(self, size, mean, std, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.),
                 prob=0.5, max_intensity=255.0):
        super(VideoGroupTrainTransformV4, self).__init__()

        from ...utils.filesystem import try_import_cv2
        self.cv2 = try_import_cv2()
        self.height = size[0]
        self.width = size[1]
        self.randomResizedCrop = RandomResizedCrop(size=size,
                                                   scale=scale,
                                                   ratio=ratio)
        self.scale = scale
        self.ratio = ratio
        self.prob = prob
        self.max_intensity = max_intensity
        self.mean = np.asarray(mean).reshape((len(mean), 1, 1))
        self.std = np.asarray(std).reshape((len(std), 1, 1))

    def forward(self, clips):
        h_off, w_off, crop_height, crop_width = self.randomResizedCrop.get_params(clips[0], self.scale, self.ratio)
        is_flip = random.random() < self.prob

        new_clips = []
        for cur_img in clips:
            crop_img = cur_img[h_off:h_off+crop_height, w_off:w_off+crop_width, :]
            resize_img = self.cv2.resize(crop_img, (self.width, self.height))
            if is_flip:
                flip_img = np.flip(resize_img, axis=1)
            else:
                flip_img = resize_img
            tensor_img = np.transpose(flip_img, axes=(2, 0, 1)) / self.max_intensity
            new_clips.append((tensor_img - self.mean) / self.std)
        return new_clips
