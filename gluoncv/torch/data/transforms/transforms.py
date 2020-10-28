"""Code adapted from https://github.com/open-mmlab/mmaction
and https://github.com/bryanyzhu/two-stream-pytorch"""

import random
from PIL import Image, ImageOps
import numpy as np
import numbers
import math
import cv2
import collections
import PIL


class Transform:

    def __call__(self, data):
        raise NotImplemented


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


###################################################################################
#   Array Related Transformations
###################################################################################


class ToNumpy(Transform):
    def __call__(self, image):
        arr = np.array(image, dtype=np.float32)
        return arr


class StackImage(Transform):
    """
    This transformation stacks all images in nested lists into one list
    It works in a depth-first manner
    """

    def __init__(self, level=2):
        self.level = level

    def __call__(self, images):
        out_array = []

        def traverse(array, depth, max_depth):
            if depth != max_depth:
                for x in array:
                    traverse(x, depth + 1, max_depth)
            else:
                out_array.extend(array)

        traverse(images, 1, self.level)
        return out_array


class GroupApply(Transform):
    def __init__(self, transform):
        self.trans = transform

    def __call__(self, input_list):
        return [self.trans(x) for x in input_list]


class ToCNNInput(Transform):
    def __call__(self, array):

        if len(array.shape) == 3:
            return array.transpose((2, 0, 1))
        else:
            return array


class Normalize(Transform):
    def __init__(self, mean=128, std=128):
        self.mean = mean
        self.std = std

    def __call__(self, array):
        return (array - self.mean) / self.std


################################################################################################
# Basic Transformations
################################################################################################


class IdentityTransform(Transform):

    def __call__(self, data):
        return data


class RandomCrop(Transform):

    def __init__(self, size):

        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):

        w, h = (img[0].shape[0], img[0].shape[1])
        th, tw = self.size
        clip_len = 64

        if tw > w or th > h:
            raise ValueError("RandomCrop Failed: crop size is larger than image size")

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        buffer = np.array(img)

        buffer = buffer[:,
                 x1:x1 + th,
                 y1:y1 + tw, :]

        return buffer


class CenterCrop(Transform):

    def __init__(self, size):

        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):

        w, h = img.size
        th, tw = self.size

        if tw > w or th > h:
            raise ValueError("CenterCrop Failed: crop size is larger than image size")

        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))

        return img.crop((x1, y1, x1 + tw, y1 + th))


class Scale(Transform):
    """Rescale the input PIL.Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):

        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):

        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)


################################################################################################
# Group Transformations: apply transformation to sequence of images
################################################################################################


class GroupRandomCrop(Transform):

    def __init__(self, size):
        self.worker = RandomCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupCenterCrop(Transform):

    def __init__(self, size):
        self.worker = CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupRandomHorizontalFlip(Transform):
    """Randomly horizontally flips the given PIL.Image with a given probability
    """

    def __init__(self, is_flow=False, prob=0.5):
        self.is_flow = is_flow
        self.prob = prob

    def __call__(self, img_group, is_flow=False):
        v = random.random()
        if v < self.prob:

            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            if self.is_flow:
                for i in range(0, len(ret), 2):
                    ret[i] = ImageOps.invert(ret[i])  # invert flow pixel values when flipping
            return ret
        else:
            return img_group


class AlternateHorizontalFlip(Transform):
    """
    Input is a list of len the number of segments. Each element is a list of PIL images of length num. of segment frames (snippet length)
    All frames are flipped horizontally for every other segment
    """

    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, frames):

        def horizontal_flip_img_list(img_group, is_flow):
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            if is_flow:
                for i in range(0, len(ret), 2):
                    ret[i] = ImageOps.invert(ret[i])  # invert flow pixel values when flipping
            return ret

        n_segments = len(frames)
        for s_idx in range(n_segments):
            if s_idx % 2 == 1:
                frames[s_idx] = horizontal_flip_img_list(frames[s_idx], self.is_flow)
        return frames


class GroupNormalize(Transform):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        rep_mean = self.mean * (tensor.shape[0] // len(self.mean))
        rep_std = self.std * (tensor.shape[0] // len(self.std))

        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.__sub__(m).__div__(s)

        return tensor


class GroupScale(Transform):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = Scale(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupOverSample(Transform):

    def __init__(self, crop_size, scale_size=None):
        self.crop_size = crop_size if not isinstance(crop_size, int) else (crop_size, crop_size)

        if scale_size is not None:
            self.scale_worker = GroupScale(scale_size)
        else:
            self.scale_worker = None

    def __call__(self, img_group):

        if self.scale_worker is not None:
            img_group = self.scale_worker(img_group)

        image_w, image_h = img_group[0].size
        crop_w, crop_h = self.crop_size

        offsets = GroupMultiScaleCrop.fill_fix_offset(False, image_w, image_h, crop_w, crop_h)
        oversample_group = list()
        for o_w, o_h in offsets:
            normal_group = list()
            flip_group = list()
            for i, img in enumerate(img_group):
                crop = img.crop((o_w, o_h, o_w + crop_w, o_h + crop_h))
                normal_group.append(crop)
                flip_crop = crop.copy().transpose(Image.FLIP_LEFT_RIGHT)

                if img.mode == 'L' and i % 2 == 0:
                    flip_group.append(ImageOps.invert(flip_crop))
                else:
                    flip_group.append(flip_crop)
            oversample_group.extend(normal_group)
            oversample_group.extend(flip_group)
        return oversample_group


class GroupMultiScaleCrop(Transform):

    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, 875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = Image.BILINEAR

    def __call__(self, img_group):
        im_size = img_group[0].size

        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = [img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h)) for img in img_group]
        ret_img_group = [img.resize((self.input_size[0], self.input_size[1]), self.interpolation)
                         for img in crop_img_group]
        return ret_img_group

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret


class GroupRandomSizedCrop(Transform):
    """Random crop the given PIL.Image to a random size of (0.08 to 1.0) of the original size
    and and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio
    This is popularly used to train the Inception networks
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img_group):
        for attempt in range(10):
            area = img_group[0].size[0] * img_group[0].size[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img_group[0].size[0] and h <= img_group[0].size[1]:
                x1 = random.randint(0, img_group[0].size[0] - w)
                y1 = random.randint(0, img_group[0].size[1] - h)
                found = True
                break
        else:
            found = False
            x1 = 0
            y1 = 0

        if found:
            out_group = list()
            for img in img_group:
                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))
                out_group.append(img.resize((self.size, self.size), self.interpolation))
            return out_group
        else:
            # Fallback
            scale = GroupScale(self.size, interpolation=self.interpolation)
            crop = GroupRandomCrop(self.size)
            return crop(scale(img_group))


class Stack(Transform):

    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L':
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':
            if self.roll:
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                return np.concatenate(img_group, axis=2)


def parse_mean_std(mean, std, is_flow=False):
    if mean is None:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255
        if is_flow:
            mean = np.float32(128)

    if std is None:
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255
        if is_flow:
            std = std.mean()
    return mean, std


class RandomHorizontalFlip(object):
    """Horizontally flip the list of given images randomly
    with a probability 0.5
    """

    def __call__(self, clip):
        """
        Args:
        img (PIL.Image or numpy.ndarray): List of images to be cropped
        in format (h, w, c) in numpy.ndarray

        Returns:
        PIL.Image or numpy.ndarray: Randomly flipped clip
        """
        if random.random() < 0.5:
            if isinstance(clip[0], np.ndarray):
                return [np.fliplr(img) for img in clip]
            elif isinstance(clip[0], PIL.Image.Image):
                return [
                    img.transpose(PIL.Image.FLIP_LEFT_RIGHT) for img in clip
                ]
            else:
                raise TypeError('Expected numpy.ndarray or PIL.Image' +
                                ' but got list of {0}'.format(type(clip[0])))
        return clip


class Resize(object):
    """Resizes a list of (H x W x C) numpy.ndarray to the final size

    The larger the original image is, the more times it takes to
    interpolate

    Args:
    interpolation (str): Can be one of 'nearest', 'bilinear'
    defaults to nearest
    size (tuple): (widht, height)
    """

    def __init__(self, size, interpolation='nearest'):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, clip):
        resized = self.resize_clip(
            clip, self.size, interpolation=self.interpolation)
        return resized

    def resize_clip(self, clip, size, interpolation='bilinear'):
        if isinstance(clip[0], np.ndarray):
            if isinstance(size, numbers.Number):
                im_h, im_w, im_c = clip[0].shape
                # Min spatial dim already matches minimal size
                if (im_w <= im_h and im_w == size) or (im_h <= im_w and im_h == size):
                    return clip
                new_h, new_w = self.get_resize_sizes(im_h, im_w, size)
                size = (new_w, new_h)
            else:
                size = size[1], size[0]
            if interpolation == 'bilinear':
                np_inter = cv2.INTER_LINEAR
            else:
                np_inter = cv2.INTER_NEAREST
            scaled = [cv2.resize(img, size, interpolation=np_inter) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            if isinstance(size, numbers.Number):
                im_w, im_h = clip[0].size
                # Min spatial dim already matches minimal size
                if (im_w <= im_h and im_w == size) or (im_h <= im_w and im_h == size):
                    return clip
                new_h, new_w = self.get_resize_sizes(im_h, im_w, size)
                size = (new_w, new_h)
            else:
                size = size[1], size[0]
            if interpolation == 'bilinear':
                pil_inter = PIL.Image.NEAREST
            else:
                pil_inter = PIL.Image.BILINEAR
            scaled = [img.resize(size, pil_inter) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))
        return scaled

    def get_resize_sizes(self, im_h, im_w, size):
        if im_w < im_h:
            ow = size
            oh = int(size * im_h / im_w)
        else:
            oh = size
            ow = int(size * im_w / im_h)
        return oh, ow


def get_standard_tsn_training_transform(input_size, scales=(1, .875, .75, .66), is_flow=False, mean=None, std=None):
    """
    This function return a composed transform ready to be pluged into the TSN dataset for training
    :param std:
    :param mean:
    :param input_size: the expected networks input size
    :param scales: scales in scale jittering
    :param is_flow: whether to flip the image in the flow style
    :return:
    """

    mean, std = parse_mean_std(mean, std, is_flow)

    if is_flow:
        scales = (1, .875, .75)

    image_wise = Compose([
        ToNumpy(),
        Normalize(mean=mean, std=std),
        ToCNNInput()
    ])
    transform = Compose(
        [
            StackImage(2 if not is_flow else 3),
            Compose([GroupMultiScaleCrop(input_size, scales),
                     GroupRandomHorizontalFlip(is_flow=is_flow)]),
            GroupApply(image_wise),
        ]
    )

    return transform


def get_standard_tsn_validation_transform(input_size, is_flow=False, mean=None, std=None):
    """
    This function returns a composed transform ready to be pluged into the TSN dataset for validation
    :param std:
    :param mean:
    :param input_size: the expected networks input size
    :param is_flow: whether to flip the image in the flow style
    :return:
    """

    mean, std = parse_mean_std(mean, std, is_flow)

    if input_size != (224, 224):
        resize = (int(340 * input_size[0] / 224), int(256 * input_size[1] / 224))
    else:
        resize = None

    image_wise = Compose([
        ToNumpy(),
        Normalize(mean=mean, std=std),
        ToCNNInput()
    ])
    transform = Compose(
        [
            StackImage(2 if not is_flow else 3),
            GroupScale(resize) if resize is not None else IdentityTransform(),
            Compose([GroupCenterCrop(input_size)]),
            GroupApply(image_wise),
        ]
    )

    return transform


def get_standard_tsn_testing_transform(oversample, input_size, is_flow=False, mean=None, std=None):
    """
    This function returns a composed transform ready to be pluged into the TSN dataset for testing
    """

    mean, std = parse_mean_std(mean, std, is_flow)

    if input_size != (224, 224):
        resize = (int(340 * input_size[0] / 224), int(256 * input_size[1] / 224))
    else:
        resize = None

    image_wise = Compose([
        ToNumpy(),
        Normalize(mean=mean, std=std),
        ToCNNInput()
    ])

    if not oversample:
        transform = Compose(
            [
                StackImage(2 if not is_flow else 3),
                GroupScale(resize) if resize is not None else IdentityTransform(),
                Compose([GroupCenterCrop(input_size)]),
                GroupApply(image_wise),
            ]
        )
    else:
        transform = Compose(
            [
                StackImage(2 if not is_flow else 3),
                GroupScale(resize) if resize is not None else IdentityTransform(),
                Compose([GroupOverSample(input_size)]),
                GroupApply(image_wise),
            ]
        )
    return transform


def get_3D_testing_transform(alternate_flip=True, center_crop=True, input_size=(224, 224), is_flow=False, mean=None,
                             std=None):
    mean, std = parse_mean_std(mean, std, is_flow)

    image_wise = Compose([
        ToNumpy(),
        Normalize(mean=mean, std=std),
        ToCNNInput()
    ])

    transform = Compose(
        [
            AlternateHorizontalFlip(is_flow) if alternate_flip else IdentityTransform(),
            StackImage(2 if not is_flow else 3),
            GroupCenterCrop(input_size) if center_crop else IdentityTransform(),
            GroupApply(image_wise),
        ]
    )

    return transform


def resize_clip_bb(clip, size, anno, interpolation='bilinear'):
    '''
    Resize clip and associate bounding boxes
    '''
    if isinstance(clip[0], np.ndarray):
        if isinstance(size, numbers.Number):
            im_h, im_w, im_c = clip[0].shape
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                   and im_h == size):
                return clip, anno
            new_h, new_w, scale = get_resize_sizes_scale(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[1], size[0]
        if interpolation == 'bilinear':
            np_inter = cv2.INTER_LINEAR
        else:
            np_inter = cv2.INTER_NEAREST
        scaled = [
            cv2.resize(img, size, interpolation=np_inter) for img in clip
        ]
    elif isinstance(clip[0], PIL.Image.Image):
        if isinstance(size, numbers.Number):
            im_w, im_h = clip[0].size
            # Min spatial dim already matches minimal size
            if (im_w <= im_h and im_w == size) or (im_h <= im_w
                                                   and im_h == size):
                return clip, anno
            new_h, new_w, scale = get_resize_sizes_scale(im_h, im_w, size)
            size = (new_w, new_h)
        else:
            size = size[1], size[0]
        if interpolation == 'bilinear':
            pil_inter = PIL.Image.NEAREST
        else:
            pil_inter = PIL.Image.BILINEAR
        scaled = [img.resize(size, pil_inter) for img in clip]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))

    for i in range(anno.__len__()):
        anno[i] = anno[i] * scale

    return scaled, anno


def get_resize_sizes_scale(im_h, im_w, size):
    if im_w < im_h:
        ow = size
        oh = int(size * im_h / im_w)
        scale = size / im_w
    else:
        oh = size
        ow = int(size * im_w / im_h)
        scale = size / im_h
    return oh, ow, scale


def randomcrop_clip_bb(clip, anno, crop_size):
    '''
        crop clip and associate bounding boxes
    '''
    h, w = (crop_size, crop_size)
    if isinstance(clip[0], np.ndarray):
        im_h, im_w, im_c = clip[0].shape
    elif isinstance(clip[0], PIL.Image.Image):
        im_w, im_h = clip[0].size
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))
    if w > im_w or h > im_h:
        error_msg = (
            'Initial image size should be larger then '
            'cropped size but got cropped sizes : ({w}, {h}) while '
            'initial image is ({im_w}, {im_h})'.format(
                im_w=im_w, im_h=im_h, w=w, h=h))
        raise ValueError(error_msg)

    x1 = random.randint(0, im_w - w)
    y1 = random.randint(0, im_h - h)
    cropped = crop_clip_bb(clip, y1, x1, h, w)
    c_anno = anno
    for i in range(anno.__len__()):
        c_anno[i][0] = np.max((int(anno[i][0] - x1), 0))
        c_anno[i][1] = np.max((int(anno[i][1] - y1), 0))
        c_anno[i][2] = np.max((int(anno[i][2]), anno[i][0] + anno[i][2] - x1 - c_anno[i][0]))
        c_anno[i][3] = np.max((int(anno[i][3]), anno[i][1] + anno[i][3] - y1 - c_anno[i][1]))
    return cropped, c_anno


def crop_clip_bb(clip, min_h, min_w, h, w):
    if isinstance(clip[0], np.ndarray):
        cropped = [img[min_h:min_h + h, min_w:min_w + w, :] for img in clip]

    elif isinstance(clip[0], PIL.Image.Image):
        cropped = [
            img.crop((min_w, min_h, min_w + w, min_h + h)) for img in clip
        ]
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))
    return cropped


def random_horizontal_flip_bb(clip, anno):
    '''
        crop clip and associate bounding boxes
    '''
    import random
    prob = random.uniform(0, 1)
    if prob < 0.5:
        return clip, anno

    if isinstance(clip[0], np.ndarray):
        im_h, im_w, im_c = clip[0].shape
    elif isinstance(clip[0], PIL.Image.Image):
        im_w, im_h = clip[0].size
    else:
        raise TypeError('Expected numpy.ndarray or PIL.Image' +
                        'but got list of {0}'.format(type(clip[0])))
    mid = im_w // 2
    for i in range(len(clip)):
        clip[i] = clip[i][:, ::-1, :]
    for i in range(anno.__len__()):
        anno[i][0] = np.max((int(mid - (anno[i][0] + anno[i][2] - mid)), 0))
        anno[i][2] = np.min((im_w - anno[i][0], anno[i][2]))
    return clip, anno


def normalize_pytorch(buffer):
    # Normalize the buffer
    buffer = buffer / 255.0
    buffer[:, :, :, 0] = (buffer[:, :, :, 0] - 0.45) / 0.225
    buffer[:, :, :, 1] = (buffer[:, :, :, 1] - 0.45) / 0.225
    buffer[:, :, :, 2] = (buffer[:, :, :, 2] - 0.45) / 0.225

    return buffer


def to_tensor_pytorch(buffer):
    # convert from [D, H, W, C] format to [C, D, H, W] (what PyTorch uses)
    # D = Depth (in this case, time), H = Height, W = Width, C = Channels
    return buffer.transpose((3, 0, 1, 2))
