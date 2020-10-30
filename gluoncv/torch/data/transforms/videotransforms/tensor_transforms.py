import random

from .utils import functional as F


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation

    Given mean: m and std: s
    will  normalize each channel as channel = (channel - mean) / std

    Args:
        mean (int): mean value
        std (int): std value
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of stacked images or image
            of size (C, H, W) to be normalized

        Returns:
            Tensor: Normalized stack of image of image
        """
        return F.normalize(tensor, self.mean, self.std)


class SpatialRandomCrop(object):
    """Crops a random spatial crop in a spatio-temporal
    numpy or tensor input [Channel, Time, Height, Width]
    """

    def __init__(self, size):
        """
        Args:
            size (tuple): in format (height, width)
        """
        self.size = size

    def __call__(self, tensor):
        h, w = self.size
        _, _, tensor_h, tensor_w = tensor.shape

        if w > tensor_w or h > tensor_h:
            error_msg = (
                'Initial tensor spatial size should be larger then '
                'cropped size but got cropped sizes : ({w}, {h}) while '
                'initial tensor is ({t_w}, {t_h})'.format(
                    t_w=tensor_w, t_h=tensor_h, w=w, h=h))
            raise ValueError(error_msg)
        x1 = random.randint(0, tensor_w - w)
        y1 = random.randint(0, tensor_h - h)
        cropped = tensor[:, :, y1:y1 + h, x1:x1 + h]
        return cropped
