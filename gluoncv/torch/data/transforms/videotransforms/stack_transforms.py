import numpy as np
import PIL

import torch

from .utils import images as imageutils


class ToStackedTensor(object):
    """Converts a list of m (H x W x C) numpy.ndarrays in the range [0, 255]
    or PIL Images to a torch.FloatTensor of shape (m*C x H x W)
    in the range [0, 1.0]
    """

    def __init__(self, channel_nb=3):
        self.channel_nb = channel_nb

    def __call__(self, clip):
        """
        Args:
            clip (list of numpy.ndarray or PIL.Image.Image): clip
            (list of images) to be converted to tensor.
        """
        # Retrieve shape
        if isinstance(clip[0], np.ndarray):
            h, w, ch = clip[0].shape
            assert ch == self.channel_nb, 'got {} channels instead of 3'.format(
                ch)
        elif isinstance(clip[0], PIL.Image.Image):
            w, h = clip[0].size
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image\
            but got list of {0}'.format(type(clip[0])))

        np_clip = np.zeros([self.channel_nb * len(clip), int(h), int(w)])

        # Convert
        for img_idx, img in enumerate(clip):
            if isinstance(img, np.ndarray):
                pass
            elif isinstance(img, PIL.Image.Image):
                img = np.array(img, copy=False)
            else:
                raise TypeError('Expected numpy.ndarray or PIL.Image\
                but got list of {0}'.format(type(clip[0])))
            img = imageutils.convert_img(img)
            np_clip[img_idx * self.channel_nb:(
                img_idx + 1) * self.channel_nb, :, :] = img
        tensor_clip = torch.from_numpy(np_clip)
        return tensor_clip.float().div(255)
