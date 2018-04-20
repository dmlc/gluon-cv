"""Base dataset methods."""
import os
import random
import numpy as np
from mxnet import cpu
from mxnet.gluon.data import dataset
import mxnet.ndarray as F
from PIL import Image, ImageOps, ImageFilter


class ClassProperty(object):
    """Readonly @ClassProperty descriptor for internal usage."""
    def __init__(self, fget):
        self.fget = fget

    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)


class VisionDataset(dataset.Dataset):
    """Base Dataset with directory checker.

    Parameters
    ----------
    root : str
        The root path of xxx.names, by defaut is '~/.mxnet/datasets/foo', where
        `foo` is the name of the dataset.
    """
    def __init__(self, root):
        if not os.path.isdir(os.path.expanduser(root)):
            helper_msg = "{} is not a valid dir. Did you forget to initalize \
                         datasets described in `gluon-vision/scripts/datasets`? You need \
                         to initialize each dataset only once.".format(root)
            raise OSError(helper_msg)

    @property
    def classes(self):
        raise NotImplementedError

    @property
    def num_class(self):
        """Number of categories."""
        return len(self.classes)


class SegmentationDataset(VisionDataset):
    """Segmentation Base Dataset"""
    def __init__(self, root, base_size=520, crop_size=480):
        super(SegmentationDataset, self).__init__(root)
        self.base_size = base_size
        self.crop_size = crop_size

    def _val_sync_transform(self, img, mask):
        outsize = self.crop_size
        short_size = outsize
        w, h = img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1+outsize, y1+outsize))
        mask = mask.crop((x1, y1, x1+outsize, y1+outsize))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge from 480 to 720)
        short_size = random.randint(int(self.base_size*0.5), int(self.base_size*2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
            long_size = oh
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
            long_size = ow
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # random rotate -10~10, mask using NN rotate
        deg = random.uniform(-10, 10)
        img = img.rotate(deg, resample=Image.BILINEAR)
        mask = mask.rotate(deg, resample=Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1+crop_size, y1+crop_size))
        mask = mask.crop((x1, y1, x1+crop_size, y1+crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _img_transform(self, img):
        return F.array(np.array(img), cpu(0))

    def _mask_transform(self, mask):
        return F.array(mask, cpu(0)).astype('int32')
