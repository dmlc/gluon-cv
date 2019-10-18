"""Base segmentation dataset"""
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import mxnet as mx
from mxnet import cpu
import mxnet.ndarray as F
from .base import VisionDataset

__all__ = ['ms_batchify_fn', 'SegmentationDataset']

class SegmentationDataset(VisionDataset):
    """Segmentation Base Dataset"""
    # pylint: disable=abstract-method
    def __init__(self, root, split, mode, transform, base_size=520, crop_size=480):
        super(SegmentationDataset, self).__init__(root)
        self.root = root
        self.transform = transform
        self.split = split
        self.mode = mode if mode is not None else split
        self.base_size = base_size
        self.crop_size = crop_size

    def _val_sync_transform(self, img, mask):
        w, h = img.size
        ow, oh, outsize_w, outsize_h = self._resize_assist(w, h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize_w) / 2.))
        y1 = int(round((h - outsize_h) / 2.))
        img = img.crop((x1, y1, x1 + outsize_w, y1 + outsize_h))
        mask = mask.crop((x1, y1, x1 + outsize_w, y1 + outsize_h))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        # random scale
        long_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        ow, oh, outsize_w, outsize_h = self._scale_assist(w, h, long_size)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        padh = outsize_h - oh if oh < outsize_h else 0
        padw = outsize_w - ow if ow < outsize_w else 0
        img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
        mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - outsize_w)
        y1 = random.randint(0, h - outsize_h)
        img = img.crop((x1, y1, x1 + outsize_w, y1 + outsize_h))
        mask = mask.crop((x1, y1, x1 + outsize_w, y1 + outsize_h))
        # gaussian blur
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        # final transform
        img, mask = self._img_transform(img), self._mask_transform(mask)
        return img, mask
    
    def _scale_assist(self, w, h, long_size):
        if h > w:
            oh = long_size
            ow = int(1.0 * w * long_size / h + 0.5)
        else:
            ow = long_size
            oh = int(1.0 * h * long_size / w + 0.5)
        # different outsize
        if isinstance(self.crop_size, int):
            outsize_w = self.crop_size
            outsize_h = self.crop_size
        elif isinstance(self.crop_size, (list, tuple)):
            assert len(self.crop_size) == 2
            outsize_h, outsize_w = self.crop_size
        else:
            raise RuntimeError("Unknown crop size: {}".format(self.crop_size))
        return ow, oh, outsize_w, outsize_h
    
    def _resize_assist(self, w, h):
        if isinstance(self.crop_size, int):
            if w > h:
                oh = self.crop_size
                ow = int(1.0 * w * oh / h)
            else:
                ow = self.crop_size
                oh = int(1.0 * h * ow / w)
            outsize_w = self.crop_size
            outsize_h = self.crop_size
            return ow, oh, outsize_w, outsize_h
        elif isinstance(self.crop_size, (list, tuple)):
            assert len(self.crop_size) == 2
            outsize_h, outsize_w = self.crop_size
            factor_h = outsize_h / h * 1.0
            factor_w = outsize_w / w * 1.0
            if factor_h > factor_w:
                oh = outsize_h
                ow = int(1.0 * w * oh / h)
            else:
                ow = outsize_w
                oh = int(1.0 * h * ow / w)
            return ow, oh, outsize_w, outsize_h
        else:
            raise RuntimeError("Unknown crop size: {}".format(self.crop_size))

    def _img_transform(self, img):
        return F.array(np.array(img), cpu(0))

    def _mask_transform(self, mask):
        return F.array(np.array(mask), cpu(0)).astype('int32')

    @property
    def num_class(self):
        """Number of categories."""
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        return 0

def ms_batchify_fn(data):
    """Multi-size batchify function"""
    if isinstance(data[0], (str, mx.nd.NDArray)):
        return list(data)
    elif isinstance(data[0], tuple):
        data = zip(*data)
        return [ms_batchify_fn(i) for i in data]
    raise RuntimeError('unknown datatype')
