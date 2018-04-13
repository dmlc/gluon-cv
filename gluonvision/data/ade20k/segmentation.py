"""Pascal ADE20K Semantic Segmentation Dataset."""
import os
import sys
import numpy as np
import random
import math
from PIL import Image, ImageOps, ImageFilter

from mxnet.gluon.data import dataset

class ADE20KSegmentation(dataset.Dataset):
    """ADE20K Semantic Segmentation Dataset.

    Parameters
    ----------
    root : string
        Path to VOCdevkit folder. Default is '$(HOME)/mxnet/datasplits/voc'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    target_transform : callable, optional
        A function that transforms the labels
    """
    BASE_DIR = 'ADEChallengeData2016'
    def __init__(self, root=os.path.expanduser('~/.mxnet/datasets/ade'),
                 split='train', transform=None, target_transform=None):
        self.root = os.path.join(root, self.BASE_DIR)
        self.transform = transform
        self.target_transform = target_transform
        self.mode = split
        self.images, self.masks = _get_ade20k_pairs(self.root, split)
        assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: \
                " + self.root + "\n"))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])

        mask = Image.open(self.masks[index])#.convert("P")
        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            raise RuntimeError('unknown mode for dataloader: {}'.format(self.mode))
        

        # general resize, normalize and toTensor
        if self.transform is not None:
            #print("transform for input")
            img = self.transform(img)
        if self.target_transform is not None:
            #print("transform for label")
            mask = self.target_transform(mask)

        return img, mask

    def __len__(self):
        return len(self.images)

    def _val_sync_transform(self, img, mask):
        outsize = 480
        short = outsize
        w, h = img.size
        if w > h:
            oh = short
            ow = int(1.0 * w * oh / h)
        else:
            ow = short
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        img = img.crop((x1, y1, x1+outsize, y1+outsize))
        mask = mask.crop((x1, y1, x1+outsize, y1+outsize))

        return img, mask

    def _sync_transform(self, img, mask):
        # random mirror
        if random.random() < 0.5:
            img  = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        base_size = 520
        crop_size = 480
        # random scale (short edge from 480 to 720)
        long_size = random.randint(int(base_size*0.5), int(base_size*2.0))
        w, h = img.size
        if h > w:
            oh = long_size
            ow = int(1.0 * w * oh / h)
            short_size = ow
        else:
            ow = long_size
            oh = int(1.0 * h * ow / w)
            short_size = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # random rotate -10~10, mask using NN rotate
        deg = random.uniform(-10,10)
        img = img.rotate(deg, resample=Image.BILINEAR)
        mask = mask.rotate(deg, resample=Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img  = ImageOps.expand(img,  border=(0,0,padw,padh), fill=0)
            mask = ImageOps.expand(mask, border=(0,0,padw,padh), fill=0)
        # random crop 480
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size) 
        img = img.crop((x1, y1, x1+crop_size, y1+crop_size))
        mask = mask.crop((x1, y1, x1+crop_size, y1+crop_size))
        # gaussian blur as in PSP ?
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        return img, mask

def _get_ade20k_pairs(folder, mode='train'):
    img_paths = []  
    mask_paths = []  
    if mode=='train':
        img_folder = os.path.join(folder, 'images/training')
        mask_folder = os.path.join(folder, 'annotations/training')
    else:
        img_folder = os.path.join(folder, 'images/validation')
        mask_folder = os.path.join(folder, 'annotations/validation')
    for filename in os.listdir(img_folder):
        basename, extension =os.path.splitext(filename)
        if filename.endswith(".jpg"):
            imgpath = os.path.join(img_folder, filename)
            maskname = basename + '.png'
            maskpath = os.path.join(mask_folder, maskname)
            if os.path.isfile(maskpath):
                img_paths.append(imgpath)
                mask_paths.append(maskpath)
            else:
                print('cannot find the mask:', maskpath)

    return img_paths, mask_paths

# acronym for easy load
_Segmentation = ADE20KSegmentation
