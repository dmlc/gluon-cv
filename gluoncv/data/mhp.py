"""Multi-Human-Parsing V1 Dataset."""
import os
import numpy as np
from PIL import Image
from PIL import ImageFile

import mxnet as mx
from .segbase import SegmentationDataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


class MHPV1Segmentation(SegmentationDataset):
    """Multi-Human-Parsing V1 Dataset.
    Parameters
    ----------
    root : string
        Path to MHPV1 folder. Default is '$(HOME)/.mxnet/datasets/mhp/LV-MHP-v1'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    Examples
    --------
    >>> from mxnet.gluon.data.vision import transforms
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    >>> ])
    >>> # Create Dataset
    >>> trainset = gluoncv.data.MHPV1Segmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = gluon.data.DataLoader(
    >>>     trainset, 4, shuffle=True, last_batch='rollover',
    >>>     num_workers=4)
    """
    # pylint: disable=abstract-method
    NUM_CLASS = 19
    CLASSES = ("background", "hat", "hair", "sunglasses", "upper clothes", "skirt",
               "pants", "dress", "belt", "left shoe", "right shoe", "face", "left leg",
               "right leg", "left arm", "right arm", "bag", "scarf", "torso skin")

    def __init__(self, root=os.path.expanduser('~/.mxnet/datasets/mhp/LV-MHP-v1'),
                 split='train', mode=None, transform=None, base_size=768, **kwargs):
        super(MHPV1Segmentation, self).__init__(root, split, mode, transform, base_size, **kwargs)
        assert os.path.exists(root), "Please setup the dataset using" + "scripts/datasets/mhp_v1.py"
        self.images, self.masks = _get_mhp_pairs(root, split)
        assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: \
                " + root + "\n"))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')

        # nan check
        img_np = np.array(img, dtype=np.uint8)
        assert not np.isnan(np.sum(img_np))

        if self.mode == 'test':
            img = self._img_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])

        mask = _get_mask(self.masks[index])

        # Here, we resize input image resolution to the multiples of 8
        # for avoiding resolution misalignment during downsampling and upsampling
        w, h = img.size
        if h < w:
            oh = self.base_size
            ow = int(1.0 * w * oh / h + 0.5)
            if ow % 8:
                ow = int(round(ow / 8) * 8)
        else:
            ow = self.base_size
            oh = int(1.0 * h * ow / w + 0.5)
            if oh % 8:
                oh = int(round(oh / 8) * 8)

        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)

        return img, mask

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        return mx.nd.array(target, mx.cpu(0))

    def __len__(self):
        return len(self.images)

    @property
    def classes(self):
        """Category names."""
        return type(self).CLASSES

    @property
    def pred_offset(self):
        return 0


def _get_mhp_pairs(folder, split='train'):
    img_paths = []
    mask_paths = []
    img_folder = os.path.join(folder, 'images')
    mask_folder = os.path.join(folder, 'annotations')

    if split == 'test':
        img_list = os.path.join(folder, 'test_list.txt')
    else:
        img_list = os.path.join(folder, 'train_list.txt')

    with open(img_list) as txt:
        for filename in txt:
            # record mask paths
            mask_short_path = []
            basename, _ = os.path.splitext(filename)
            for maskname in os.listdir(mask_folder):
                if maskname.startswith(basename):
                    maskpath = os.path.join(mask_folder, maskname)
                    if os.path.isfile(maskpath):
                        mask_short_path.append(maskpath)
                    else:
                        print('cannot find the mask:', maskpath)

            # mask_short_path is not empty
            if mask_short_path:
                mask_paths.append(mask_short_path)

            # record img paths
            imgpath = os.path.join(img_folder, filename.rstrip('\n'))
            if os.path.isfile(imgpath):
                img_paths.append(imgpath)
            else:
                print('cannot find the image:', imgpath)

    if split == 'train':
        img_paths = img_paths[:3000]
        mask_paths = mask_paths[:3000]
    elif split == 'val':
        img_paths = img_paths[3001:4000]
        mask_paths = mask_paths[3001:4000]

    return img_paths, mask_paths


def _get_mask(mask_paths):
    mask_np = None
    mask_idx = None
    for _, mask_path in enumerate(mask_paths):
        mask_sub = Image.open(mask_path)
        mask_sub_np = np.array(mask_sub, dtype=np.uint8)
        if mask_idx is None:
            mask_idx = np.zeros(mask_sub_np.shape, dtype=np.uint8)
        mask_sub_np = np.ma.masked_array(mask_sub_np, mask=mask_idx)
        mask_idx += np.minimum(mask_sub_np, 1)

        if mask_np is None:
            mask_np = mask_sub_np
        else:
            mask_np += mask_sub_np

    # nan check
    assert not np.isnan(np.sum(mask_np))

    # categories check
    assert (np.max(mask_np) <= 18 and np.min(mask_np) >= 0)

    mask = Image.fromarray(mask_np)

    return mask
