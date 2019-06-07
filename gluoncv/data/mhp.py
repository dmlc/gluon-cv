"""Multi-Human-Parsing Dataset."""
import os
from PIL import Image
import numpy as np
import mxnet as mx
from .segbase import SegmentationDataset

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class MHPSegmentation(SegmentationDataset):
    """Multi-Human-Parsing Dataset.

    Parameters
    ----------
    root : string
        Path to VOCdevkit folder. Default is '$(HOME)/mxnet/datasplits/mhp'
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
    >>> trainset = gluoncv.data.MHPSegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = gluon.data.DataLoader(
    >>>     trainset, 4, shuffle=True, last_batch='rollover',
    >>>     num_workers=4)
    """
    # pylint: disable=abstract-method
    NUM_CLASS = 18
    def __init__(self, root=os.path.expanduser('~/.mxnet/datasets/mhp'),
                 split='train', mode=None, transform=None, **kwargs):
        super(MHPSegmentation, self).__init__(root, split, mode, transform, **kwargs)
        assert os.path.exists(root), "Please setup the dataset using" + \
            "scripts/datasets/mhp.py"
        self.images, self.masks = _get_mhp_pairs(root, split)
        assert (len(self.images) == len(self.masks))
        if len(self.images) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: \
                " + root + "\n"))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.mode == 'test':
            img = self._img_transform(img)
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])
        mask = Image.open(self.masks[index])
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
        return mx.nd.array(np.array(mask), mx.cpu(0)).astype('int32')# - 1

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
    for filename in os.listdir(img_folder):
        basename, _ = os.path.splitext(filename)
        if filename.endswith(".jpg"):
            imgpath = os.path.join(img_folder, filename)
            maskname = basename + '.png'
            maskpath = os.path.join(mask_folder, maskname)
            if os.path.isfile(maskpath):
                try:
                    Image.open(imgpath)
                    Image.open(maskpath)
                except Exception:
                    continue
                img_paths.append(imgpath)
                mask_paths.append(maskpath)
            else:
                print('imagepath', imgpath)
                print('cannot find the mask:', maskpath)
    return img_paths, mask_paths
