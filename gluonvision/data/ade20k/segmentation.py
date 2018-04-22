"""Pascal ADE20K Semantic Segmentation Dataset."""
import os
from PIL import Image
import numpy as np
import mxnet as mx
from ..segbase import SegmentationDataset

class ADE20KSegmentation(SegmentationDataset):
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
    # pylint: disable=abstract-method
    BASE_DIR = 'ADEChallengeData2016'
    def __init__(self, root=os.path.expanduser('~/.mxnet/datasets/ade'),
                 split='train', transform=None, target_transform=None):
        super(ADE20KSegmentation, self).__init__(root)
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
            raise RuntimeError('unknown mode for dataloader: {}'.format(self.mode))

        # general resize, normalize and toTensor
        if self.transform is not None:
            #print("transform for input")
            img = self.transform(img)
        if self.target_transform is not None:
            #print("transform for label")
            mask = self.target_transform(mask)

        return img, mask

    def _mask_transform(self, mask):
        return mx.nd.array(np.array(mask), mx.cpu(0)).astype('int32') - 1

    def __len__(self):
        return len(self.images)

    @property
    def num_class(self):
        """Number of categories."""
        return 150

def _get_ade20k_pairs(folder, mode='train'):
    img_paths = []
    mask_paths = []
    if mode == 'train':
        img_folder = os.path.join(folder, 'images/training')
        mask_folder = os.path.join(folder, 'annotations/training')
    else:
        img_folder = os.path.join(folder, 'images/validation')
        mask_folder = os.path.join(folder, 'annotations/validation')
    for filename in os.listdir(img_folder):
        basename, _ = os.path.splitext(filename)
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
