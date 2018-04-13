"""Pascal Augmented VOC Semantic Segmentation Dataset."""
import os
import random
import scipy.io
from PIL import Image, ImageOps, ImageFilter
from mxnet.gluon.data import dataset


class VOCAugSegmentation(dataset.Dataset):
    """Pascal VOC Augmented Semantic Segmentation Dataset.

    Parameters
    ----------
    root : string
        Path to VOCdevkit folder. Default is '$(HOME)/mxnet/datasplits/voc'
    split: string
        'train' or 'val'
    transform : callable, optional
        A function that transforms the image
    target_transform : callable, optional
        A function that transforms the labels
    """
    TRAIN_BASE_DIR = 'VOCaug/dataset/'
    def __init__(self, root=os.path.expanduser('~/.mxnet/datasets/voc'),
                 split='train', transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.train = split

        # train/val/test splits are pre-cut
        _voc_root = os.path.join(root, self.TRAIN_BASE_DIR)
        _mask_dir = os.path.join(_voc_root, 'cls')
        _image_dir = os.path.join(_voc_root, 'img')
        if self.train == 'train':
            _split_f = os.path.join(_voc_root, 'trainval.txt')
        elif self.train == 'val':
            _split_f = os.path.join(_voc_root, 'val.txt')
        else:
            raise RuntimeError('Unknown dataset split.')

        self.images = []
        self.masks = []
        with open(os.path.join(_split_f), "r") as lines:
            for line in lines:
                _image = os.path.join(_image_dir, line.rstrip('\n')+".jpg")
                assert os.path.isfile(_image)
                self.images.append(_image)
                _mask = os.path.join(_mask_dir, line.rstrip('\n')+".mat")
                assert os.path.isfile(_mask)
                self.masks.append(_mask)

        assert (len(self.images) == len(self.masks))

    @property
    def classes(self):
        """Category names."""
        return ('background', 'airplane', 'bicycle', 'bird', 'boat', 'bottle',
                'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
                'motorcycle', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
                'tv', 'ambigious')

    @property
    def num_class(self):
        """Number of categories."""
        return len(self.classes)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        if self.train == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])

        target = self._load_mat(self.masks[index])

        # synchrosized transform
        if self.train == 'train':
            img, target = self._sync_transform(img, target)
        elif self.train == 'val':
            img, target = self._val_sync_transform(img, target)

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def _load_mat(self, filename):
        mat = scipy.io.loadmat(filename, mat_dtype=True, squeeze_me=True,
                               struct_as_record=False)
        mask = mat['GTcls'].Segmentation
        return Image.fromarray(mask)

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
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
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
        deg = random.uniform(-10, 10)
        img = img.rotate(deg, resample=Image.BILINEAR)
        mask = mask.rotate(deg, resample=Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop 480
        w, h = img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        img = img.crop((x1, y1, x1+crop_size, y1+crop_size))
        mask = mask.crop((x1, y1, x1+crop_size, y1+crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
        return img, mask

# acronym for easy load
_Segmentation = VOCAugSegmentation
