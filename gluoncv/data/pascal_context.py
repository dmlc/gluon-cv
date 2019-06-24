# coding=utf-8
# author: Quan Tang
# 2019/6/21

import os
import numpy as np
from PIL import Image
from tqdm import trange
from .segbase import SegmentationDataset


def install_pcontext_api():
    import shutil
    repo_url = "https://github.com/ccvl/detail-api"
    os.system("git clone " + repo_url)
    os.system("cd detail-api/PythonAPI/ && python setup.py install")
    shutil.rmtree('detail-api')
    try:
        import detail
    except ImportError:
        print("Installing PASCAL Context API failed, please install it manually %s" % repo_url)


# Detail API
try:
    from detail import Detail
except ImportError:
    install_pcontext_api()


def download_trainval_json(root):
    url = "https://codalabuser.blob.core.windows.net/public/trainval_merged.json"
    checksum = "169325d9f7e9047537fedca7b04de4dddf10b881"
    download(url, path=root, sha1_hash=checksum)


def download(url, path=None, overwrite=False, sha1_hash=None):
    """Download an given URL
    Parameters
    ----------
    url : str
        URL to download
    path : str, optional
        Destination path to store downloaded file. By default stores to the
        current directory with same name as in url.
    overwrite : bool, optional
        Whether to overwrite destination file if already exists.
    sha1_hash : str, optional
        Expected sha1 hash in hexadecimal digits. Will ignore existing file when hash is specified
        but doesn't match.
    Returns
    -------
    str
        The file path of the downloaded file.
    """
    import requests
    from tqdm import tqdm
    if path is None:
        fname = url.split('/')[-1]
    else:
        path = os.path.expanduser(path)
        if os.path.isdir(path):
            fname = os.path.join(path, url.split('/')[-1])
        else:
            fname = path

    if overwrite or not os.path.exists(fname) or (sha1_hash and not check_sha1(fname, sha1_hash)):
        dirname = os.path.dirname(os.path.abspath(os.path.expanduser(fname)))
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        print('Downloading %s from %s...' % (fname, url))
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise RuntimeError("Failed downloading url %s" % url)
        total_length = r.headers.get('content-length')
        with open(fname, 'wb') as f:
            if total_length is None:  # no content length header
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
            else:
                total_length = int(total_length)
                for chunk in tqdm(r.iter_content(chunk_size=1024),
                                  total=int(total_length / 1024. + 0.5),
                                  unit='KB', unit_scale=False, dynamic_ncols=True):
                    f.write(chunk)

        if sha1_hash and not check_sha1(fname, sha1_hash):
            raise UserWarning('File {} is downloaded but the content hash does not match. ' \
                              'The repo may be outdated or download may be incomplete. ' \
                              'If the "repo_url" is overridden, consider switching to ' \
                              'the default repo.'.format(fname))

    return fname


def check_sha1(filename, sha1_hash):
    """Check whether the sha1 hash of the file content matches the expected hash.
    Parameters
    ----------
    filename : str
        Path to the file.
    sha1_hash : str
        Expected sha1 hash in hexadecimal digits.
    Returns
    -------
    bool
        Whether the file content matches the expected hash.
    """
    import hashlib
    sha1 = hashlib.sha1()
    with open(filename, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)

    return sha1.hexdigest() == sha1_hash


class PContextSegmentation(SegmentationDataset):
    """ Pascal context semantic segmentation dataset with 60 semantic labels.
    R. Mottaghi, et al. The role of context for object detection and semantic segmentation in the wild. CVPR 2014.
    """
    NUM_CLASSES = 59

    def __init__(self, root=os.path.expanduser('~/.mxnet/dataset/PContext'), split='train', mode=None, transform=None,
                 **kwargs):
        super(PContextSegmentation, self).__init__(root, split, mode, transform, **kwargs)
        # trainval_merged.json
        trainval_merged_json = os.path.join(root, 'trainval_merged.json')
        if not os.path.exists(trainval_merged_json):
            print("Downloading trainval_merged.json...Only for once.")
            download_trainval_json(root)
        # images dir
        self._img_dir = os.path.join(root, 'JPEGImages')
        # .txt split file
        if split == 'train':
            _split_f = os.path.join(root, 'train.txt')
        elif split == 'val':
            _split_f = os.path.join(root, 'val.txt')
        else:
            raise RuntimeError('Unknown dataset split: {}'.format(split))
        if not os.path.exists(_split_f):
            self._generate_split_f(_split_f)
        # 59 + background labels directory
        _mask_dir = os.path.join(root, 'Labels_59')
        if not os.path.exists(_mask_dir):
            self._preprocess_mask(_mask_dir)

        self.images = []
        self.masks = []
        with open(os.path.join(_split_f), 'r') as lines:
            for line in lines:
                _image = os.path.join(self._img_dir, line.strip() + '.jpg')
                assert os.path.isfile(_image)
                self.images.append(_image)

                _mask = os.path.join(_mask_dir, line.strip() + '.png')
                assert os.path.isfile(_mask)
                self.masks.append(_mask)
        assert len(self.images) == len(self.masks)

    def _get_imgs(self, split='trainval'):
        """ get images by split type using Detail API. """
        annotation = os.path.join(self.root, 'trainval_merged.json')
        detail = Detail(annotation, self._img_dir, split)
        imgs = detail.getImgs()
        return imgs, detail

    def _generate_split_f(self, split_f):
        print("Processing %s...Only run once to generate this split file." % (self.split + '.txt'))
        imgs, _ = self._get_imgs(self.split)
        img_list = []
        for img in imgs:
            file_id, _ = img.get('file_name').split('.')
            img_list.append(file_id)
        with open(split_f, 'a') as split_file:
            split_file.write('\n'.join(img_list))

    @staticmethod
    def _class_to_index(mapping, key, mask):
        # assert the values
        values = np.unique(mask)
        for i in range(len(values)):
            assert (values[i] in mapping)
        index = np.digitize(mask.ravel(), mapping, right=True)
        return key[index].reshape(mask.shape)

    def _preprocess_mask(self, _mask_dir):
        print("Processing mask...Only run once to generate 59-class mask.")
        os.makedirs(_mask_dir)
        mapping = np.sort(np.array([
            0, 2, 259, 260, 415, 324, 9, 258, 144, 18, 19, 22,
            23, 397, 25, 284, 158, 159, 416, 33, 162, 420, 454, 295, 296,
            427, 44, 45, 46, 308, 59, 440, 445, 31, 232, 65, 354, 424,
            68, 326, 72, 458, 34, 207, 80, 355, 85, 347, 220, 349, 360,
            98, 187, 104, 105, 366, 189, 368, 113, 115]))
        key = np.array(range(len(mapping))).astype('uint8')
        imgs, detail = self._get_imgs()
        bar = trange(len(imgs))
        for i in bar:
            img = imgs[i]
            img_name, _ = img.get('file_name').split('.')
            mask = Image.fromarray(self._class_to_index(mapping, key, detail.getMask(img)))
            mask.save(os.path.join(_mask_dir, img_name + '.png'))
            bar.set_description("Processing mask {}".format(img.get('image_id')))

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        mask = Image.open(self.masks[idx])
        # synchronized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            raise RuntimeError('Unknown dataset split: {}'.format(self.mode))
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)

        return img, mask

    def __len__(self):
        return len(self.images)

    @property
    def classes(self):
        return (
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'table', 'dog',
            'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', 'bag', 'bed', 'bench',
            'book', 'building', 'cabinet', 'ceiling', 'cloth', 'computer', 'cup', 'door', 'fence', 'floor', 'flower',
            'food', 'grass', 'ground', 'keyboard', 'light', 'mountain', 'mouse', 'curtain', 'platform', 'sign', 'plate',
            'road', 'rock', 'shelves', 'sidewalk', 'sky', 'snow', 'bedclothes', 'track', 'tree', 'truck', 'wall',
            'water', 'window', 'wood')

