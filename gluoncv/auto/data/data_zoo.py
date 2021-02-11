"""Registry of most widely used datasets"""
import os
import shutil
import tarfile
import zipfile
from ...data import VOCDetection, COCODetection
from ...utils import download as _download, makedirs

__all__ = ['get_dataset', 'list_dataset']

_DATASETS = {}


def list_dataset():
    """List available dataset registered in auto data.

    Returns
    -------
    list
        List of names.

    """
    return list(_DATASETS.keys())

def get_dataset(ds_name, train=True, download=True):
    """Get the auto dataset with `ds_name`.

    Parameters
    ----------
    ds_name : str
        The name of dataset, you may view registered with `list_dataset`.
    train : bool, default is `True`
        If `True`, get the training set, if `False`, try to get the validation set.
    download : bool, default is `True`
        If `True`, download the dataset automatically if not found on disk.

    """
    if not ds_name in _DATASETS:
        choices = str(_DATASETS.keys())
        raise ValueError('{} not found. Available choices: {}'.format(ds_name, choices))
    ret = _DATASETS[ds_name](download=download)
    if not isinstance(ret, (list, tuple)):
        if train:
            return ret
        else:
            raise ValueError('No validation dataset available for {}'.format(ds_name))
    if train:
        return ret[0]
    else:
        if len(ret) < 2:
            raise ValueError('No validation dataset available for {}'.format(ds_name))
        return ret[1]

def _register_dataset(ds_name, fn):
    if ds_name in _DATASETS:
        raise ValueError('{} already been registered'.format(ds_name))
    assert callable(fn), "register with non-function is not allowed"
    _DATASETS[ds_name] = fn

def _pascal_0712_detection(download=True):
    root = os.environ.get('MXNET_HOME', os.path.join('~', '.mxnet'))
    root = os.path.expanduser(os.path.join(root, 'datasets', 'voc'))
    if not os.path.isdir(root):
        if not download:
            raise OSError('No `pascal` dataset found on disk and `download=False`')
        # download automatically
        makedirs(root)
        filename = _download('https://s3.amazonaws.com/fast-ai-imagelocal/pascal-voc.tgz',
                             path=root, overwrite=True)
        with tarfile.open(filename) as tar:
            tar.extractall(path=root)
        shutil.move(os.path.join(root, 'pascal-voc', 'VOC2007'), os.path.join(root, 'VOC2007'))
        shutil.move(os.path.join(root, 'pascal-voc', 'VOC2012'), os.path.join(root, 'VOC2012'))
        shutil.rmtree(os.path.join(root, 'pascal-voc'))
    return VOCDetection(root=root, splits=[(2007, 'trainval'), (2012, 'trainval')]), \
           VOCDetection(root=root, splits=[(2007, 'test')])

def _coco_2017_detection(download=True):
    root = os.environ.get('MXNET_HOME', os.path.join('~', '.mxnet'))
    root = os.path.expanduser(os.path.join(root, 'datasets', 'coco'))
    if not os.path.isdir(root):
        if not download:
            raise OSError('No `coco` dataset found on disk and `download=False`')
        # download automatically
        makedirs(root)
        _DOWNLOAD_URLS = [
            ('http://images.cocodataset.org/zips/train2017.zip',
             '10ad623668ab00c62c096f0ed636d6aff41faca5'),
            ('http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
             '8551ee4bb5860311e79dace7e79cb91e432e78b3'),
            ('http://images.cocodataset.org/zips/val2017.zip',
             '4950dc9d00dbe1c933ee0170f5797584351d2a41'),
        ]
        for url, checksum in _DOWNLOAD_URLS:
            filename = _download(url, path=root, overwrite=True, sha1_hash=checksum)
            # extract
            with zipfile.ZipFile(filename) as zf:
                zf.extractall(path=root)
    return COCODetection(root=root, splits='instances_train2017'), \
           COCODetection(root=root, splits='instances_val2017', skip_empty=False)

_register_dataset('pascal_voc_0712_detection', _pascal_0712_detection)
_register_dataset('coco_2017_detection', _coco_2017_detection)
