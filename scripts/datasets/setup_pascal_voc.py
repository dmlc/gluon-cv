"""
Initialize PASCAL VOC datasets
==============================

. http://host.robots.ox.ac.uk/pascal/VOC/
This example script will try to download dataset from specifed years
(07trainvaltest, 2012trainval by default) if not exist, extract contents to disk
and make symbolic link to '~/.mxnet/datasets/voc' so user can use
voc datasets our-of-box.
"""
import os
import shutil
import argparse
import tarfile
from gluonvision.utils import download, makedirs

_TARGET_DIR = os.path.expanduser('~/.mxnet/datasets/voc')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Initialize PASCAL VOC dataset.',
        epilog='Example: python setup_pascal_voc.py ~/datasets/VOCdevkit --download',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', required=True, help='dataset directory on disk')
    parser.add_argument('--download', action='store_true', help='try download if set')
    parser.add_argument('--overwrite', action='store_true', help='overwrite downloaded if set')
    args = parser.parse_args()
    return args

def download_voc(path, overwrite=False):
    _DOWNLOAD_URLS = [
        ('http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
         '34ed68851bce2a36e2a223fa52c661d592c66b3c'),
        ('http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',
         '41a8d6e12baa5ab18ee7f8f8029b9e11805b4ef1'),
        ('http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',
         '4e443f8a2eca6b1dac8a6c57641b67dd40621a49')]
    download_dir = os.path.join(path, 'downloads')
    makedirs(download_dir)
    for url, checksum in _DOWNLOAD_URLS:
        filename = download(url, path=download_dir, overwrite=overwrite, sha1_hash=checksum)
        # extract
        with tarfile.open(filename) as tar:
            tar.extractall(path=path)


def download_aug(path, overwrite=False):
    _AUG_DOWNLOAD_URLS = [
        ('http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz', '7129e0a480c2d6afb02b517bb18ac54283bfaa35')]
    download_dir = os.path.join(path, 'downloads')
    makedirs(download_dir)
    for url, checksum in _AUG_DOWNLOAD_URLS:
        filename = download(url, path=download_dir, overwrite=overwrite, sha1_hash=checksum)
        # extract
        with tarfile.open(filename) as tar:
            tar.extractall(path=path)
            shutil.move(os.path.join(path, 'benchmark_RELEASE'),
                        os.path.join(path, 'VOCaug'))
            filenames = ['VOCaug/dataset/train.txt', 'VOCaug/dataset/val.txt']
            # generate trainval.txt
            with open(os.path.join(path, 'VOCaug/dataset/trainval.txt'), 'w') as outfile:
                for fname in filenames:
                    fname = os.path.join(path, fname)
                    with open(fname) as infile:
                        for line in infile:
                            outfile.write(line)


if __name__ == '__main__':
    args = parse_args()
    path = os.path.expanduser(args.path)
    if not os.path.isdir(path) or not os.path.isdir(os.path.join(path, 'VOC2007')) \
        or not os.path.isdir(os.path.join(path, 'VOC2012')):
        if not args.download:
            raise ValueError(('{} is not a valid directory, make sure it is present.'
                              ' Or you can try "--download" to grab it'.format(path)))
        else:
            download_voc(path, overwrite=args.overwrite)
            shutil.move(os.path.join(path, 'VOCdevkit', 'VOC2007'), os.path.join(path, 'VOC2007'))
            shutil.move(os.path.join(path, 'VOCdevkit', 'VOC2012'), os.path.join(path, 'VOC2012'))
            shutil.rmtree(os.path.join(path, 'VOCdevkit'))

    if not os.path.isdir(os.path.join(path, 'VOCaug')):
        if not args.download:
            raise ValueError(('{} is not a valid directory, make sure it is present.'
                              ' Or you can try "--download" to grab it'.format(path)))
        else:
            download_aug(path, overwrite=args.overwrite)

    # make symlink
    makedirs(os.path.expanduser('~/.mxnet/datasets'))
    if os.path.isdir(_TARGET_DIR):
        os.remove(_TARGET_DIR)
    os.symlink(args.path, _TARGET_DIR)
