"""Prepare ADE20K datasets. 
========================

Scene parsing http://sceneparsing.csail.mit.edu/ is to segment and parse an image into different image regions associated with semantic categories, such as sky, road, person, and bed. The data for this benchmark comes from ADE20K Dataset which contains more than 20K scene-centric images exhaustively annotated with objects and object parts. g. There are totally 150 semantic categories included for evaluation, which include stuffs like sky, road, grass, and discrete objects like person, car, bed.

.. image:: http://groups.csail.mit.edu/vision/datasets/ADE20K/assets/images/examples.png
    :width: 500 px

Preprocess the Dataset
----------------------

This example script will try to download dataset if not exist, extract contents to disk
and make symbolic link to '~/.mxnet/datasets/ade' so user can use
ade datasets our-of-box.

- Create symbolic link for existing dataset in the folder `~/Datasets/ade`:

.. code-block:: bash

    python examples/setup_pascal_voc.py --path ~/Dataset/ade/

- Download the dataset to `~/Datasets/ade` and make symlink

.. code-block:: bash

    python examples/setup_pascal_voc.py --path ~/Dataset/ade/ --download

Dive Deep into Source Code
--------------------------

"""
import os
import shutil
import argparse
import zipfile
from gluonvision.utils import download, makedirs

_TARGET_DIR = os.path.expanduser('~/.mxnet/datasets/ade')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Initialize ADE20K dataset.',
        epilog='Example: python setup_ade20k.py --path ~/Datasets --download',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', required=True, help='dataset directory on disk')
    parser.add_argument('--download', action='store_true', help='try download if set')
    parser.add_argument('--overwrite', action='store_true', help='overwrite downloaded if set')
    args = parser.parse_args()
    return args


def download_aug(path, overwrite=False):
    _AUG_DOWNLOAD_URLS = [
        ('http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip', '219e1696abb36c8ba3a3afe7fb2f4b4606a897c7'),
        ('http://data.csail.mit.edu/places/ADEchallenge/release_test.zip', 'e05747892219d10e9243933371a497e905a4860c'),]
    download_dir = os.path.join(path, 'downloads')
    makedirs(download_dir)
    for url, checksum in _AUG_DOWNLOAD_URLS:
        filename = download(url, path=download_dir, overwrite=overwrite, sha1_hash=checksum)
        # extract
        with zipfile.ZipFile(filename,"r") as zip_ref:
            zip_ref.extractall(path=path)
        

if __name__ == '__main__':
    args = parse_args()
    path = os.path.expanduser(args.path)
    if not os.path.isdir(os.path.join(path, 'ADEChallengeData2016')):
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
