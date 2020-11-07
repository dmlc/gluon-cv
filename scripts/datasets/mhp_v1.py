"""Prepare Multi-Human Parsing V1 dataset"""
import os
import shutil
import argparse
import zipfile
from gluoncv.utils import makedirs
from gluoncv.utils.filesystem import try_import_gdfDownloader, try_import_html5lib

_TARGET_DIR = os.path.expanduser('~/.mxnet/datasets/mhp')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Initialize MHP V1 dataset.',
        epilog='Example: python mhp_v1.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--download-dir', default=None, help='dataset directory on disk')
    args = parser.parse_args()
    return args


def download_mhp_v1(path, overwrite=False):
    try_import_html5lib()
    gdf = try_import_gdfDownloader()
    downloader = gdf.googleDriveFileDownloader()

    file_link = 'https://drive.google.com/uc?id=1hTS8QJBuGdcppFAr_bvW2tsD9hW_ptr5&export=download'
    download_dir = os.path.join(path, 'downloads')
    makedirs(download_dir)
    filename = os.path.join(download_dir, 'LV-MHP-v1.zip')

    # donwload MHP_v1 zip file with Google-Drive-File-Downloader
    downloader.downloadFile(file_link)

    # move zip file to download_dir
    shutil.move('./LV-MHP-v1.zip', filename)

    # extract
    with zipfile.ZipFile(filename,"r") as zip_ref:
        zip_ref.extractall(path=path)


if __name__ == '__main__':
    args = parse_args()
    makedirs(os.path.expanduser('~/.mxnet/datasets'))
    if args.download_dir is not None:
        if os.path.isdir(_TARGET_DIR):
            os.remove(_TARGET_DIR)
        # make symlink
        os.symlink(args.download_dir, _TARGET_DIR)

    download_mhp_v1(_TARGET_DIR, overwrite=False)
