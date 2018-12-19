"""Prepare DIV2K - bicubic downscaling x4 competition datasets"""
import os
import argparse
import zipfile
from gluoncv.utils import download, makedirs

def parse_args():
    parser = argparse.ArgumentParser(
        description='Initialize DIV2K - bicubic downscaling x4 competition dataset.',
        epilog='Example: python download_dataset.py --download-dir ./',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--download-dir', type=str, default='./', help='dataset directory on disk')
    parser.add_argument('--overwrite', action='store_true', help='overwrite downloaded files if set, in case they are corrputed')
    parser.add_argument('--file',type=str,default='DIV2K_train_HR',choices=['DIV2K_train_HR','DIV2K_test_LR_bicubic_X4','DIV2K_valid_HR','DIV2K_valid_LR_bicubic_X4'])
    args = parser.parse_args()
    return args

#####################################################################################
# Download and extract datasets into ``path``

def download_data(path,file, overwrite=False):
    _DOWNLOAD_URL = 'https://data.vision.ee.ethz.ch/cvl/DIV2K/'
    filename = download(_DOWNLOAD_URL + file, path=path, overwrite=overwrite)
    # extract
    with zipfile.ZipFile(filename,'r') as zip:
        zip.extractall(path=path)



if __name__ == '__main__':
    args = parse_args()
    args.file = args.file + '.zip'
    path = os.path.expanduser(args.download_dir)
    if not os.path.isdir(path) :
        makedirs(path)
    download_data(path, args.file,overwrite=args.overwrite)
