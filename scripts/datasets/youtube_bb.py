"""this script is used to prepare Youtube-bb dataset for tracking,
which is YouTube-BoundingBoxes Dataset"""
import os
import time
import argparse
import tarfile
from gluoncv.utils import download, makedirs

def parse_args():
    """Youtube_bb dataset parameter."""
    parser = argparse.ArgumentParser(
        description='Download Youtube_bb dataset and prepare for tracking')
    parser.add_argument('--download-dir', type=str, default='~/.mxnet/datasets/Youtube_bb/',
                        help='dataset directory on disk')
    args = parser.parse_args()
    args.download_dir = os.path.expanduser(args.download_dir)
    return args

def download_youtube_bb(args, overwrite=False):
    """
    download Youtube-bb dataset and Unzip to download_dir
    The data has been crop
    """
    url_json = 'https://yizhu-migrate-data.s3.amazonaws.com/YouTube-BB/train.json'
    url_train = 'https://yizhu-migrate-data.s3.amazonaws.com/YouTube-BB/train'
    url_val = 'https://yizhu-migrate-data.s3.amazonaws.com/YouTube-BB/val'
    download(url_json, path=args.download_dir, overwrite=overwrite)
    if not isdir(args.download_dir):
        makedirs(args.download_dir)
    for i in range(100):
        print(url_train+str(i).zfill(4)+'.tar')
        filename = download(url_train+str(i).zfill(4)+'.tar', path=args.download_dir, overwrite=overwrite)
        with tarfile.open(filename) as tar:
            tar.extractall(path=args.download_dir)
        filename = download(url_val+str(i).zfill(4)+'.tar', path=args.download_dir, overwrite=overwrite)
        with tarfile.open(filename) as tar:
            tar.extractall(path=args.download_dir)

def main(args):
    # download Youtube_bb dataset
    download_youtube_bb(args)
    print('Youtube_bb dataset for tracking has already download completed')

if __name__ == '__main__':
    since = time.time()
    args = parse_args()
    main(args)
    time_elapsed = time.time() - since
    print('Total complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
