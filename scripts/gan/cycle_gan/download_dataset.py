"""Prepare cycle gan datasets"""
import os
import argparse
import zipfile
from gluoncv.utils import download, makedirs

def parse_args():
    parser = argparse.ArgumentParser(
        description='Initialize Cycle Gan dataset.',
        epilog='Example: python download_dataset.py --download-dir ./',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--download-dir', type=str, default='./', help='dataset directory on disk')
    parser.add_argument('--overwrite', action='store_true', help='overwrite downloaded files if set, in case they are corrupted')
    parser.add_argument('--file',type=str,default='horse2zebra',choices=['apple2orange','summer2winter_yosemite','horse2zebra','monet2photo','cezanne2photo','ukiyoe2photo','vangogh2photo','maps','cityscapes','facades','iphone2dslr_flower','ae_photos'],
                        help='Available datasets are: apple2orange, summer2winter_yosemite, horse2zebra, monet2photo, cezanne2photo, ukiyoe2photo, vangogh2photo, maps, cityscapes, facades, iphone2dslr_flower, ae_photos')
    args = parser.parse_args()
    return args

#####################################################################################
# Download and extract VOC datasets into ``path``

def download_data(path,file, overwrite=False):
    _DOWNLOAD_URL = 'https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/'
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
