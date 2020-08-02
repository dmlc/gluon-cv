"""this script is used to prepare Otb2015 dataset for tracking,
which is Single Object Tracking benchmark"""

import argparse
import tarfile
import os
import time
import json
import numpy as np
from gluoncv.utils import download, makedirs

otb50 = ['Basketball','Biker','Bird1','BlurBody','BlurCar2','BlurFace','BlurOwl',
'Bolt','Box','Car1','Car4','CarDark','CarScale','ClifBar','Couple','Crowds','David','Deer',
'Diving','DragonBaby','Dudek','Football','Freeman4','Girl','Human3','Human4','Human6','Human9',
'Ironman','Jump','Jumping','Liquor','Matrix','MotorRolling','Panda','RedTeam','Shaking','Singer2',
'Skating1','Skating2','Skiing','Soccer','Surfer','Sylvester','Tiger2','Trellis','Walking',
'Walking2','Woman']
otb100 = ['Bird2','BlurCar1','BlurCar3','BlurCar4','Board','Bolt2','Boy',
'Car2','Car24','Coke','Coupon','Crossing','Dancer','Dancer2','David2',
'David3','Dog','Dog1','Doll','FaceOcc1','FaceOcc2','Fish','FleetFace','Football1',
'Freeman1','Freeman3','Girl2','Gym','Human2','Human5','Human7','Human8','Jogging',
'KiteSurf','Lemming','Man','Mhyang','MountainBike','Rubik','Singer1','Skater',
'Skater2','Subway','Suv','Tiger1','Toy','Trans','Twinnings','Vase']

def parse_args():
    """Otb2015 dataset parameter."""
    parser = argparse.ArgumentParser(
        description='Download Otb2015 and prepare for tracking')
    parser.add_argument('--download-dir', type=str, default='~/.mxnet/datasets/otb/',
                        help='dataset directory on disk')
    args = parser.parse_args()
    args.download_dir = os.path.expanduser(args.download_dir)
    return args

def download_otb(args, overwrite=False):
    """download otb2015 dataset and Unzip to download_dir"""
    _DOWNLOAD_URLS = 'http://cvlab.hanyang.ac.kr/tracker_benchmark/seq/'
    if not os.path.isdir(args.download_dir):
        makedirs(args.download_dir)
    for per_otb50 in otb50:
        url = os.path.join(_DOWNLOAD_URLS,per_otb50+'.zip')
        filename = download(url, path=args.download_dir, overwrite=overwrite)
        with zipfile.ZipFile(filename) as zf:
            zf.extractall(path=args.download_dir)
    for per_otb100 in otb100:
        url = os.path.join(_DOWNLOAD_URLS,per_otb100+'.zip')
        filename = download(url, path=args.download_dir, overwrite=overwrite)
        with zipfile.ZipFile(filename) as zf:
            zf.extractall(path=args.download_dir)
    
    os.rename(os.path.join(args,'Jogging'),os.path.join(args,'Jogging-1'))
    os.rename(os.path.join(args,'Jogging'),os.path.join(args,'Jogging-2'))
    os.rename(os.path.join(args,'Skating2'),os.path.join(args,'Skating2-1'))
    os.rename(os.path.join(args,'Skating2'),os.path.join(args,'Skating2-2'))
    os.rename(os.path.join(args,' Human4'),os.path.join(args,'Human4-2'))
    

def main(args):
    # download otb2015 dataset
    download_otb(args)
    print('otb2015 dataset has already download completed')

if __name__ == '__main__':
    since = time.time()
    args = parse_args()
    main(args)
    time_elapsed = time.time() - since
    print('Total complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
