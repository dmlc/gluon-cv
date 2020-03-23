#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import json
import argparse
from os import walk
from os import path as osp
from zipfile import ZipFile
from gluoncv.utils import download, makedirs


def parse_args():
    parser = argparse.ArgumentParser(
        description='Initialize Market1501 dataset.',
        epilog='Example: python market1501.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--download-dir', type=str, default='~/.mxnet/datasets/', help='dataset directory on disk')
    parser.add_argument('--no-download', action='store_true', help='disable automatic download if set')
    args = parser.parse_args()
    return args


def extract(fpath, exdir):
    print("Extracting zip file")
    with ZipFile(fpath) as z:
        z.extractall(path=exdir)
    print("Extracting Done")


def make_list(exdir):
    train_dir = osp.join(exdir, "bounding_box_train")
    train_list = {}
    for _, _, files in walk(train_dir, topdown=False):
        for name in files:
            if '.jpg' in name:
                name_split = name.split('_')
                pid = name_split[0]
                pcam = name_split[1][1]
                if pid not in train_list:
                    train_list[pid] = []
                train_list[pid].append({"name": name, "pid": pid, "pcam": pcam})

    with open(osp.join(exdir, 'train.txt'), 'w') as f:
        for i, key in enumerate(train_list):
            for item in train_list[key]:
                f.write(item['name'] + " " + str(i) + " " + item["pcam"] + "\n")
    print("Make Label List Done")


def main():
    args = parse_args()
    name = "Market-1501-v15.09.15"
    url = "http://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/{name}.zip".format(name=name)
    root = osp.expanduser(args.download_dir)
    makedirs(root)
    fpath = osp.join(root, name + '.zip')
    exdir = osp.join(root, name)
    if not osp.exists(fpath) and not osp.isdir(exdir) and args.no_download:
        raise ValueError(('{} dataset archive not found, make sure it is present.'
                          ' Or you should not disable "--no-download" to grab it'.format(fpath)))
    # Download by default
    if not args.no_download:
        print('Downloading dataset')
        download(url, fpath, overwrite=False)
        print('Dataset downloaded')
    # Extract dataset if fresh copy downloaded or existing archive is yet to be extracted
    if not args.no_download or not osp.isdir(exdir):
        extract(fpath, root)
        make_list(exdir)


if __name__ == '__main__':
    main()
