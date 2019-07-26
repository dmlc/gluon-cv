#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, division
import json, os
from os import path as osp
from zipfile import ZipFile
from gluoncv.utils import download


def extract(fpath, exdir):
    print("Extracting zip file")
    with ZipFile(fpath) as z:
        z.extractall(path=exdir)
    print("Extracting Done")

def make_list(exdir):
    train_dir = osp.join(exdir, "bounding_box_train")
    train_list = {}
    for _, _, files in os.walk(train_dir, topdown=False):
        for name in files:
            if '.jpg' in name:
                name_split = name.split('_')
                pid = name_split[0]
                pcam = name_split[1][1]
                if pid not in train_list:
                    train_list[pid] = []
                train_list[pid].append({"name":name, "pid":pid, "pcam":pcam})


    with open(osp.join(exdir, 'train.txt'), 'w') as f:
        for i, key in enumerate(train_list):
            for item in train_list[key]:
                f.write(item['name']+" "+str(i)+" "+item["pcam"]+"\n")
    print("Make Label List Done")


def main():
    name = "Market-1501-v15.09.15"
    url = "http://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/gluon/dataset/"+name+".zip"
    root = osp.expanduser("~/.mxnet/datasets")
    if not os.path.exists(root):
        os.mkdir(root)
    fpath = osp.join(root, name+'.zip')
    exdir = osp.join(root, name)

    if os.path.exists(fpath):
        if not osp.isdir(exdir):
            extract(fpath, root)
            make_list(exdir)
            
    else:
        download(url, fpath, False)
        extract(fpath, root)
        make_list(exdir)


if __name__ == '__main__':
    main()
