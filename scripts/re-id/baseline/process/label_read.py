from __future__ import absolute_import

from mxnet.gluon import Block
import os, random
from os import path as osp


def LabelList(ratio=1, name="market1501"):
    root = "../../../scripts/datasets/"

    if name == "market1501":
        path = osp.join(root, "Market-1501-v15.09.15")
        train_txt = osp.join(path, "train.txt")
        image_path = osp.join(path, "bounding_box_train")

        # item_list = [(osp.join(image_path, line.split()[0]), int(line.split()[1]), int(line.split()[2])) for line in open(train_txt).readlines()]
        item_list = [(osp.join(image_path, line.split()[0]), int(line.split()[1])) for line in open(train_txt).readlines()]
        random.shuffle(item_list)
        count = len(item_list)
        train_count = int(count * ratio)

        train_set = item_list[:train_count]
        valid_set = item_list[train_count:]
        
        return train_set, valid_set
