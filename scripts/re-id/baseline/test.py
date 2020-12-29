# -*- coding: utf-8 -*-
from __future__ import print_function, division

import mxnet as mx
import numpy as np
from mxnet import gluon, nd
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

from networks import resnet18, resnet34, resnet50
import gluoncv as gcv
gcv.utils.check_version('0.6.0')
from gluoncv.data.market1501.data_read import ImageTxtDataset

import time, os, sys
import scipy.io as sio
from os import path as osp

def get_data(batch_size, test_set, query_set):
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform_test = transforms.Compose([
        transforms.Resize(size=(128, 384), interpolation=1),
        transforms.ToTensor(),
        normalizer])

    test_imgs = ImageTxtDataset(test_set, transform=transform_test)
    query_imgs = ImageTxtDataset(query_set, transform=transform_test)

    test_data = gluon.data.DataLoader(test_imgs, batch_size, shuffle=False, last_batch='keep', num_workers=8)
    query_data = gluon.data.DataLoader(query_imgs, batch_size, shuffle=False, last_batch='keep', num_workers=8)
    return test_data, query_data


def load_network(network, ctx):
    network.load_parameters('params/resnet50.params', ctx=ctx, allow_missing=True, ignore_extra=True)
    return network


def fliplr(img):
    '''flip horizontal'''
    img_flip = nd.flip(img, axis=3)
    return img_flip


def extract_feature(model, dataloaders, ctx):
    count = 0
    features = []
    for img, _ in dataloaders:
        n = img.shape[0]
        count += n
        print(count)
        ff = np.zeros((n, 2048))
        for i in range(2):
            if(i==1):
                img = fliplr(img)
            f = model(img.as_in_context(ctx)).as_in_context(mx.cpu()).asnumpy()
            ff = ff+f
        features.append(ff)
    features = np.concatenate(features)
    return features/np.linalg.norm(features, axis=1, keepdims=True)


def get_id(img_path):
    cameras = []
    labels = []
    for path in img_path:
        cameras.append(int(path[0].split('/')[-1].split('_')[1][1]))
        labels.append(path[1])
    return np.array(cameras), np.array(labels)


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = np.zeros(len(index))
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc


if __name__ == '__main__':
    batch_size = 256
    data_dir = osp.expanduser("~/.mxnet/datasets/Market-1501-v15.09.15/")
    gpu_ids = [0]

    # set gpu ids
    if len(gpu_ids)>0:
        context = mx.gpu()

    test_set = [(osp.join(data_dir,'bounding_box_test',line), int(line.split('_')[0])) for line in os.listdir(data_dir+'bounding_box_test') if "jpg" in line and "-1" not in line]
    query_set = [(osp.join(data_dir,'query',line), int(line.split('_')[0])) for line in os.listdir(data_dir+'query') if "jpg" in line]

    test_cam, test_label = get_id(test_set)
    query_cam, query_label = get_id(query_set)

    ######################################################################
    # Load Collected data Trained model
    model_structure = resnet50(ctx=context, pretrained=False)
    model = load_network(model_structure, context)

    # Extract feature
    test_loader, query_loader = get_data(batch_size, test_set, query_set)
    print('start test')
    test_feature = extract_feature(model, test_loader, context)
    print('start query')
    query_feature = extract_feature(model, query_loader, context)


    query_feature = nd.array(query_feature).as_in_context(mx.gpu(0))
    test_feature = nd.array(test_feature).as_in_context(mx.gpu(0))

    num = query_label.size
    dist_all = nd.linalg.gemm2(query_feature, test_feature, transpose_b=True)

    CMC = np.zeros(test_label.size)
    ap = 0.0
    for i in range(num):
        cam = query_cam[i]
        label = query_label[i]
        index = dist_all[i].argsort(is_ascend=False).as_in_context(mx.cpu()).asnumpy().astype("int32")

        query_index = np.argwhere(test_label==label)
        camera_index = np.argwhere(test_cam==cam)

        good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
        junk_index = np.intersect1d(query_index, camera_index)

        ap_tmp, CMC_tmp = compute_mAP(index, good_index, junk_index)
        CMC = CMC + CMC_tmp
        ap += ap_tmp

    CMC = CMC/num #average CMC
    print('top1:%f top5:%f top10:%f mAP:%f'%(CMC[0],CMC[4],CMC[9],ap/num))
