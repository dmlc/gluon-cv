"""
Script to preprocess MOT17 data into our format for easier evaluation.
Please first download MOT17 data from official website
https://motchallenge.net/data/MOT17/
"""

import os
import argparse
import json
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--mot-folder', type=str, required=True)
parser.add_argument('--mot-save-folder', type=str, required=True)
parser.add_argument('--mot-save-npy', type=str, required=True)
parser.add_argument('--endfix', type=str, default='FRCNN')
args = parser.parse_args()


mot_folder = args.mot_folder
mot_save_foldr = args.mot_save_folder
mot_save_npy = args.mot_save_npy
endfix = args.endfix

if not os.path.exists(mot_save_foldr):
    os.makedirs(mot_save_foldr)

if not os.path.exists(mot_save_npy):
    os.makedirs(mot_save_npy)

folder_list = os.listdir(mot_folder)
valid_foder_list = [t for t in folder_list if endfix in t ]

for fd in valid_foder_list:
    full_gt_path = os.path.join(mot_folder, fd, 'gt/gt.txt')
    lines = open(full_gt_path).readlines()
    vid_dic = {}
    vid_array = []
    for bb in lines:
        frameid, pid, x, y, w, h, a, b, c = bb.split(',')
        if int(a) == 1:
            if frameid not in vid_dic:
                vid_dic[frameid] = [[int(pid), float(x), float(y), float(w), float(h)]]
            else:
                vid_dic[frameid].append([int(pid), float(x), float(y), float(w), float(h)])
    fid_list = list(vid_dic.keys())
    fid_list.sort()
    for fid in fid_list:
        for l in vid_dic[fid]:
            vid_array.append([int(fid) -1, l[0], l[1], l[2], l[3], l[4], -1, -1, -1, -1 ]) ## in gt.txt the frame starts from index 1
    out_json = json.dumps(vid_dic, indent=4)
    fp = open(os.path.join(mot_save_foldr, fd + '.json'), 'w')
    fp.write(out_json)
    fp.close()
    np.save(os.path.join(mot_save_npy, 'gt_' + fd), np.array(vid_array))
