from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import argparse
import os
import cv2
import torch
import numpy as np
import pdb
from gluoncv.model_zoo.siamrpn.siam_net import SiamrpnNet as ModelBuilder
from gluoncv.utils.siamrpn_tracker import SiamRPNTracker as build_tracker
from gluoncv.data.otb.tracking import OTBTracking as OTBDataset
from gluoncv.utils.siamrpn_tracker import get_axis_aligned_bbox
from mxnet import nd, gpu, gluon, autograd
parser = argparse.ArgumentParser(description='siamrpn tracking test result')
parser.add_argument('--dataset',default='OTB2015', type=str,help='dataset name')
parser.add_argument('--dataset_root', type=str,help='dataset_root')
parser.add_argument('--config',type=str,help='config file')
parser.add_argument('--model_path', type=str,help='path of models to eval')
parser.add_argument('--results_path', type=str,help='results path')
parser.add_argument('--video', default='', type=str,
        help='eval one special video')
parser.add_argument('--vis', action='store_true',
        help='whether visualzie result')
args = parser.parse_args()

def test(dataset,tracker):
    model_name = args.model_path.split('/')[-1].split('.')[0]
    for v_idx, video in enumerate(dataset):
        if args.video != '':
            if video.name != args.video:
                continue
        toc = 0
        pred_bboxes = []
        scores = []
        track_times = []
        for idx, (img, gt_bbox) in enumerate(video):
            tic = cv2.getTickCount()
            if idx == 0:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                tracker.init(img, gt_bbox_)
                pred_bbox = gt_bbox_
                scores.append(None)
                pred_bboxes.append(pred_bbox)
            else:
                outputs = tracker.track(img)
                pred_bbox = outputs['bbox']
                pred_bboxes.append(pred_bbox)
                scores.append(outputs['best_score'])
            toc += cv2.getTickCount() - tic
            track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
            if idx == 0:
                cv2.destroyAllWindows()
            if args.vis and idx > 0:
                gt_bbox = list(map(int, gt_bbox))
                pred_bbox = list(map(int, pred_bbox))
                cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                                (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)
                cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                                (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)
                cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow(video.name, img)
                cv2.waitKey(1)
        toc /= cv2.getTickFrequency()
        model_path = os.path.join(args.results_path, args.dataset, model_name)
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        result_path = os.path.join(model_path, '{}.txt'.format(video.name))
        with open(result_path, 'w') as f:
            for x in pred_bboxes:
                f.write(','.join([str(i) for i in x])+'\n')
        print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
            v_idx+1, video.name, toc, idx / toc))


def main():
    dataset = OTBDataset(name=args.dataset,dataset_root=args.dataset_root,load_img=False)
    model = ModelBuilder()
    model.load_parameters(args.model_path,ctx=gpu()) 
    tracker = build_tracker(model)
    test(dataset,tracker)

if __name__ == '__main__':
    main()
