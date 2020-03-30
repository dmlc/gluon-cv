""" SiamRPN test
Code adapted from https://github.com/STVIR/pysot"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import argparse
import os
import numpy as np
import mxnet as mx
from gluoncv.model_zoo.siamrpn.siamrpn_tracker import SiamRPNTracker as build_tracker
from gluoncv.data.otb.tracking import OTBTracking as OTBDataset
from gluoncv.model_zoo.siamrpn.siamrpn_tracker import get_axis_aligned_bbox
from gluoncv.model_zoo import get_model
from gluoncv.utils.filesystem import try_import_cv2

def parse_args():
    """parameter test."""
    parser = argparse.ArgumentParser(description='siamrpn tracking test result')
    parser.add_argument('--dataset', default='OTB2015', type=str, help='dataset name')
    parser.add_argument('--dataset-root', type=str, default='~/.mxnet/datasets/OTB2015',
                        help='dataset root')
    parser.add_argument('--model-path', type=str, help='path of models to eval')
    parser.add_argument('--results-path', type=str, help='results path')
    parser.add_argument('--video', default='', type=str,
                        help='eval one special video')
    parser.add_argument('--vis', action='store_true',
                        help='whether visualzie result')
    parser.add_argument('--mode', type=str, default='hybrid',
                        help='mode in which to train the model.options are symbolic, hybrid')
    parser.add_argument('--num-gpus', type=int, default=0,
                        help='number of gpus to use.')
    parser.add_argument('--model-name', type=str, default='siamrpn_alexnet_v2_otb15',
                        help='name of model.')
    parser.add_argument('--num-workers', default=4, type=int,
                        help='number of preprocessing workers')
    parser.add_argument('--pretrained', action='store_true', default='True',
                        help='enable using pretrained model from gluon.')
    parser.add_argument('--use-gpu', action='store_true', default='True',
                        help='Whether to use GPU.')
    opt = parser.parse_args()
    return opt
def main():
    """SiamRPN test.

    function
    ----------
        record the output of the model. The output information of each video is recorded in the txt
    corresponding to the video name.
        if you want to evaluation, you need to python benchmark.py according to txt of text result.
        Currently only supports test OTB 2015 dataset

    Parameters
    ----------
    dataset_root : str, default '~/mxnet/datasets/OTB2015'
                   Path to folder test the dataset.
    model_path :   str, Path of test model .
    results_path:  str, Path to store txt of test reslut .
    """
    opt = parse_args()
    if opt.use_gpu:
        ctx = mx.gpu()
    else:
        ctx = mx.cpu()
    # dataloader
    dataset = OTBDataset(name=opt.dataset, dataset_root=opt.dataset_root, load_img=False)
    net = get_model(opt.model_name, pretrained=True)
    net.collect_params().reset_ctx(ctx)
    if opt.mode == 'hybrid':
        net.hybridize(static_alloc=True, static_shape=True)
    if opt.model_path:
        net.load_parameters(opt.model_path, ctx=ctx)
        print('Pre-trained model %s is successfully loaded.' % (opt.model_path))
    else:
        print('Pre-trained model is successfully loaded from the model zoo.')
    # bulid tracker
    tracker = build_tracker(net)
    # record the output of the model.
    test(dataset, tracker, opt, ctx)

def test(dataset, tracker, opt, ctx):
    """SiamRPN test."""
    cv2 = try_import_cv2()
    for v_idx, video in enumerate(dataset):
        if opt.video != '':
            if video.name != opt.video:
                continue
        toc = 0
        pred_bboxes = []
        scores = []
        track_times = []
        for idx, (img, gt_bbox) in enumerate(video):
            tic = cv2.getTickCount()
            if idx == 0:
                x_max, y_max, gt_w, gt_t = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [x_max-(gt_w-1)/2, y_max-(gt_t-1)/2, gt_w, gt_t]
                tracker.init(img, gt_bbox_, ctx)
                pred_bbox = gt_bbox_
                scores.append(None)
                pred_bboxes.append(pred_bbox)
            else:
                outputs = tracker.track(img, ctx)
                pred_bbox = outputs['bbox']
                pred_bboxes.append(pred_bbox)
                scores.append(outputs['best_score'])
            toc += cv2.getTickCount() - tic
            track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
            if idx == 0:
                cv2.destroyAllWindows()
            if opt.vis and idx > 0:
                gt_bbox = list(map(int, gt_bbox))
                pred_bbox = list(map(int, pred_bbox))
                cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                              (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]),
                              (0, 255, 0), 3)
                cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                              (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]),
                              (0, 255, 255), 3)
                cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow(video.name, img)
                cv2.waitKey(1)
        toc /= cv2.getTickFrequency()
        model_path = os.path.join(opt.results_path, opt.dataset,
                                  opt.model_path.split('/')[-1].split('.')[0])
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        with open(os.path.join(model_path, '{}.txt'.format(video.name)), 'w') as f_w:
            for per_pbbox in pred_bboxes:
                f_w.write(','.join([str(i) for i in per_pbbox])+'\n')
        print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format
              (v_idx+1, video.name, toc, len(video) / toc))

if __name__ == '__main__':
    main()
