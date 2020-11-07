"""SiamRPN Demo script.
Code adapted from https://github.com/STVIR/pysot"""
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import mxnet as mx
from gluoncv import model_zoo, utils
from gluoncv.model_zoo.siamrpn.siamrpn_tracker import SiamRPNTracker as build_tracker
from gluoncv.model_zoo.siamrpn.siamrpn_tracker import get_axis_aligned_bbox
from gluoncv.utils.filesystem import try_import_cv2
cv2 = try_import_cv2()

def parse_args():
    """ benchmark test."""
    parser = argparse.ArgumentParser(description='make ovject tracking.')
    parser.add_argument('--data-dir', type=str, default='',
                        help='if video-loader set to True, data-dir store videos frames.')
    parser.add_argument('--video-loader', action='store_true', default=True,
                        help='if set to True, read videos directly instead of reading frames.')
    parser.add_argument('--video-path',
                        default=
                        'https://raw.githubusercontent.com/dmlc/web-data/master/gluoncv/tracking/Coke.mp4',
                        help='if set to True, read videos directly instead of reading frames.')
    parser.add_argument('--netwrok', type=str, default='siamrpn_alexnet_v2_otb15',
                        help='SiamRPN network name')
    parser.add_argument('--gt-bbox', type=int, nargs='+', default=[298, 160, 48, 80],
                        help='first frame object location')
    parser.add_argument('--save-dir', type=str, default='./predictions',
                        help='directory of saved results')
    opt = parser.parse_args()
    return opt

def read_data(opt):
    """
    Pre-process data
    --------------------

    Next we need a video or video frame
    if you want to test video frame, you can change opt.video_loader to False
    and opt.data-dir is your video frame path.
    meanwhile you need first frame object coordinates in opt.gt-bbox
    gt_bbox is first frame object coordinates, and it is bbox(center_x,center_y,weight,height)
    """
    video_frames = []
    if opt.video_loader:
        im_video = utils.download(opt.video_path)
        cap = cv2.VideoCapture(im_video)
        while(True):
            ret, img = cap.read()
            if not ret:
                break
            video_frames.append(img)
    else:
        for data in sorted(os.listdir(opt.data_dir)):
            video_frames.append(cv2.imread(os.path.join(opt.data_dir, data)))
    return video_frames

def inference(video_frames, tracker, opt):
    """
    Predict with a SiamRPN and make inference
    --------------------

    this function returns a dictionaries result. which has two keys. one is bbox,
    which represents the coordinates of the predicted frame,
    the other is best_score, which records everyframe best_score.
    Save output in current path
    """
    scores = []
    pred_bboxes = []
    gt_bbox = list(map(int, opt.gt_bbox))
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir)
    for ind, frame in enumerate(video_frames):
        if ind == 0:
            cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
            gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
            tracker.init(frame, gt_bbox_, ctx=mx.cpu())
            pred_bbox = gt_bbox_
            scores.append(None)
            pred_bboxes.append(pred_bbox)
        else:
            outputs = tracker.track(frame, ctx=mx.cpu())
            pred_bbox = outputs['bbox']
            pred_bboxes.append(pred_bbox)
            scores.append(outputs['best_score'])
        pred_bbox = list(map(int, pred_bbox))
        cv2.rectangle(frame, (pred_bbox[0], pred_bbox[1]),
                      (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]),
                      (0, 255, 255), 3)
        cv2.imwrite(os.path.join(opt.save_dir, '%04d.jpg'%(ind+1)), frame)

if __name__ == '__main__':
    opt = parse_args()
    # ######################################################################
    # Load a pretrained model
    # -------------------------
    #
    # Let's get an SiamRPN model trained. We pick the one using Alexnet as the base model.
    # By specifying ``pretrained=True``, it will automatically download the model from the model
    # zoo if necessary. For more pretrained models, please refer to
    # :doc:`../../model_zoo/index`.
    net = model_zoo.get_model(opt.netwrok, ctx=mx.cpu(), pretrained=True)
    tracker = build_tracker(net)
    # Pre-process data
    video_frames = read_data(opt)
    ######################################################################
    plt.imshow(video_frames[0])
    plt.show()
    # Predict with a SiamRPN and make inference
    inference(video_frames, tracker, opt)
