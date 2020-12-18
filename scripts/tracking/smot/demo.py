"""
MXNet implementation of SMOT: Single-Shot Multi Object Tracking
https://arxiv.org/abs/2010.16031
"""
import os
import argparse
import logging
import numpy as np
import cv2

from gluoncv.model_zoo.smot import fartracker as Tracker
from gluoncv.model_zoo.smot.tracktors import GluonSSDMultiClassTracktor


parser = argparse.ArgumentParser("SMOT")
parser.add_argument('filename', type=str)
parser.add_argument('--input-type', type=str, default='video')
parser.add_argument('--tracktor-version', type=str, default='joint')
parser.add_argument('--use-motion', help='whether to use motion information between frames', action='store_true')
parser.add_argument('--motion', type=str, default='farneback')
parser.add_argument('--verbose', help='whether to output all logging', action='store_true')
parser.add_argument('--vis-off', help='whether to turn off visualization', action='store_true')
parser.add_argument('--detect-thresh', type=float, default=0.9)
parser.add_argument('--track-thresh', type=float, default=0.3)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--save-path', type=str, default='/home/ubuntu/smot_vis')


def track_viz(results, video, plot_save_folder):
    video_cap = cv2.VideoCapture(video)
    if not os.path.exists(plot_save_folder):
        os.makedirs(plot_save_folder)

    cnt = 0
    color_list = [(255, 0, 0), (128, 0, 0), (255, 255, 0), (128, 128, 0), (0, 255, 0), (0, 128, 0), (0, 255, 255),
                  (0, 128, 128), \
                  (0, 255, 255), (0, 0, 128), (255, 0, 255), (128, 0, 128)]

    while (video_cap.isOpened()):
        ret, frame = video_cap.read()
        if ret:
            frame_id, trackInfo = results[cnt]
            cnt += 1
            imarr_plot = np.copy(frame)

            for bb in trackInfo:
                track_id, x, y, w, h = bb["track_id"], bb["bbox"]["left"], bb["bbox"]["top"], bb["bbox"]["width"], bb["bbox"]["height"]
                cv2.rectangle(imarr_plot, (int(x), int(y)), (int(x + w), int(y + h)), color_list[track_id % len(color_list)], 2)
            cv2.imwrite(os.path.join(plot_save_folder, ('output_%05d.jpg' % (frame_id))), imarr_plot)
        else:
            break


def decoder_iter(video_file):
    cap = cv2.VideoCapture(video_file)
    assert cap.isOpened(), "Cannot open video file: {}".format(video_file)

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if ret:
            yield frame_id, frame[:, :, ::-1]
            frame_id += 1
        else:
            break


if __name__ == '__main__':
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%Y-%m-%d:%H:%M:%S',
                        level=logging.DEBUG if args.verbose else logging.INFO)

    assert args.input_type == 'video', 'We only support video input at this moment!'

    # get tracktor
    tracktor = GluonSSDMultiClassTracktor(gpu_id=args.gpu,
                                          detector_thresh=args.detect_thresh)
    logging.info('Trackor is loaded')

    # get tracker
    tracker = Tracker.FARTracker(match_top_k=5,
                                 motion_model=args.motion,
                                 use_motion=args.use_motion,
                                 anchor_assignment_method='iou',
                                 track_keep_alive_thresh=args.track_thresh)
    logging.info('Tracker is defined')

    # get MOT results
    results = list(tracker.process_frame_sequence(decoder_iter(args.filename), tracktor))
    logging.info('Tracking is done')

    # dump tracking results for visualization
    if not args.vis_off:
        track_viz(results, args.filename, args.save_path)
        logging.info('Tracking results are saved to %s' % (args.save_path))
