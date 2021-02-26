"""
MXNet implementation of SMOT: Single-Shot Multi Object Tracking
https://arxiv.org/abs/2010.16031
"""
import os
import argparse
import logging
import numpy as np
import cv2

from gluoncv.model_zoo.smot import smot_tracker as Tracker
from gluoncv.model_zoo.smot.tracktors import GluonSSDMultiClassTracktor


parser = argparse.ArgumentParser("SMOT")
parser.add_argument('filename', type=str, help='could be a path to a video or a folder of images')
parser.add_argument('--input-type', type=str, default='video', choices=['video', 'images'])
parser.add_argument('--tracktor-version', type=str, default='joint')
parser.add_argument('--use-motion', help='whether to use motion information between frames', action='store_true')
parser.add_argument('--motion', type=str, default='farneback')
parser.add_argument('--verbose', help='whether to output all logging', action='store_true')
parser.add_argument('--vis-off', help='whether to turn off visualization', action='store_true')
parser.add_argument('--eval', help='whether to save results for MOT17 evaluation', action='store_true')
parser.add_argument('--detect-thresh', type=float, default=0.9)
parser.add_argument('--track-thresh', type=float, default=0.3)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--save-path', type=str, default='./smot_vis')
parser.add_argument('--save-filename', type=str, default='pred.npy')
parser.add_argument('--param-path', type=str, default='')
parser.add_argument('--network-name', type=str, default='')
parser.add_argument('--use-pretrained', action='store_true', help='enable using pretrained model from gluon.')
parser.add_argument('--custom-classes', type=str,nargs='+', default=["person"])
parser.add_argument('--data-shape', type=int, default=512)


def track_viz_video(results, video, plot_save_folder):
    video_cap = cv2.VideoCapture(video)

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


def track_viz_image(results, video_folder, plot_save_folder):
    cnt = 0
    color_list = [(255, 0, 0), (128, 0, 0), (255, 255, 0), (128, 128, 0), (0, 255, 0), (0, 128, 0), (0, 255, 255),
                  (0, 128, 128), \
                  (0, 255, 255), (0, 0, 128), (255, 0, 255), (128, 0, 128)]

    video_frames = os.listdir(video_folder)
    video_frames.sort()

    for frame_id, frame_path in enumerate(video_frames):
        frame = cv2.imread(os.path.join(video_folder, frame_path))

        frame_id, trackInfo = results[cnt]
        cnt += 1
        imarr_plot = np.copy(frame)

        for bb in trackInfo:
            track_id, x, y, w, h = bb["track_id"], bb["bbox"]["left"], bb["bbox"]["top"], bb["bbox"]["width"], bb["bbox"]["height"]
            cv2.rectangle(imarr_plot, (int(x), int(y)), (int(x + w), int(y + h)), color_list[track_id % len(color_list)], 2)
        cv2.imwrite(os.path.join(plot_save_folder, ('output_%05d.jpg' % (frame_id))), imarr_plot)


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


def imageloader_iter(video_folder):
    # assume the video frames can be ordered by naming
    video_frames = os.listdir(video_folder)
    video_frames.sort()

    for frame_id, frame_path in enumerate(video_frames):
        frame = cv2.imread(os.path.join(video_folder, frame_path))
        yield frame_id, frame[:, :, ::-1]


if __name__ == '__main__':
    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%Y-%m-%d:%H:%M:%S',
                        level=logging.DEBUG if args.verbose else logging.INFO)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # get tracktor
    tracktor = GluonSSDMultiClassTracktor(gpu_id=args.gpu,
                                          detector_thresh=args.detect_thresh,
                                          model_name=args.network_name,
                                          use_pretrained=args.use_pretrained,
                                          param_path=args.param_path,
                                          data_shape=args.data_shape
                                          )
    logging.info('Trackor is loaded')

    # get tracker
    tracker = Tracker.SMOTTracker(match_top_k=5,
                                  motion_model=args.motion,
                                  use_motion=args.use_motion,
                                  tracking_classes=args.custom_classes,
                                  anchor_assignment_method='iou',
                                  track_keep_alive_thresh=args.track_thresh)
    logging.info('Tracker is defined')

    # get MOT results
    if args.input_type == 'video':
        # args.filename is the path to the video file
        results = list(tracker.process_frame_sequence(decoder_iter(args.filename), tracktor))
    elif args.input_type == 'images':
        # args.filename is the path to the folder containing image sequences
        results = list(tracker.process_frame_sequence(imageloader_iter(args.filename), tracktor))
    else:
        logging.info('We only support input in video or image sequences format.')

    logging.info('Tracking is done')

    # NOTE: save results for MOT17 evaluation.
    if args.eval:
        out_npy_list = []
        for i in range(len(results)):
            frame_id, trackInfo = results[i]
            for bb in trackInfo:
                track_id, x, y, w, h = bb["track_id"], bb["bbox"]["left"], bb["bbox"]["top"], bb["bbox"]["width"], \
                                       bb["bbox"]["height"]
                if int(bb["class_id"]) == 1:
                    ## only save the body box
                    out_npy_list.append([frame_id, track_id, x, y, w, h, -1, -1, -1, -1])
        out_npy = np.array(out_npy_list)
        track_results_path = os.path.join(args.save_path, args.save_filename)
        np.save(track_results_path, out_npy)

    # dump tracking results for visualization
    if not args.vis_off:
        if args.input_type == 'video':
            # args.filename is the path to the video file
            track_viz_video(results, args.filename, args.save_path)
        elif args.input_type == 'images':
            # args.filename is the path to the folder containing image sequences
            track_viz_image(results, args.filename, args.save_path)
        else:
            logging.info('We only support input in video or image sequences format.')

        logging.info('Tracking results are saved to %s' % (args.save_path))
