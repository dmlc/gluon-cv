import os
import argparse
import time
import PIL.Image as pil
import numpy as np

import mxnet as mx
from mxnet.gluon.data.vision import transforms

import gluoncv
from gluoncv.model_zoo.monodepthv2.layers import disp_to_depth

import matplotlib as mpl
import matplotlib.cm as cm
import cv2


# using cpu
ctx = mx.cpu(0)


def parse_args():
    """Training Options for Depth Prediction Experiments"""
    parser = argparse.ArgumentParser(description='MXNet Gluon Monodepth2 Demo')

    # model and dataset
    parser.add_argument('--model_zoo', type=str,
                        choices=['monodepth2_resnet18_kitti_stereo_640x192',
                                 'monodepth2_resnet18_kitti_mono_640x192',
                                 'monodepth2_resnet18_kitti_mono_stereo_640x192'],
                        default='monodepth2_resnet18_kitti_mono_stereo_640x192',
                        help='choose depth model from model zoo model')

    parser.add_argument('--input_format', type=str,
                        choices=['image', 'video'], default='image',
                        help='choose the format of input data')
    parser.add_argument("--data_path", type=str, help="path to the data")
    parser.add_argument("--height", type=int, help="input image height", default=192)
    parser.add_argument("--width", type=int, help="input image width", default=640)

    parser.add_argument('--prediction_only', action="store_true",
                        help='if true, just store pure prediction results')
    parser.add_argument('--use_depth', action="store_true",
                        help='use depth map as prediction results')
    parser.add_argument('--output_format', type=str,
                        choices=['image', 'video'], default='video',
                        help='choose the format of output')
    parser.add_argument("--output_path", type=str, help="path to store the results",
                        default=os.path.join(os.path.expanduser("."), "tmp"))

    # the parser
    args = parser.parse_args()

    return args


def read_img(files, data_path):
    raw_img_sequences = []
    for file in files:
        file = os.path.join(data_path, file)
        img = pil.open(file).convert('RGB')
        raw_img_sequences.append(img)

    original_width, original_height = raw_img_sequences[0].size

    return raw_img_sequences, original_width, original_height


def read_video(data_path):
    raw_img_sequences = []
    files = []
    frame_index = 0

    cap = cv2.VideoCapture(data_path)
    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = pil.fromarray(img)
        raw_img_sequences.append(img)

        f_str = "{:010d}.png".format(frame_index)
        files.append(f_str)
        frame_index += 1
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    original_width, original_height = raw_img_sequences[0].size

    return raw_img_sequences, files, original_width, original_height


if __name__ == '__main__':
    args = parse_args()

    ############################ Loading Data ############################
    print("Loading Data......")
    tic = time.time()

    if args.input_format == 'image':
        assert os.path.isdir(args.data_path), \
            "--data_path must be a direction when input_format is 'image'"

        files = os.listdir(args.data_path)
        files.sort()
        raw_img_sequences, original_width, original_height = \
            read_img(files=files, data_path=args.data_path)
    elif args.input_format == 'video':
        assert os.path.isfile(args.data_path), \
            "--data_path must be a video file when input_format is 'video'"
        raw_img_sequences, files, original_width, original_height = \
            read_video(data_path=args.data_path)

    feed_height = args.height
    feed_width = args.width

    t_consuming = time.time() - tic
    print("Data loaded! Time consuming: {:0.3f}s\n".format(t_consuming))

    ############################ Prepare Models and Prediction ############################
    print("Loading Model and Prediction......")
    tic = time.time()

    # while use stereo or mono+stereo model, we could get real depth value
    min_depth = 0.1
    max_depth = 100

    scale_factor = 5.4
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    model = gluoncv.model_zoo.get_model(args.model_zoo,
                                        pretrained_base=False, ctx=ctx, pretrained=True)
    pred_sequences = []
    for img in raw_img_sequences:
        img = img.resize((feed_width, feed_height), pil.LANCZOS)
        img = transforms.ToTensor()(mx.nd.array(img)).expand_dims(0).as_in_context(context=ctx)

        outputs = model.predict(img)
        mx.nd.waitall()
        pred_disp, _ = disp_to_depth(outputs[("disp", 0)], min_depth, max_depth)
        t = time.time()
        pred_disp = pred_disp.squeeze().as_in_context(mx.cpu()).asnumpy()

        pred_disp = cv2.resize(src=pred_disp, dsize=(original_width, original_height))
        pred_depth = 1 / pred_disp

        if args.model_zoo != 'monodepth2_resnet18_kitti_mono_640x192':
            pred_depth *= scale_factor
            pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
            pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        if args.use_depth:
            pred_sequences.append(pred_depth)
        else:
            pred_sequences.append(pred_disp)

    t_consuming = time.time() - tic
    print("Finished prediction! Time consuming: {:0.3f}s\n".format(t_consuming))

    ############################ Visualization & Store Videos ############################
    print("Visualization and Store Results......")
    tic = time.time()

    if args.prediction_only:
        pred_path = os.path.join(args.output_path, 'pred')
        if not os.path.exists(pred_path):
            os.makedirs(pred_path)

        for pred, file in zip(pred_sequences, files):
            pred_out_file = os.path.join(pred_path, file)
            cv2.imwrite(pred_out_file, pred)
    else:
        rgb_path = os.path.join(args.output_path, 'rgb')
        if not os.path.exists(rgb_path):
            os.makedirs(rgb_path)

        output_sequences = []
        for raw_img, pred, file in zip(raw_img_sequences, pred_sequences, files):
            vmax = np.percentile(pred, 95)
            normalizer = mpl.colors.Normalize(vmin=pred.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(pred)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)

            raw_img = np.array(raw_img)
            pred = np.array(im)
            output = np.concatenate((raw_img, pred), axis=0)
            output_sequences.append(output)

            if args.output_format == 'image':
                pred_out_file = os.path.join(rgb_path, file)
                cv2.imwrite(pred_out_file, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))

        if args.output_format == 'video':
            width = int(output_sequences[0].shape[1] + 0.5)
            height = int(output_sequences[0].shape[0] + 0.5)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                os.path.join(args.output_path, 'demo.mp4'), fourcc, 20.0, (width, height))

            for frame in output_sequences:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                out.write(frame)
                cv2.imshow('demo', frame)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            out.release()
            cv2.destroyAllWindows()

    t_consuming = time.time() - tic
    print("Finished! Time consuming: {:0.3f}s".format(t_consuming))
