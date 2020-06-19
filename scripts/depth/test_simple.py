from __future__ import absolute_import, division, print_function

import os
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm

import mxnet as mx
from mxnet.gluon.data.vision import transforms
from gluoncv.model_zoo import monodepthv2

from gluoncv.model_zoo.monodepthv2.layers import disp_to_depth
from utils import download_model_if_doesnt_exist


def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', required=True)
    parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use',
                        choices=["mono+stereo_640x192_mx"])
    parser.add_argument("--height", type=int,
                        help="input image height", default=192)
    parser.add_argument("--width", type=int,
                        help="input image width", default=640)
    parser.add_argument("--use_stereo",
                        help="if set, uses stereo pair for training", action="store_true")
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')

    args = parser.parse_args()
    args.ctx = mx.gpu(0)

    if args.no_cuda:
        args.ctx = mx.cpu()

    return args


def test_simple(args):
    ############################# loading model ############################
    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    # download_model_if_doesnt_exist(args.model_name)
    model_path = os.path.join("models", args.model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.params")
    depth_decoder_path = os.path.join(model_path, "depth.params")

    ############################ loading pretained model ############################
    print("   Loading pretrained encoder")
    encoder = monodepthv2.ResnetEncoder(18, False, ctx=args.ctx)
    encoder.load_parameters(encoder_path, ctx=args.ctx)

    feed_height = args.height
    feed_width = args.width

    print("   Loading pretrained decoder")
    depth_decoder = monodepthv2.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))
    depth_decoder.load_parameters(depth_decoder_path, ctx=args.ctx)

    ############################ finding input images ############################
    if os.path.isfile(args.image_path):
        # Only testing on a single image
        paths = [args.image_path]
        output_directory = os.path.dirname(args.image_path)
    elif os.path.isdir(args.image_path):
        # Searching folder for images
        paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
        output_directory = args.image_path
    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))

    print("-> Predicting on {:d} test images".format(len(paths)))

    ############################ Prediction ############################
    for idx, image_path in enumerate(paths):

        if image_path.endswith("_disp.jpg"):
            # don't try to predict disparity for a disparity image!
            continue

        ########## Load image and preprocess ##########
        input_image = pil.open(image_path).convert('RGB')
        original_width, original_height = input_image.size
        input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
        input_image = transforms.ToTensor()(mx.nd.array(input_image)).expand_dims(0)

        ########## PREDICTION ##########
        input_image = input_image.as_in_context(context=args.ctx)
        features = encoder(input_image)
        outputs = depth_decoder(features)

        disp = outputs[("disp", 0)]
        disp_resized = mx.nd.contrib.BilinearResize2D(
            disp, height=original_height, width=original_width)

        ########## Saving numpy file ##########
        output_name = os.path.splitext(os.path.basename(image_path))[0]
        name_dest_npy = os.path.join(output_directory, "{}_disp_mx.npy".format(output_name))
        scaled_disp, _ = disp_to_depth(disp, 0.1, 100)  # up-to-scale

        np.save(name_dest_npy, scaled_disp.as_in_context(mx.cpu()).asnumpy())

        ########## Saving colormapped depth image ##########
        disp_resized_np = disp_resized.squeeze().as_in_context(mx.cpu()).asnumpy()
        vmax = np.percentile(disp_resized_np, 95)
        normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
        im = pil.fromarray(colormapped_im)

        name_dest_im = os.path.join(output_directory, "{}_disp_mx.jpeg".format(output_name))
        im.save(name_dest_im)

        print("   Processed {:d} of {:d} images - saved prediction to {}".format(
            idx + 1, len(paths), name_dest_im))


if __name__ == '__main__':
    args = parse_args()
    test_simple(args)