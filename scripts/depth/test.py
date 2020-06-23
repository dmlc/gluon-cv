# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
from tqdm import tqdm
import cv2
import numpy as np
import time

import mxnet as mx
from mxnet import gluon
from utils import readlines
from gluoncv.data.kitti import kitti_dataset
from gluoncv.data.kitti.kitti_utils import dict_batchify_fn
from gluoncv.model_zoo import monodepthv2
from gluoncv.utils.parallel import *

from gluoncv.model_zoo.monodepthv2.layers import disp_to_depth
from options import MonodepthOptions


cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)


splits_dir = os.path.join(os.path.dirname(__file__), "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1
    """
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
    MIN_DEPTH = 1e-3
    MAX_DEPTH = 80

    # DO NOT modify!!! Only support batch_size=ngus
    batch_size = 16  # opt.ngpus

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.ext_disp_to_eval is None:
        ############################ loading weights ############################
        encoder_path = os.path.join(opt.load_weights_folder, "encoder.params")
        decoder_path = os.path.join(opt.load_weights_folder, "depth.params")

        ############################ loading dataset ############################
        tic = time.time()
        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))

        img_ext = '.png' if opt.png else '.jpg'
        dataset = kitti_dataset.KITTIRAWDataset(opt.data_path, filenames,
                                                opt.height, opt.width,
                                                [0], 4, is_train=False, img_ext=img_ext)
        dataloader = gluon.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                           batchify_fn=dict_batchify_fn,
                                           num_workers=opt.num_workers,
                                           pin_memory=True, last_batch='rollover')
        print('Runtime of create dataloader : %.2f' % (time.time() - tic))

        ############################ loading model ############################
        tic = time.time()
        encoder = monodepthv2.ResnetEncoder(opt.num_layers, pretrained=False, ctx=opt.ctx)
        depth_decoder = monodepthv2.DepthDecoder(encoder.num_ch_enc)

        # encoder.load_parameters(encoder_path, ctx=opt.ctx[0])
        # depth_decoder.load_parameters(decoder_path, ctx=opt.ctx[0])
        encoder.load_parameters(encoder_path, ctx=opt.ctx)
        depth_decoder.load_parameters(decoder_path, ctx=opt.ctx)

        # encoder.initialize(ctx=opt.ctx[0])
        # depth_decoder.initialize(ctx=opt.ctx[0])

        encoder_ = DataParallelModel(encoder, ctx_list=opt.ctx)
        depth_decoder_ = DataParallelModel(depth_decoder, ctx_list=opt.ctx)
        print('Runtime of create model : %.2f' % (time.time() - tic))

        ############################ inference ############################
        pred_disps = []
        tbar = tqdm(dataloader)
        for i, data in enumerate(tbar):
            # input_color = data[("color", 0, 0)]
            # input_color = input_color.as_in_context(context=opt.ctx[0])
            # features = encoder(input_color)
            # output = depth_decoder(features)

            input_color = data[("color", 0, 0)]
            features = encoder_(input_color)
            encoder_outputs = [x for x in features[0]]
            for i in range(1, len(features)):
                for j in range(len(features[i])):
                    encoder_outputs[j] = mx.nd.concat(
                        encoder_outputs[j],
                        features[i][j].as_in_context(encoder_outputs[j].context),
                        dim=0
                    )

            output = depth_decoder_(encoder_outputs)
            decoder_output = output[0]
            for i in range(1, len(output)):
                for key in decoder_output.keys():
                    decoder_output[key] = mx.nd.concat(
                        decoder_output[key],
                        output[i][key].as_in_context(decoder_output[key].context),
                        dim=0
                    )

            pred_disp, _ = disp_to_depth(decoder_output[("disp", 0)], opt.min_depth, opt.max_depth)
            pred_disp = pred_disp.as_in_context(mx.cpu())[:, 0].asnumpy()

            pred_disps.append(pred_disp)

        pred_disps = np.concatenate(pred_disps)

    else:
        # Load predictions from file
        print("-> Loading predictions from {}".format(opt.ext_disp_to_eval))
        pred_disps = np.load(opt.ext_disp_to_eval)

        if opt.eval_eigen_to_benchmark:
            eigen_to_benchmark_ids = np.load(
                os.path.join(splits_dir, "benchmark", "eigen_to_benchmark_ids.npy"))

            pred_disps = pred_disps[eigen_to_benchmark_ids]

    if opt.save_pred_disps:
        output_path = os.path.join(
            opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
        print("-> Saving predicted disparities to ", output_path)
        np.save(output_path, pred_disps)

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    elif opt.eval_split == 'benchmark':
        save_dir = os.path.join(opt.load_weights_folder, "benchmark_predictions")
        print("-> Saving out benchmark predictions to {}".format(save_dir))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx in range(len(pred_disps)):
            disp_resized = cv2.resize(pred_disps[idx], (1216, 352))
            depth = STEREO_SCALE_FACTOR / disp_resized
            depth = np.clip(depth, 0, 80)
            depth = np.uint16(depth * 256)
            save_path = os.path.join(save_dir, "{:010d}.png".format(idx))
            cv2.imwrite(save_path, depth)

        print("-> No ground truth is available for the KITTI benchmark, so not evaluating. Done.")
        quit()

    gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
    gt_depths = np.load(gt_path, allow_pickle=True,fix_imports=True, encoding='latin1')["data"]

    print("-> Evaluating")

    if opt.eval_stereo:
        print("   Stereo evaluation - "
              "disabling median scaling, scaling by {}".format(STEREO_SCALE_FACTOR))
        opt.disable_median_scaling = True
        opt.pred_depth_scale_factor = STEREO_SCALE_FACTOR
    else:
        print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []

    for i in range(pred_disps.shape[0]):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        if opt.eval_split == "eigen":
            mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)

            crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                             0.03594771 * gt_width,  0.96405229 * gt_width]).astype(np.int32)
            crop_mask = np.zeros(mask.shape)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
            mask = np.logical_and(mask, crop_mask)

        else:
            mask = gt_depth > 0

        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions()
    evaluate(options.parse())