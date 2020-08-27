# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import time
from tqdm import tqdm
import cv2
import numpy as np

import mxnet as mx
from mxnet import gluon
from gluoncv.data import KITTIRAWDataset
from gluoncv.data.kitti.kitti_utils import dict_batchify_fn, readlines
from gluoncv.model_zoo import get_model

from gluoncv.model_zoo.monodepthv2.layers import disp_to_depth, compute_depth_errors
from options import MonodepthOptions

cv2.setNumThreads(0)  # This speeds up evaluation 5x on our unix systems (OpenCV 3.3.1)

splits_dir = os.path.join(os.path.expanduser("~"), ".mxnet/datasets/kitti", "splits")

# Models which were trained with stereo supervision were trained with a nominal
# baseline of 0.1 units. The KITTI rig has a baseline of 54cm. Therefore,
# to convert our stereo predictions to real-world scale we multiply our depths by 5.4.
STEREO_SCALE_FACTOR = 5.4


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

    assert sum((opt.eval_mono, opt.eval_stereo)) == 1, \
        "Please choose mono or stereo evaluation by setting either --eval_mono or --eval_stereo"

    if opt.ext_disp_to_eval is None:
        ############################ loading dataset ############################
        filenames = readlines(os.path.join(splits_dir, opt.eval_split, "test_files.txt"))

        img_ext = '.png' if opt.png else '.jpg'
        dataset = KITTIRAWDataset(opt.data_path, filenames, opt.height, opt.width,
                                  [0], 4, is_train=False, img_ext=img_ext)
        dataloader = gluon.data.DataLoader(
            dataset, batch_size=opt.batch_size, shuffle=False,
            batchify_fn=dict_batchify_fn, num_workers=opt.num_workers,
            pin_memory=True, last_batch='keep')

        ############################ loading model ############################
        model = None
        # create network
        if opt.model_zoo is not None:
            if opt.pretrained_type == "gluoncv":
                # use gluoncv pretrained model
                model = get_model(opt.model_zoo, pretrained_base=False, ctx=opt.ctx,
                                  pretrained=True)
            else:
                # loading weights from customer
                assert opt.eval_model is not None, \
                    '=> Please provide the checkpoint using --eval_model'
                weights_path = os.path.join(opt.load_weights_folder, opt.eval_model)
                model = get_model(opt.model_zoo, pretrained_base=False, ctx=opt.ctx)
                model.load_parameters(weights_path, ctx=opt.ctx)
        else:
            assert "Must choose a depth model from model_zoo, " \
                   "please provide the model_zoo using --model_zoo"

        # use hybridize mode
        if opt.hybridize:
            model.hybridize()

        ############################ inference ############################
        pred_disps = []
        tbar = tqdm(dataloader)
        t_gpu = 0
        for i, data in enumerate(tbar):
            input_color = data[("color", 0, 0)]
            input_color = input_color.as_in_context(context=opt.ctx[0])

            tic = time.time()
            if opt.hybridize:
                decoder_output = model(input_color)

                # for hybridize mode, the output of HybridBlock must is NDArray or List.
                # Here, we have to transfer the output to dict type.
                outputs = {}
                idx = 0
                for scale in range(4, -1, -1):
                    if scale in opt.scales:
                        outputs[("disp", scale)] = decoder_output[idx]
                        idx += 1
            else:
                outputs = model.predict(input_color)
            t_gpu += time.time() - tic
            pred_disp, _ = disp_to_depth(outputs[("disp", 0)], opt.min_depth, opt.max_depth)
            pred_disp = pred_disp.as_in_context(mx.cpu())[:, 0].asnumpy()

            pred_disps.append(pred_disp)

        pred_disps = np.concatenate(pred_disps)
        gpu_time = t_gpu / len(dataset)
        print("\nAverage inference time {:0.3f}ms, {:0.3f}fps\n".format(gpu_time * 1000, 1 / gpu_time))

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
    gt_depths = np.load(gt_path, allow_pickle=True, fix_imports=True, encoding='latin1')["data"]

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
                             0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
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

        errors.append(compute_depth_errors(gt_depth, pred_depth))

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format(
        "abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions()
    opts = options.parse()
    print("Testing model named:\n  ", opts.model_zoo)
    print("Weights are loaded from:\n  ",
          "gluoncv pretrained model" if opts.pretrained_type == "gluoncv"
          else opts.load_weights_folder)
    print("Inference is using:\n  ", "CPU" if opts.ctx[0] is mx.cpu() else "GPU")

    evaluate(opts)
