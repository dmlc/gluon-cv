# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
from tqdm import tqdm
import numpy as np

import mxnet as mx
from mxnet import gluon
from gluoncv.data import KITTIOdomDataset
from gluoncv.data.kitti.kitti_utils import dict_batchify_fn, readlines
from gluoncv.model_zoo import get_model

from gluoncv.model_zoo.monodepthv2 import *
from gluoncv.model_zoo.monodepthv2.layers import transformation_from_parameters
from options import MonodepthOptions

splits_dir = os.path.join(os.path.expanduser("~"), ".mxnet/datasets/kitti", "splits")


# from https://github.com/tinghuiz/SfMLearner
def dump_xyz(source_to_target_transformations):
    xyzs = []
    cam_to_world = np.eye(4)
    xyzs.append(cam_to_world[:3, 3])
    for source_to_target_transformation in source_to_target_transformations:
        cam_to_world = np.dot(cam_to_world, source_to_target_transformation)
        xyzs.append(cam_to_world[:3, 3])
    return xyzs


# from https://github.com/tinghuiz/SfMLearner
def compute_ate(gtruth_xyz, pred_xyz_o):
    # Make sure that the first matched frames align (no need for rotational alignment as
    # all the predicted/ground-truth snippets have been converted to use the same coordinate
    # system with the first frame of the snippet being the origin).
    offset = gtruth_xyz[0] - pred_xyz_o[0]
    pred_xyz = pred_xyz_o + offset[None, :]

    # Optimize the scaling factor
    scale = np.sum(gtruth_xyz * pred_xyz) / np.sum(pred_xyz ** 2)
    alignment_error = pred_xyz * scale - gtruth_xyz
    rmse = np.sqrt(np.sum(alignment_error ** 2)) / gtruth_xyz.shape[0]
    return rmse


def evaluate(opt):
    """Evaluate odometry on the KITTI dataset
    """
    ############################ loading dataset ############################
    assert opt.eval_split == "odom_9" or opt.eval_split == "odom_10", \
        "eval_split should be either odom_9 or odom_10"

    sequence_id = int(opt.eval_split.split("_")[1])

    filenames = readlines(
        os.path.join(splits_dir, "odom",
                     "test_files_{:02d}.txt".format(sequence_id)))

    img_ext = '.png' if opt.png else '.jpg'

    dataset = KITTIOdomDataset(data_path=opt.data_path, filenames=filenames,
                               height=opt.height, width=opt.width, frame_idxs=[0, 1],
                               num_scales=4, is_train=False, img_ext=img_ext)
    dataloader = gluon.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=False,
        batchify_fn=dict_batchify_fn, num_workers=opt.num_workers,
        pin_memory=True, last_batch='keep')

    ############################ loading model ############################
    posenet = None
    # create network
    if opt.model_zoo_pose is not None:
        if opt.pretrained_type == "gluoncv":
            # use gluoncv pretrained model
            posenet = get_model(
                opt.model_zoo_pose, pretrained_base=False, num_input_images=2,
                num_input_features=1, num_frames_to_predict_for=2, pretrained=True, ctx=opt.ctx)
        else:
            # loading weights from customer
            assert opt.eval_model is not None, \
                '=> Please provide the checkpoint using --eval_model'
            assert os.path.isdir(opt.load_weights_folder), \
                "Cannot find a folder at {}".format(opt.load_weights_folder)

            weights_path = os.path.join(opt.load_weights_folder, opt.eval_model)
            posenet = get_model(
                opt.model_zoo_pose, pretrained_base=False, num_input_images=2,
                num_input_features=1, num_frames_to_predict_for=2, ctx=opt.ctx)
            posenet.load_parameters(weights_path, ctx=opt.ctx)
    else:
        assert "Must choose a pose model from model_zoo, " \
               "please provide the model_zoo using --model_zoo_pose"

    # use hybridize mode
    if opt.hybridize:
        posenet.hybridize()

    ############################ inference ############################
    pred_poses = []
    print("-> Computing pose predictions")

    opt.frame_ids = [0, 1]  # pose network only takes two frames as input
    tbar = tqdm(dataloader)
    for i, data in enumerate(tbar):
        for key, ipt in data.items():
            data[key] = ipt.as_in_context(context=opt.ctx[0])

        all_color_aug = mx.nd.concat(*[data[("color_aug", i, 0)] for i in opt.frame_ids], dim=1)
        axisangle, translation = posenet(all_color_aug)

        pred_poses.append(
            transformation_from_parameters(
                axisangle[:, 0], translation[:, 0]).as_in_context(mx.cpu()).asnumpy()
        )

    pred_poses = np.concatenate(pred_poses)

    ############################ evaluation ############################
    gt_poses_path = os.path.join(opt.data_path, "poses", "{:02d}.txt".format(sequence_id))
    gt_global_poses = np.loadtxt(gt_poses_path).reshape(-1, 3, 4)
    gt_global_poses = np.concatenate(
        (gt_global_poses, np.zeros((gt_global_poses.shape[0], 1, 4))), 1)
    gt_global_poses[:, 3, 3] = 1
    gt_xyzs = gt_global_poses[:, :3, 3]

    gt_local_poses = []
    for i in range(1, len(gt_global_poses)):
        gt_local_poses.append(
            np.linalg.inv(np.dot(np.linalg.inv(gt_global_poses[i - 1]), gt_global_poses[i])))

    ates = []
    num_frames = gt_xyzs.shape[0]
    track_length = 5
    for i in range(0, num_frames - 1):
        local_xyzs = np.array(dump_xyz(pred_poses[i:i + track_length - 1]))
        gt_local_xyzs = np.array(dump_xyz(gt_local_poses[i:i + track_length - 1]))
        ates.append(compute_ate(gt_local_xyzs, local_xyzs))

    print("\n   Trajectory error: {:0.3f}, std: {:0.3f}\n".format(np.mean(ates), np.std(ates)))
    print("\n-> Done!")


if __name__ == "__main__":
    options = MonodepthOptions()
    opts = options.parse()
    print("Testing model named:\n  ", opts.model_zoo_pose)
    print("Weights are loaded from:\n  ",
          "gluoncv pretrained model" if opts.pretrained_type == "gluoncv"
          else opts.load_weights_folder)
    print("Inference is using:\n  ", "CPU" if opts.ctx[0] is mx.cpu() else "GPU")

    evaluate(opts)
