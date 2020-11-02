"""04. Testing PoseNet from image sequences with pre-trained Monodepth2 Pose models
===========================================================================

This is a quick demo of using the GluonCV Monodepth2 model for KITTI on real-world images.
Please follow the `installation guide <../../index.html#installation>`__
to install MXNet and GluonCV if not yet. This tutorial is divided into three parts:
prepare KITTI Odometry datasets, get pre-trained PoseNet through GluonCV, and testing PoseNet on the dataset.

Start Testing Now
~~~~~~~~~~~~~~~~~~

.. hint::

    Feel free to skip the tutorial because the testing script is self-complete and ready to launch.

    :download:`Download Full Python Script: test_pose.py<../../../scripts/depth/test_pose.py>`

    Example testing PoseNet command::

        python test_pose.py --model_zoo_pose monodepth2_resnet18_posenet_kitti_mono_stereo_640x192 --data_path ~/.mxnet/datasets/kitti/kitti_odom --eval_split odom_9 --pretrained_type gluoncv --png

    For more training command options, please run ``python test_pose.py -h``
    Please checkout the `model_zoo <../../model_zoo/depth.html>`_ for training commands of reproducing the pretrained model.

Dive into Deep
~~~~~~~~~~~~~~
"""

import os
import mxnet as mx
from mxnet import gluon
import gluoncv
# using cpu
ctx = mx.cpu(0)

##############################################################################
# KITTI Odometry Dataset
# -----------------------------
#
# - Prepare KITTI Odometry Dataset:
#
#     You can download KITTI Odometry Dataset from http://www.cvlibs.net/datasets/kitti/eval_odometry.php.
#
#     You need to download http://www.cvlibs.net/download.php?file=data_odometry_color.zip and
#     http://www.cvlibs.net/download.php?file=data_odometry_poses.zip (you will get the
#     download link via emails.) to ``$(HOME)/.mxnet/datasets/kitti/``, then extract them. Here is an example commands::
#
#       cd ~/.mxnet/datasets/kitti/
#       wget [the link of data_odometry_color.zip file]
#       wget [the link of data_odometry_poses.zip file]
#       unzip data_odometry_color.zip
#       unzip data_odometry_poses.zip
#       mv dataset/ kitti_odom/
#
#
# KITTI dataset is provided in :class:`gluoncv.data`.
# For example, we can easily get the KITTI Odometry dataset (we suppose you have prepared a split file as described in
# `Dive Deep into Training Monodepth2 Models <./train_monodepth2.html>`_.)

from gluoncv.data.kitti import readlines, dict_batchify_fn

splits_dir = os.path.join(os.path.expanduser("~"), ".mxnet/datasets/kitti", "splits")

eval_split = "odom_9"

sequence_id = int(eval_split.split("_")[1])

data_path = os.path.join(
    os.path.expanduser("~"), '.mxnet/datasets/kitti/kitti_odom')
filenames = readlines(
    os.path.join(splits_dir, "odom",
                 "test_files_{:02d}.txt".format(sequence_id)))

dataset = gluoncv.data.KITTIOdomDataset(
    data_path=data_path, filenames=filenames, height=192, width=640, frame_idxs=[0, 1],
    num_scales=4, is_train=False, img_ext=".png")
dataloader = gluon.data.DataLoader(
    dataset, batch_size=1, shuffle=False, batchify_fn=dict_batchify_fn, num_workers=0,
    pin_memory=True, last_batch='keep')

##############################################################################
# Here, ``frame_idxs`` argument is [0, 1]. It means that the dataloader provide the current frame and the next frame
# as input. The PoseNet need to predict the relative pose between two frames.
#
# Please check out the full :download:`test_pose.py<../../../scripts/depth/test_pose.py>` for complete implementation.


##############################################################################
# Pre-trained PoseNet
# -----------------------------
# Next, we get a pre-trained model from our model zoo,
from gluoncv.model_zoo import get_model

model_zoo = 'monodepth2_resnet18_posenet_kitti_mono_stereo_640x192'

posenet = get_model(model_zoo, pretrained_base=False, num_input_images=2,
                    num_input_features=1, num_frames_to_predict_for=2, pretrained=True, ctx=ctx)

##############################################################################
# Testing
# -----------------------------
#
# - Inference of PoseNet:
#
#     Firstly, we need to generate transformation between two frames via PoseNet. PoseNet will output axisangle and
#     translation directly, and we transfer it to the transformation matrix. Then, we store predicted pose to pose
#     sequence.
#
# Please check out the full :download:`test_pose.py<../../../scripts/depth/test_pose.py>` for complete implementation.
# This is an example of test::
#
#       pred_poses = []
#       print("-> Computing pose predictions")
#
#       opt.frame_ids = [0, 1]  # pose network only takes two frames as input
#       tbar = tqdm(dataloader)
#       for i, data in enumerate(tbar):
#           for key, ipt in data.items():
#               data[key] = ipt.as_in_context(context=opt.ctx[0])
#
#           all_color_aug = mx.nd.concat(*[data[("color_aug", i, 0)] for i in opt.frame_ids], dim=1)
#           axisangle, translation = posenet(all_color_aug)
#
#           pred_poses.append(
#               transformation_from_parameters(
#                   axisangle[:, 0], translation[:, 0]).as_in_context(mx.cpu()).asnumpy()
#           )
#
#       pred_poses = np.concatenate(pred_poses)
#
#
# - Calculate ATE:
#
#     Here, we use absolute trajectory error (ATE) to evaluate the performance of PoseNet. Because the PoseNet
#     also belongs to the monocular system, it have a scale ambiguity problem. For evaluating the model with
#     ground truth pose, we align the first pose with ground truth and calculating the scaling factor by comparing
#     predicted pose and ground truth pose.
#
# The ATE function is defined as::
#
#       def compute_ate(gtruth_xyz, pred_xyz_o):
#           offset = gtruth_xyz[0] - pred_xyz_o[0]
#           pred_xyz = pred_xyz_o + offset[None, :]
#
#           # Optimize the scaling factor
#           scale = np.sum(gtruth_xyz * pred_xyz) / np.sum(pred_xyz ** 2)
#           alignment_error = pred_xyz * scale - gtruth_xyz
#           rmse = np.sqrt(np.sum(alignment_error ** 2)) / gtruth_xyz.shape[0]
#           return rmse


##############################################################################
# You can `Start Testing Now`_.
#
# References
# ----------
# .. [Godard19] Clement Godard, Oisin Mac Aodha, Michael Firman and Gabriel Brostow. \
#       "Digging Into Self-Supervised Monocular Depth Estimation." \
#       Proceedings of the IEEE conference on computer vision (ICCV). 2019.
#

