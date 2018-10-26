"""Pose related transformation functions"""
from __future__ import absolute_import
from __future__ import division

import numpy as np

def flip_heatmap(heatmap, joint_pairs):
    assert heatmap.ndim == 4, "heatmap should have shape (batch_size, num_joints, height, width)"
    out = heatmap[:, :, :, ::-1]

    for pairs in joint_pairs:
        tmp = out[:, pair[0], :, :].copy()
        out[:, pair[0], :, :] = out[:, pair[1], :, :]
        out[:, pair[1], :, :] = tmp

    return out

def flip_joints_3d(joints_3d, width, joint_pairs):
    joints = joints_3d.copy()
    # flip horizontally
    joints[:, 0, 0] = width - joints[:, 0, 0] - 1
    # change left-right parts
    for pair in joint_pairs:
        joints[pair[0], :, 0], joints[pair[1], :, 0] = joints[pair[1], :]
