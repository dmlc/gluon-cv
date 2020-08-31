"""Decoder module of Monodepth2
Code partially borrowed from
https://github.com/nianticlabs/monodepth2/blob/master/networks/pose_decoder.py
"""
# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from collections import OrderedDict
import mxnet as mx
import mxnet.gluon.nn as nn


class PoseDecoder(nn.HybridBlock):
    r"""Decoder of Monodepth2 PoseNet

    Parameters
    ----------
    num_ch_enc : list
        The channels number of encoder.
    num_input_features: int
        The number of input sequences. 1 for depth encoder, larger than 1 for pose encoder.
        (Default: 2)
    num_frames_to_predict_for: int
        The number of output pose between frames; If None, it equals num_input_features - 1.
        (Default: 2)
    stride: int
        The stride number for Conv in pose decoder. (Default: 1)

    """
    def __init__(self, num_ch_enc, num_input_features, num_frames_to_predict_for=2, stride=1):
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2D(
            in_channels=self.num_ch_enc[-1], channels=256, kernel_size=1)
        self.convs[("pose", 0)] = nn.Conv2D(
            in_channels=num_input_features * 256, channels=256,
            kernel_size=3, strides=stride, padding=1)
        self.convs[("pose", 1)] = nn.Conv2D(
            in_channels=256, channels=256, kernel_size=3, strides=stride, padding=1)
        self.convs[("pose", 2)] = nn.Conv2D(
            in_channels=256, channels=6 * num_frames_to_predict_for, kernel_size=1)

        # register blocks
        for k in self.convs:
            self.register_child(self.convs[k])
        self.net = nn.HybridSequential()
        self.net.add(*list(self.convs.values()))

    def hybrid_forward(self, F, input_features):
        # pylint: disable=unused-argument, missing-function-docstring
        last_features = [f[-1] for f in input_features]

        cat_features = [F.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = F.concat(*cat_features, dim=1)

        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = F.relu(out)

        out = out.mean(3).mean(2)

        out = 0.01 * out.reshape(-1, self.num_frames_to_predict_for, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation

    def predict(self, input_features):
        # pylint: disable=unused-argument, missing-function-docstring
        last_features = [f[-1] for f in input_features]

        cat_features = [mx.nd.relu()(self.convs["squeeze"](f)) for f in last_features]
        cat_features = mx.nd.concat(*cat_features, dim=1)

        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = mx.nd.relu(out)

        out = out.mean(3).mean(2)

        out = 0.01 * out.reshape(-1, self.num_frames_to_predict_for, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation
