"""Decoder module of Monodepth2
Code partially borrowed from
https://github.com/nianticlabs/monodepth2/blob/master/networks/depth_decoder.py
"""
# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

from collections import OrderedDict
import numpy as np
import mxnet as mx
import mxnet.gluon.nn as nn

from .layers import ConvBlock, Conv3x3


class DepthDecoder(nn.HybridBlock):
    r"""Decoder of Monodepth2

    Parameters
    ----------
    num_ch_enc : list
        The channels number of encoder.
    scales: list
        The scales used in the loss. (Default: range(4))
    num_output_channels: int
        The number of output channels. (Default: 1)
    use_skips: bool
        This will use skip architecture in the network. (Default: True)

    """
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1,
                 use_skips=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        with self.name_scope():
            self.convs = OrderedDict()
            for i in range(4, -1, -1):
                # upconv_0
                num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
                num_ch_out = self.num_ch_dec[i]
                self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

                # upconv_1
                num_ch_in = self.num_ch_dec[i]
                if self.use_skips and i > 0:
                    num_ch_in += self.num_ch_enc[i - 1]
                num_ch_out = self.num_ch_dec[i]
                self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

            for s in self.scales:
                self.convs[("dispconv", s)] = Conv3x3(
                    self.num_ch_dec[s], self.num_output_channels)

            # register blocks
            for k in self.convs:
                self.register_child(self.convs[k])
            self.decoder = nn.HybridSequential()
            self.decoder.add(*list(self.convs.values()))

            self.sigmoid = nn.Activation('sigmoid')

    def hybrid_forward(self, F, input_features):
        # pylint: disable=unused-argument, missing-function-docstring
        self.outputs = []

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [F.UpSampling(x, scale=2, sample_type='nearest')]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = F.concat(*x, dim=1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs.append(self.sigmoid(self.convs[("dispconv", i)](x)))

        return self.outputs

    def predict(self, input_features):
        # pylint: disable=unused-argument, missing-function-docstring
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):
            x = self.convs[("upconv", i, 0)].predict(x)
            x = [mx.nd.UpSampling(x, scale=2, sample_type='nearest')]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = mx.nd.concat(*x, dim=1)
            x = self.convs[("upconv", i, 1)].predict(x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)].predict(x))

        return self.outputs
