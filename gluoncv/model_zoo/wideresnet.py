"""
# Code adapted from:
# https://github.com/mapillary/inplace_abn/
#
# BSD 3-Clause License
#
# Copyright (c) 2017, mapillary
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
# pylint: disable=missing-docstring,arguments-differ,unused-argument

import sys
from collections import OrderedDict
from functools import partial
from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock

def bnrelu(channels, norm_layer=nn.BatchNorm, norm_kwargs=None, **kwargs):
    """
    Single Layer BN and Relu
    """
    out = nn.HybridSequential(prefix='')
    out.add(norm_layer(in_channels=channels, **({} if norm_kwargs is None else norm_kwargs)))
    out.add(nn.Activation('relu'))
    return out

class IdentityResidualBlock(HybridBlock):
    """
    Identity Residual Block for WideResnet
    """
    def __init__(self,
                 in_channels,
                 channels,
                 strides=1,
                 dilation=1,
                 groups=1,
                 norm_act=bnrelu,
                 dropout=None,
                 dist_bn=False):
        """Configurable identity-mapping residual block

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        channels : list of int
            Number of channels in the internal feature maps.
            Can either have two or three elements: if three construct
            a residual block with two `3 x 3` convolutions,
            otherwise construct a bottleneck block with `1 x 1`, then
            `3 x 3` then `1 x 1` convolutions.
        stride : int
            Stride of the first `3 x 3` convolution
        dilation : int
            Dilation to apply to the `3 x 3` convolutions.
        groups : int
            Number of convolution groups.
            This is used to create ResNeXt-style blocks and is only compatible with
            bottleneck blocks.
        norm_act : callable
            Function to create normalization / activation Module.
        dropout: callable
            Function to create Dropout Module.
        dist_bn: Boolean
            A variable to enable or disable use of distributed BN
        """
        super(IdentityResidualBlock, self).__init__()
        self.dist_bn = dist_bn

        # Check parameters for inconsistencies
        if len(channels) != 2 and len(channels) != 3:
            raise ValueError("channels must contain either two or three values")
        if len(channels) == 2 and groups != 1:
            raise ValueError("groups > 1 are only valid if len(channels) == 3")

        is_bottleneck = len(channels) == 3
        need_proj_conv = strides != 1 or in_channels != channels[-1]

        self.bn1 = norm_act(in_channels)
        if not is_bottleneck:
            layers = [
                ("conv1", nn.Conv2D(in_channels=in_channels,
                                    channels=channels[0],
                                    kernel_size=3,
                                    strides=strides,
                                    padding=dilation,
                                    use_bias=False,
                                    dilation=dilation)),
                ("bn2", norm_act(channels[0])),
                ("conv2", nn.Conv2D(in_channels=channels[0],
                                    channels=channels[1],
                                    kernel_size=3,
                                    strides=1,
                                    padding=dilation,
                                    use_bias=False,
                                    dilation=dilation))
            ]
            if dropout is not None:
                layers = layers[0:2] + [("dropout", dropout())] + layers[2:]
        else:
            layers = [
                ("conv1",
                 nn.Conv2D(in_channels=in_channels,
                           channels=channels[0],
                           kernel_size=1,
                           strides=strides,
                           padding=0,
                           use_bias=False)),
                ("bn2", norm_act(channels[0])),
                ("conv2", nn.Conv2D(in_channels=channels[0],
                                    channels=channels[1],
                                    kernel_size=3,
                                    strides=1,
                                    padding=dilation,
                                    use_bias=False,
                                    groups=groups,
                                    dilation=dilation)),
                ("bn3", norm_act(channels[1])),
                ("conv3", nn.Conv2D(in_channels=channels[1],
                                    channels=channels[2],
                                    kernel_size=1,
                                    strides=1,
                                    padding=0,
                                    use_bias=False))
            ]
            if dropout is not None:
                layers = layers[0:4] + [("dropout", dropout())] + layers[4:]

        layer_dict = OrderedDict(layers)
        self.convs = nn.HybridSequential(prefix='')
        for key in layer_dict.keys():
            self.convs.add(layer_dict[key])

        if need_proj_conv:
            self.proj_conv = nn.Conv2D(in_channels=in_channels,
                                       channels=channels[-1],
                                       kernel_size=1,
                                       strides=strides,
                                       padding=0,
                                       use_bias=False)

    def hybrid_forward(self, F, x):
        """
        This is the standard forward function for non-distributed batch norm
        """
        if hasattr(self, "proj_conv"):
            bn1 = self.bn1(x)
            shortcut = self.proj_conv(bn1)
        else:
            shortcut = x
            bn1 = self.bn1(x)

        out = self.convs(bn1)
        out = out + shortcut
        return out

class WiderResNetA2(HybridBlock):
    """
    Wider ResNet with pre-activation (identity mapping) blocks

    This variant uses down-sampling by max-pooling in the first two blocks and
     by strided convolution in the others.

    Parameters
    ----------
    structure : list of int
        Number of residual blocks in each of the six modules of the network.
    norm_act : callable
        Function to create normalization / activation Module.
    classes : int
        If not `0` also include global average pooling and a fully-connected layer
        with `classes` outputs at the end
        of the network.
    dilation : bool
        If `True` apply dilation to the last three modules and change the
        down-sampling factor from 32 to 8.
    """
    def __init__(self,
                 structure,
                 norm_act=bnrelu,
                 classes=0,
                 dilation=False,
                 dist_bn=False):
        super(WiderResNetA2, self).__init__()
        self.dist_bn = dist_bn

        norm_act = bnrelu
        self.structure = structure
        self.dilation = dilation

        if len(structure) != 6:
            raise ValueError("Expected a structure with six values")

        self.mod1 = nn.HybridSequential(prefix='mod1')
        self.mod1.add(nn.Conv2D(in_channels=3, channels=64,
                                kernel_size=3, strides=1, padding=1, use_bias=False))

        # Groups of residual blocks
        in_channels = 64
        channels = [(128, 128), (256, 256), (512, 512), (512, 1024), (512, 1024, 2048),
                    (1024, 2048, 4096)]
        for mod_id, num in enumerate(structure):
            # Create blocks for module
            blocks = []
            for block_id in range(num):
                if not dilation:
                    dil = 1
                    strides = 2 if block_id == 0 and 2 <= mod_id <= 4 else 1
                else:
                    if mod_id == 3:
                        dil = 2
                    elif mod_id > 3:
                        dil = 4
                    else:
                        dil = 1
                    strides = 2 if block_id == 0 and mod_id == 2 else 1

                if mod_id == 4:
                    drop = partial(nn.Dropout, rate=0.3)
                elif mod_id == 5:
                    drop = partial(nn.Dropout, rate=0.5)
                else:
                    drop = None

                blocks.append((
                    "block%d" % (block_id + 1),
                    IdentityResidualBlock(in_channels=in_channels,
                                          channels=channels[mod_id],
                                          norm_act=norm_act,
                                          strides=strides,
                                          dilation=dil,
                                          dropout=drop,
                                          dist_bn=self.dist_bn)
                ))

                # Update channels and p_keep
                in_channels = channels[mod_id][-1]

            # Create module
            if mod_id == 0:
                self.pool2 = nn.MaxPool2D(pool_size=3, strides=2, padding=1)
                blocks_dict = OrderedDict(blocks)
                self.mod2 = nn.HybridSequential(prefix='mod2')
                for key in blocks_dict.keys():
                    self.mod2.add(blocks_dict[key])

            if mod_id == 1:
                self.pool3 = nn.MaxPool2D(pool_size=3, strides=2, padding=1)
                blocks_dict = OrderedDict(blocks)
                self.mod3 = nn.HybridSequential(prefix='mod3')
                for key in blocks_dict.keys():
                    self.mod3.add(blocks_dict[key])


            if mod_id == 2:
                blocks_dict = OrderedDict(blocks)
                self.mod4 = nn.HybridSequential(prefix='mod4')
                for key in blocks_dict.keys():
                    self.mod4.add(blocks_dict[key])

            if mod_id == 3:
                blocks_dict = OrderedDict(blocks)
                self.mod5 = nn.HybridSequential(prefix='mod5')
                for key in blocks_dict.keys():
                    self.mod5.add(blocks_dict[key])

            if mod_id == 4:
                blocks_dict = OrderedDict(blocks)
                self.mod6 = nn.HybridSequential(prefix='mod6')
                for key in blocks_dict.keys():
                    self.mod6.add(blocks_dict[key])

            if mod_id == 5:
                blocks_dict = OrderedDict(blocks)
                self.mod7 = nn.HybridSequential(prefix='mod7')
                for key in blocks_dict.keys():
                    self.mod7.add(blocks_dict[key])

        # Pooling and predictor
        self.bn_out = norm_act(in_channels)
        if classes != 0:
            self.classifier = nn.HybridSequential(prefix='classifier')
            self.classifier.add(nn.GlobalAvgPool2D())
            self.classifier.add(nn.Dense(in_units=in_channels, units=classes))

    def hybrid_forward(self, F, img):
        out = self.mod1(img)
        out = self.mod2(self.pool2(out))
        out = self.mod3(self.pool3(out))
        out = self.mod4(out)
        out = self.mod5(out)
        out = self.mod6(out)
        out = self.mod7(out)
        out = self.bn_out(out)

        if hasattr(self, "classifier"):
            return self.classifier(out)
        return out

_NETS = {
    "16": {"structure": [1, 1, 1, 1, 1, 1]},
    "20": {"structure": [1, 1, 1, 3, 1, 1]},
    "38": {"structure": [3, 3, 6, 3, 1, 1]},
}

__all__ = []
for name, params in _NETS.items():
    net_name = "wider_resnet" + name + "_a2"
    setattr(sys.modules[__name__], net_name, partial(WiderResNetA2, **params))
    __all__.append(net_name)
