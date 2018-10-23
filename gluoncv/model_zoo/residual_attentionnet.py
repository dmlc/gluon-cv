# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding: utf-8
# pylint: disable= unused-argument,missing-docstring
"""ResidualAttentionNetwork, implemented in Gluon."""

__all__ = ['ResidualAttentionModel', 'ResidualAttentionModel_32input',
           'residualattentionnet56', 'residualattentionnet56_32input',
           'residualattentionnet92', 'residualattentionnet92_32input']


__modify__ = 'X.Yang'
__modified_date__ = '18/10/23'

from mxnet.gluon import nn
from mxnet.gluon.block import HybridBlock


class UpsamplingBilinear2d(HybridBlock):
    r"""
    Parameters
    ----------
    size : int
        Upsampling size.
    """

    def __init__(self, size, **kwargs):
        super(UpsamplingBilinear2d, self).__init__(**kwargs)
        self.size = size

    def hybrid_forward(self, F, x):
        return F.contrib.BilinearResize2D(x, self.size, self.size)


class ResidualBlock(HybridBlock):
    r"""ResNet V2 model from
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.

    Parameters
    ----------
    channels : int
        Output channels
    in_channels : int
        Input channels
    stride : int
        Stride size.
    """

    def __init__(self, channels, in_channels=None, stride=1):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.stride = stride
        self.in_channels = in_channels if in_channels else channels
        with self.name_scope():
            self.bn1 = nn.BatchNorm()
            self.conv1 = nn.Conv2D(channels // 4, 1, 1, use_bias=False)
            self.bn2 = nn.BatchNorm()
            self.conv2 = nn.Conv2D(channels // 4, 3, stride, padding=1, use_bias=False)
            self.bn3 = nn.BatchNorm()
            self.conv3 = nn.Conv2D(channels, 1, 1, use_bias=False)
            if stride != 1 or (self.in_channels != self.channels):
                self.conv4 = nn.Conv2D(channels, 1, stride, use_bias=False)

    def hybrid_forward(self, F, x):
        residual = x
        out = self.bn1(x)
        out1 = F.Activation(out, act_type='relu')
        out = self.conv1(out1)
        out = self.bn2(out)
        out = F.Activation(out, act_type='relu')
        out = self.conv2(out)
        out = self.bn3(out)
        out = F.Activation(out, act_type='relu')
        out = self.conv3(out)
        if self.stride != 1 or (self.channels != self.in_channels):
            residual = self.conv4(out1)

        out = out + residual
        return out


class AttentionModule_stage1(nn.HybridBlock):
    r"""AttentionModel 56 model from
    `"Residual Attention Network for Image Classification"
    <https://arxiv.org/pdf/1704.06904.pdf>`_ paper.
    Input size is 56 x 56.
    Default size is for 56 stage input.
    If input size is different you need to change it suiting for your input size.

    Parameters
    ----------
    channels : int
        Output channels.
    size1 : int, default 56
        Upsampling size1.
    size2 : int, default 28
        Upsampling size2.
    size3 : int, default 14
        Upsampling size3.
    """

    def __init__(self, channels, size1=56, size2=28, size3=14, **kwargs):
        super(AttentionModule_stage1, self).__init__(**kwargs)
        with self.name_scope():
            self.first_residual_blocks = ResidualBlock(channels)

            self.trunk_branches = nn.HybridSequential()
            with self.trunk_branches.name_scope():
                self.trunk_branches.add(ResidualBlock(channels))
                self.trunk_branches.add(ResidualBlock(channels))

            self.mpool1 = nn.MaxPool2D(pool_size=3, strides=2, padding=1)
            self.softmax1_blocks = ResidualBlock(channels)
            self.skip1_connection_residual_block = ResidualBlock(channels)

            self.mpool2 = nn.MaxPool2D(pool_size=3, strides=2, padding=1)
            self.softmax2_blocks = ResidualBlock(channels)
            self.skip2_connection_residual_block = ResidualBlock(channels)

            self.mpool3 = nn.MaxPool2D(pool_size=3, strides=2, padding=1)

            self.softmax3_blocks = nn.HybridSequential()
            with self.softmax3_blocks.name_scope():
                self.softmax3_blocks.add(ResidualBlock(channels))
                self.softmax3_blocks.add(ResidualBlock(channels))

            self.interpolation3 = UpsamplingBilinear2d(size=size3)
            self.softmax4_blocks = ResidualBlock(channels)

            self.interpolation2 = UpsamplingBilinear2d(size=size2)
            self.softmax5_blocks = ResidualBlock(channels)

            self.interpolation1 = UpsamplingBilinear2d(size=size1)
            self.softmax6_blocks = nn.HybridSequential()
            with self.softmax6_blocks.name_scope():
                self.softmax6_blocks.add(nn.BatchNorm())
                self.softmax6_blocks.add(nn.Activation('relu'))
                self.softmax6_blocks.add(nn.Conv2D(channels, kernel_size=1, use_bias=False))
                self.softmax6_blocks.add(nn.BatchNorm())
                self.softmax6_blocks.add(nn.Activation('relu'))
                self.softmax6_blocks.add(nn.Conv2D(channels, kernel_size=1, use_bias=False))
                self.softmax6_blocks.add(nn.Activation('sigmoid'))

            self.last_blocks = ResidualBlock(channels)

    def hybrid_forward(self, F, x):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)

        out_mpool1 = self.mpool1(x)
        out_softmax1 = self.softmax1_blocks(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(out_softmax1)

        out_mpool2 = self.mpool2(out_softmax1)
        out_softmax2 = self.softmax2_blocks(out_mpool2)
        out_skip2_connection = self.skip2_connection_residual_block(out_softmax2)

        out_mpool3 = self.mpool3(out_softmax2)
        out_softmax3 = self.softmax3_blocks(out_mpool3)

        out_interp3 = F.elemwise_add(self.interpolation3(out_softmax3), out_softmax2)
        out = F.elemwise_add(out_interp3, out_skip2_connection)

        out_softmax4 = self.softmax4_blocks(out)
        out_interp2 = F.elemwise_add(self.interpolation2(out_softmax4), out_softmax1)
        out = F.elemwise_add(out_interp2, out_skip1_connection)

        out_softmax5 = self.softmax5_blocks(out)
        out_interp1 = F.elemwise_add(self.interpolation1(out_softmax5), out_trunk)

        out_softmax6 = self.softmax6_blocks(out_interp1)
        out = F.elemwise_add(F.ones_like(out_softmax6), out_softmax6)
        out = F.elemwise_mul(out, out_trunk)

        out_last = self.last_blocks(out)
        return out_last


class AttentionModule_stage2(nn.HybridBlock):
    r"""AttentionModel 56 model from
    `"Residual Attention Network for Image Classification"
    <https://arxiv.org/pdf/1704.06904.pdf>`_ paper.
    Input size is 28 x 28.
    Default size is for 28 stage input.
    If input size is different you need to change it suiting for your input size.

    Parameters
    ----------
    channels : int
        Output channels.
    size1 : int, default 28
        Upsampling size1.
    size2 : int, default 14
        Upsampling size2.
    """

    def __init__(self, channels, size1=28, size2=14, **kwargs):
        super(AttentionModule_stage2, self).__init__(**kwargs)
        with self.name_scope():
            self.first_residual_blocks = ResidualBlock(channels)

            self.trunk_branches = nn.HybridSequential()
            with self.trunk_branches.name_scope():
                self.trunk_branches.add(ResidualBlock(channels))
                self.trunk_branches.add(ResidualBlock(channels))

            self.mpool1 = nn.MaxPool2D(pool_size=3, strides=2, padding=1)
            self.softmax1_blocks = ResidualBlock(channels)
            self.skip1_connection_residual_block = ResidualBlock(channels)

            self.mpool2 = nn.MaxPool2D(pool_size=3, strides=2, padding=1)

            self.softmax2_blocks = nn.HybridSequential()
            with self.softmax2_blocks.name_scope():
                self.softmax2_blocks.add(ResidualBlock(channels))
                self.softmax2_blocks.add(ResidualBlock(channels))

            self.interpolation2 = UpsamplingBilinear2d(size=size2)
            self.softmax3_blocks = ResidualBlock(channels)

            self.interpolation1 = UpsamplingBilinear2d(size=size1)

            self.softmax4_blocks = nn.HybridSequential()
            with self.softmax4_blocks.name_scope():
                self.softmax4_blocks.add(nn.BatchNorm())
                self.softmax4_blocks.add(nn.Activation('relu'))
                self.softmax4_blocks.add(nn.Conv2D(channels, kernel_size=1, use_bias=False))
                self.softmax4_blocks.add(nn.BatchNorm())
                self.softmax4_blocks.add(nn.Activation('relu'))
                self.softmax4_blocks.add(nn.Conv2D(channels, kernel_size=1, use_bias=False))
                self.softmax4_blocks.add(nn.Activation('sigmoid'))
            self.last_blocks = ResidualBlock(channels)

    def hybrid_forward(self, F, x):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_mpool1 = self.mpool1(x)
        out_softmax1 = self.softmax1_blocks(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(out_softmax1)
        out_mpool2 = self.mpool2(out_softmax1)
        out_softmax2 = self.softmax2_blocks(out_mpool2)
        out_interp2 = F.elemwise_add(self.interpolation2(out_softmax2), out_softmax1)
        out = F.elemwise_add(out_interp2, out_skip1_connection)
        out_softmax3 = self.softmax3_blocks(out)
        out_interp1 = F.elemwise_add(self.interpolation1(out_softmax3), out_trunk)
        out_softmax4 = self.softmax4_blocks(out_interp1)
        out = F.elemwise_add(F.ones_like(out_softmax4), out_softmax4)
        out = F.elemwise_mul(out, out_trunk)
        out_last = self.last_blocks(out)
        return out_last


class AttentionModule_stage3(nn.HybridBlock):
    r"""AttentionModel 56 model from
    `"Residual Attention Network for Image Classification"
    <https://arxiv.org/pdf/1704.06904.pdf>`_ paper.
    Input size is 14 x 14.
    Default size is for 14 stage input.
    If input size is different you need to change it suiting for your input size.

    Parameters
    ----------
    channels : int
        Output channels.
    size1 : int, default 14
        Upsampling size1.
    """

    def __init__(self, channels, size1=14, **kwargs):
        super(AttentionModule_stage3, self).__init__(**kwargs)
        with self.name_scope():
            self.first_residual_blocks = ResidualBlock(channels)

            self.trunk_branches = nn.HybridSequential()
            with self.trunk_branches.name_scope():
                self.trunk_branches.add(ResidualBlock(channels))
                self.trunk_branches.add(ResidualBlock(channels))

            self.mpool1 = nn.MaxPool2D(pool_size=3, strides=2, padding=1)

            self.softmax1_blocks = nn.HybridSequential()
            with self.softmax1_blocks.name_scope():
                self.softmax1_blocks.add(ResidualBlock(channels))
                self.softmax1_blocks.add(ResidualBlock(channels))

            self.interpolation1 = UpsamplingBilinear2d(size=size1)

            self.softmax2_blocks = nn.HybridSequential()
            with self.softmax2_blocks.name_scope():
                self.softmax2_blocks.add(nn.BatchNorm())
                self.softmax2_blocks.add(nn.Activation('relu'))
                self.softmax2_blocks.add(nn.Conv2D(channels, kernel_size=1, use_bias=False))
                self.softmax2_blocks.add(nn.BatchNorm())
                self.softmax2_blocks.add(nn.Activation('relu'))
                self.softmax2_blocks.add(nn.Conv2D(channels, kernel_size=1, use_bias=False))
                self.softmax2_blocks.add(nn.Activation('sigmoid'))
            self.last_blocks = ResidualBlock(channels)

    def hybrid_forward(self, F, x):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_mpool1 = self.mpool1(x)
        out_softmax1 = self.softmax1_blocks(out_mpool1)
        out_interp1 = F.elemwise_add(self.interpolation1(out_softmax1), out_trunk)
        out_softmax2 = self.softmax2_blocks(out_interp1)
        out = F.elemwise_add(F.ones_like(out_softmax2), out_softmax2)
        out = F.elemwise_mul(out, out_trunk)
        out_last = self.last_blocks(out)
        return out_last


class AttentionModule_stage4(nn.HybridBlock):
    r"""AttentionModel 56 model from
    `"Residual Attention Network for Image Classification"
    <https://arxiv.org/pdf/1704.06904.pdf>`_ paper.
    Input size is 14 x 14.

    Parameters
    ----------
    channels : int
        Output channels.
    """

    def __init__(self, channels, **kwargs):
        super(AttentionModule_stage4, self).__init__(**kwargs)
        with self.name_scope():
            self.first_residual_blocks = ResidualBlock(channels)

            self.trunk_branches = nn.HybridSequential()
            with self.trunk_branches.name_scope():
                self.trunk_branches.add(ResidualBlock(channels))
                self.trunk_branches.add(ResidualBlock(channels))

            self.softmax1_blocks = nn.HybridSequential()
            with self.softmax1_blocks.name_scope():
                self.softmax1_blocks.add(ResidualBlock(channels))
                self.softmax1_blocks.add(ResidualBlock(channels))

            self.softmax2_blocks = nn.HybridSequential()
            with self.softmax2_blocks.name_scope():
                self.softmax2_blocks.add(nn.BatchNorm())
                self.softmax2_blocks.add(nn.Activation('relu'))
                self.softmax2_blocks.add(nn.Conv2D(channels, kernel_size=1, use_bias=False))
                self.softmax2_blocks.add(nn.BatchNorm())
                self.softmax2_blocks.add(nn.Activation('relu'))
                self.softmax2_blocks.add(nn.Conv2D(channels, kernel_size=1, use_bias=False))
                self.softmax2_blocks.add(nn.Activation('sigmoid'))
            self.last_blocks = ResidualBlock(channels)

    def hybrid_forward(self, F, x):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_softmax1 = self.softmax1_blocks(x)
        out_softmax2 = self.softmax2_blocks(out_softmax1)
        out = F.elemwise_add(F.ones_like(out_softmax2), out_softmax2)
        out = F.elemwise_mul(out, out_trunk)
        out_last = self.last_blocks(out)
        return out_last


class ResidualAttentionModel(nn.HybridBlock):
    r"""AttentionModel model from
    `"Residual Attention Network for Image Classification"
    <https://arxiv.org/pdf/1704.06904.pdf>`_ paper.
    Input size is 224 x 224.

    Parameters
    ----------
    classes : int, default 1000
        Number of classification classes.
    additional_stage : bool, default False
        If False means Attention56, True means Attention92.
    """

    def __init__(self, classes=1000, additional_stage=False, **kwargs):
        super(ResidualAttentionModel, self).__init__(**kwargs)
        self.additional_stage = additional_stage
        with self.name_scope():
            self.conv1 = nn.HybridSequential()
            with self.conv1.name_scope():
                self.conv1.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3, use_bias=False))
                self.conv1.add(nn.BatchNorm())
                self.conv1.add(nn.Activation('relu'))
            self.mpool1 = nn.MaxPool2D(pool_size=3, strides=2, padding=1)

            self.residual_block1 = ResidualBlock(256, in_channels=64)
            self.attention_module1 = AttentionModule_stage1(256)
            self.residual_block2 = ResidualBlock(512, in_channels=256, stride=2)
            self.attention_module2 = AttentionModule_stage2(512)
            if additional_stage:
                self.attention_module2_2 = AttentionModule_stage2(512)
            self.residual_block3 = ResidualBlock(1024, in_channels=512, stride=2)
            self.attention_module3 = AttentionModule_stage3(1024)
            if additional_stage:
                self.attention_module3_2 = AttentionModule_stage3(1024)
                self.attention_module3_3 = AttentionModule_stage3(1024)
            self.residual_block4 = ResidualBlock(2048, in_channels=1024, stride=2)
            self.residual_block5 = ResidualBlock(2048)
            self.residual_block6 = ResidualBlock(2048)
            self.mpool2 = nn.HybridSequential()
            with self.mpool2.name_scope():
                self.mpool2.add(nn.BatchNorm())
                self.mpool2.add(nn.Activation('relu'))
                self.mpool2.add(nn.AvgPool2D(pool_size=7, strides=1))
            self.fc = nn.Conv2D(classes, kernel_size=1)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.mpool1(x)
        x = self.residual_block1(x)
        x = self.attention_module1(x)
        x = self.residual_block2(x)
        x = self.attention_module2(x)
        if self.additional_stage:
            x = self.attention_module2_2(x)
        x = self.residual_block3(x)
        x = self.attention_module3(x)
        if self.additional_stage:
            x = self.attention_module3_2(x)
            x = self.attention_module3_3(x)
        x = self.residual_block4(x)
        x = self.residual_block5(x)
        x = self.residual_block6(x)
        x = self.mpool2(x)
        x = self.fc(x)
        x = F.Flatten(x)
        return x


class ResidualAttentionModel_32input(nn.HybridBlock):
    r"""AttentionModel model from
    `"Residual Attention Network for Image Classification"
    <https://arxiv.org/pdf/1704.06904.pdf>`_ paper.
    Input size is 32 x 32.

    Parameters
    ----------
    classes : int, default 10
        Number of classification classes.
    additional_stage : bool, default False
        If False means Attention56, True means Attention92.
    """

    def __init__(self, classes=10, additional_stage=False, **kwargs):
        super(ResidualAttentionModel_32input, self).__init__(**kwargs)
        self.additional_stage = additional_stage
        with self.name_scope():
            self.conv1 = nn.HybridSequential()
            with self.conv1.name_scope():
                self.conv1.add(nn.Conv2D(32, kernel_size=3, strides=1, padding=1, use_bias=False))
                self.conv1.add(nn.BatchNorm())
                self.conv1.add(nn.Activation('relu'))
            # 32 x 32
            # self.mpool1 = nn.MaxPool2D(pool_size=2, strides=2, padding=0)

            self.residual_block1 = ResidualBlock(128, in_channels=32)
            self.attention_module1 = AttentionModule_stage2(128, size1=32, size2=16)
            self.residual_block2 = ResidualBlock(256, in_channels=128, stride=2)
            self.attention_module2 = AttentionModule_stage3(256, size1=16)
            if additional_stage:
                self.attention_module2_2 = AttentionModule_stage3(256, size1=16)
            self.residual_block3 = ResidualBlock(512, in_channels=256, stride=2)
            self.attention_module3 = AttentionModule_stage4(512)
            if additional_stage:
                self.attention_module3_2 = AttentionModule_stage4(512)
                self.attention_module3_3 = AttentionModule_stage4(512)
            self.residual_block4 = ResidualBlock(1024, in_channels=512)
            self.residual_block5 = ResidualBlock(1024)
            self.residual_block6 = ResidualBlock(1024)
            self.mpool2 = nn.HybridSequential()
            with self.mpool2.name_scope():
                self.mpool2.add(nn.BatchNorm())
                self.mpool2.add(nn.Activation('relu'))
                self.mpool2.add(nn.AvgPool2D(pool_size=8, strides=1))
            self.fc = nn.Conv2D(classes, kernel_size=1)

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.residual_block1(x)
        x = self.attention_module1(x)
        x = self.residual_block2(x)
        x = self.attention_module2(x)
        if self.additional_stage:
            x = self.attention_module2_2(x)
        x = self.residual_block3(x)
        x = self.attention_module3(x)
        if self.additional_stage:
            x = self.attention_module3_2(x)
            x = self.attention_module3_3(x)
        x = self.residual_block4(x)
        x = self.residual_block5(x)
        x = self.residual_block6(x)
        x = self.mpool2(x)
        x = self.fc(x)
        x = F.Flatten(x)
        return x


def get_residualAttentionModel(input_size, additional_stage=False,
                               pretrained=None, ctx=None,
                               root=None, **kwargs):
    r"""AttentionModel model from
    `"Residual Attention Network for Image Classification"
    <https://arxiv.org/pdf/1704.06904.pdf>`_ paper.

    Parameters
    ----------
    input_size : int
        Input size of net. Options are 32, 224, 448.
    additional_stage : bool, default False
        If False means Attention56, True means Attention92.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    assert input_size in (32, 224)
    if input_size == 32:
        net = ResidualAttentionModel_32input(additional_stage=additional_stage, **kwargs)
    elif input_size == 224:
        net = ResidualAttentionModel(additional_stage=additional_stage, **kwargs)

    if pretrained:
        pass

    return net


def residualattentionnet56(**kwargs):
    r"""AttentionModel model from
    `"Residual Attention Network for Image Classification"
    <https://arxiv.org/pdf/1704.06904.pdf>`_ paper.

    Parameters
    ----------
    input_size : int
        Input size of net. Options are 32, 224, 448.
    additional_stage : bool, default False
        If False means Attention56, True means Attention92.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """

    return get_residualAttentionModel(224, False, **kwargs)


def residualattentionnet92(**kwargs):
    r"""AttentionModel model from
    `"Residual Attention Network for Image Classification"
    <https://arxiv.org/pdf/1704.06904.pdf>`_ paper.

    Parameters
    ----------
    input_size : int
        Input size of net. Options are 32, 224, 448.
    additional_stage : bool, default False
        If False means Attention56, True means Attention92.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """

    return get_residualAttentionModel(224, True, **kwargs)


def residualattentionnet56_32input(**kwargs):
    r"""AttentionModel model from
    `"Residual Attention Network for Image Classification"
    <https://arxiv.org/pdf/1704.06904.pdf>`_ paper.

    Parameters
    ----------
    input_size : int
        Input size of net. Options are 32, 224, 448.
    additional_stage : bool, default False
        If False means Attention56, True means Attention92.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """

    return get_residualAttentionModel(32, False, **kwargs)


def residualattentionnet92_32input(**kwargs):
    r"""AttentionModel model from
    `"Residual Attention Network for Image Classification"
    <https://arxiv.org/pdf/1704.06904.pdf>`_ paper.

    Parameters
    ----------
    input_size : int
        Input size of net. Options are 32, 224, 448.
    additional_stage : bool, default False
        If False means Attention56, True means Attention92.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """

    return get_residualAttentionModel(32, True, **kwargs)
