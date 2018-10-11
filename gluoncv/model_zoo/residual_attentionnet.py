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
# pylint: disable= arguments-differ,unused-argument
"""ResidualAttentionNetwork, implemented in Gluon."""

__all__ = ['ResidualAttentionModel_56', 'ResidualAttentionModel_92',
           'ResidualAttentionModel_56_32input', 'ResidualAttentionModel_92_32input',
           'ResidualAttentionModel_448input']

__modify__ = 'Piston Yang'
__modified_date__ = '18/10/11'

from mxnet.gluon import nn
from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock


class UpsamplingBilinear2d(HybridBlock):
    def __init__(self, size, **kwargs):
        super(UpsamplingBilinear2d, self).__init__(**kwargs)
        self.size = size

    def hybrid_forward(self, F, x, *args, **kwargs):
        return F.contrib.BilinearResize2D(x, self.size, self.size)


class ResidualBlock(HybridBlock):
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

    def hybrid_forward(self, F, x, *args, **kwargs):
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


class AttentionModule_stage0(nn.HybridBlock):
    def __init__(self, channels, size1=112, size2=56, size3=28, size4=14, **kwargs):
        """
        Input size is 112 x 112
        :param channels:
        :param size1:
        :param size2:
        :param size3:
        :param size4:
        :param kwargs:
        """
        super(AttentionModule_stage0, self).__init__(**kwargs)
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
            self.softmax3_blocks = ResidualBlock(channels)
            self.skip3_connection_residual_block = ResidualBlock(channels)

            self.mpool4 = nn.MaxPool2D(pool_size=3, strides=2, padding=1)
            self.softmax4_blocks = nn.HybridSequential()
            with self.softmax4_blocks.name_scope():
                self.softmax4_blocks.add(ResidualBlock(channels))
                self.softmax4_blocks.add(ResidualBlock(channels))

            self.interpolation4 = UpsamplingBilinear2d(size=size4)
            self.softmax5_blocks = ResidualBlock(channels)

            self.interpolation3 = UpsamplingBilinear2d(size=size3)
            self.softmax6_blocks = ResidualBlock(channels)

            self.interpolation2 = UpsamplingBilinear2d(size=size2)
            self.softmax7_blocks = ResidualBlock(channels)

            self.interpolation1 = UpsamplingBilinear2d(size=size1)

            self.softmax8_blocks = nn.HybridSequential()
            with self.softmax8_blocks.name_scope():
                self.softmax8_blocks.add(nn.BatchNorm())
                self.softmax8_blocks.add(nn.Activation('relu'))
                self.softmax8_blocks.add(nn.Conv2D(channels, kernel_size=1, strides=1, use_bias=False))
                self.softmax8_blocks.add(nn.BatchNorm())
                self.softmax8_blocks.add(nn.Activation('relu'))
                self.softmax8_blocks.add(nn.Conv2D(channels, kernel_size=1, strides=1, use_bias=False))
                self.softmax8_blocks.add(nn.Activation('sigmoid'))

            self.last_blocks = ResidualBlock(channels)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)

        out_mpool1 = self.mpool1(x)
        out_softmax1 = self.softmax1_blocks(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(out_softmax1)
        # 56 x 56

        out_mpool2 = self.mpool2(out_softmax1)
        out_softmax2 = self.softmax2_blocks(out_mpool2)
        out_skip2_connection = self.skip2_connection_residual_block(out_softmax2)
        # 28 x 28

        out_mpool3 = self.mpool3(out_softmax2)
        out_softmax3 = self.softmax3_blocks(out_mpool3)
        out_skip3_connection = self.skip3_connection_residual_block(out_softmax3)
        # 14 x 14

        out_mpool4 = self.mpool4(out_softmax3)
        out_softmax4 = self.softmax4_blocks(out_mpool4)
        # 7 x 7

        out_interp4 = F.elemwise_add(self.interpolation4(out_softmax4), out_softmax3)
        out = F.elemwise_add(out_interp4, out_skip3_connection)

        out_softmax5 = self.softmax5_blocks(out)
        out_interp3 = F.elemwise_add(self.interpolation3(out_softmax5), out_softmax2)
        out = F.elemwise_add(out_interp3, out_skip2_connection)

        out_softmax6 = self.softmax5_blocks(out)
        out_interp2 = F.elemwise_add(self.interpolation2(out_softmax6), out_softmax1)
        out = F.elemwise_add(out_interp2, out_skip1_connection)

        out_softmax7 = self.softmax7_blocks(out)
        out_interp1 = F.elemwise_add(self.interpolation1(out_softmax7), out_trunk)

        out_softmax8 = self.softmax8_blocks(out_interp1)
        out = F.elemwise_add(F.ones_like(out_softmax8), out_softmax8)
        out = F.elemwise_mul(out, out_trunk)

        out_last = self.last_blocks(out)

        return out_last


class AttentionModule_stage1(nn.HybridBlock):
    def __init__(self, channels, size1=56, size2=28, size3=14, **kwargs):
        """
        Input size is 56 x 56
        :param channels:
        :param size1:
        :param size2:
        :param size3:
        """
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
                self.softmax6_blocks.add(nn.Conv2D(channels, kernel_size=1, strides=1, use_bias=False))
                self.softmax6_blocks.add(nn.BatchNorm())
                self.softmax6_blocks.add(nn.Activation('relu'))
                self.softmax6_blocks.add(nn.Conv2D(channels, kernel_size=1, strides=1, use_bias=False))
                self.softmax6_blocks.add(nn.Activation('sigmoid'))

            self.last_blocks = ResidualBlock(channels)

    def hybrid_forward(self, F, x, *args, **kwargs):
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
    def __init__(self, channels, size1=28, size2=14, **kwargs):
        """
        Input size is 28 x 28
        :param channels:
        :param size1:
        :param size2:
        """
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
                self.softmax4_blocks.add(nn.Conv2D(channels, kernel_size=1, strides=1, use_bias=False))
                self.softmax4_blocks.add(nn.BatchNorm())
                self.softmax4_blocks.add(nn.Activation('relu'))
                self.softmax4_blocks.add(nn.Conv2D(channels, kernel_size=1, strides=1, use_bias=False))
                self.softmax4_blocks.add(nn.Activation('sigmoid'))

            self.last_blocks = ResidualBlock(channels)

    def hybrid_forward(self, F, x, *args, **kwargs):
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
    def __init__(self, channels, size1=14, **kwargs):
        """
        Input size is 14 x 14
        :param channels:
        :param size1:
        :param kwargs:
        """
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
                self.softmax2_blocks.add(nn.Conv2D(channels, kernel_size=1, strides=1, use_bias=False))
                self.softmax2_blocks.add(nn.BatchNorm())
                self.softmax2_blocks.add(nn.Activation('relu'))
                self.softmax2_blocks.add(nn.Conv2D(channels, kernel_size=1, strides=1, use_bias=False))
                self.softmax2_blocks.add(nn.Activation('sigmoid'))

            self.last_blocks = ResidualBlock(channels)

    def hybrid_forward(self, F, x, *args, **kwargs):
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


class AttentionModule_stage1_cifar(nn.HybridBlock):
    def __init__(self, channels, size1=16, size2=8, **kwargs):
        """
        Input size is 16 x 16
        :param channels:
        :param size1:
        :param size2:
        :param kwargs:
        """
        super(AttentionModule_stage1_cifar, self).__init__(**kwargs)
        with self.name_scope():
            self.first_residual_blocks = ResidualBlock(channels)

            self.trunk_branches = nn.HybridSequential()
            with self.trunk_branches.name_scope():
                self.trunk_branches.add(ResidualBlock(channels))
                self.trunk_branches.add(ResidualBlock(channels))

            self.mpool1 = nn.MaxPool2D(pool_size=3, strides=2, padding=1)
            self.down_residual_blocks1 = ResidualBlock(channels)
            self.skip1_connection_residual_block = ResidualBlock(channels)

            self.mpool2 = nn.MaxPool2D(pool_size=3, strides=2, padding=1)

            self.middle_2r_blocks = nn.HybridSequential()
            with self.middle_2r_blocks.name_scope():
                self.middle_2r_blocks.add(ResidualBlock(channels))
                self.middle_2r_blocks.add(ResidualBlock(channels))

            self.interpolation1 = UpsamplingBilinear2d(size=size2)
            self.up_residual_blocks1 = ResidualBlock(channels)

            self.interpolation2 = UpsamplingBilinear2d(size=size1)

            self.conv1_1_blocks = nn.HybridSequential()
            with self.conv1_1_blocks.name_scope():
                self.conv1_1_blocks.add(nn.BatchNorm())
                self.conv1_1_blocks.add(nn.Activation('relu'))
                self.conv1_1_blocks.add(nn.Conv2D(channels, kernel_size=1, strides=1, use_bias=False))
                self.conv1_1_blocks.add(nn.BatchNorm())
                self.conv1_1_blocks.add(nn.Activation('relu'))
                self.conv1_1_blocks.add(nn.Conv2D(channels, kernel_size=1, strides=1, use_bias=False))
                self.conv1_1_blocks.add(nn.Activation('sigmoid'))

            self.last_blocks = ResidualBlock(channels)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)

        out_mpool1 = self.mpool1(x)
        out_down_residual_blocks1 = self.down_residual_blocks1(out_mpool1)
        out_skip1_connection = self.skip1_connection_residual_block(out_down_residual_blocks1)

        out_mpool2 = self.mpool2(out_down_residual_blocks1)
        out_middle_2r_blocks = self.middle_2r_blocks(out_mpool2)

        out_interp = F.elemwise_add(self.interpolation1(out_middle_2r_blocks), out_down_residual_blocks1)
        out = F.elemwise_add(out_interp, out_skip1_connection)

        out_up_residual_blocks1 = self.up_residual_blocks1(out)
        out_interp2 = F.elemwise_add(self.interpolation2(out_up_residual_blocks1), out_trunk)

        out_conv1_1_blocks = self.conv1_1_blocks(out_interp2)
        out = F.elemwise_add(F.ones_like(out_conv1_1_blocks), out_conv1_1_blocks)
        out = F.elemwise_mul(out, out_trunk)

        out_last = self.last_blocks(out)

        return out_last


class AttentionModule_stage2_cifar(nn.HybridBlock):
    def __init__(self, channels, size1=8, **kwargs):
        """
        Input size is 14 x 14
        :param channels:
        :param size1:
        :param kwargs:
        """
        super(AttentionModule_stage2_cifar, self).__init__(**kwargs)
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
                self.softmax2_blocks.add(nn.Conv2D(channels, kernel_size=1, strides=1, use_bias=False))
                self.softmax2_blocks.add(nn.BatchNorm())
                self.softmax2_blocks.add(nn.Activation('relu'))
                self.softmax2_blocks.add(nn.Conv2D(channels, kernel_size=1, strides=1, use_bias=False))
                self.softmax2_blocks.add(nn.Activation('sigmoid'))

            self.last_blocks = ResidualBlock(channels)

    def hybrid_forward(self, F, x, *args, **kwargs):
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


class AttentionModule_stage3_cifar(nn.HybridBlock):
    def __init__(self, channels, **kwargs):
        """
        Input size is 14 x 14
        :param channels:
        :param kwargs:
        """
        super(AttentionModule_stage3_cifar, self).__init__(**kwargs)
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
                self.softmax2_blocks.add(nn.Conv2D(channels, kernel_size=1, strides=1, use_bias=False))
                self.softmax2_blocks.add(nn.BatchNorm())
                self.softmax2_blocks.add(nn.Activation('relu'))
                self.softmax2_blocks.add(nn.Conv2D(channels, kernel_size=1, strides=1, use_bias=False))
                self.softmax2_blocks.add(nn.Activation('sigmoid'))

            self.last_blocks = ResidualBlock(channels)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)

        out_softmax1 = self.softmax1_blocks(x)

        out_softmax2 = self.softmax2_blocks(out_softmax1)
        out = F.elemwise_add(F.ones_like(out_softmax2), out_softmax2)
        out = F.elemwise_mul(out, out_trunk)

        out_last = self.last_blocks(out)

        return out_last


class ResidualAttentionModel_448input(nn.HybridBlock):

    def __init__(self, **kwargs):
        """
        input size is 448
        :param kwargs:
        """
        super(ResidualAttentionModel_448input, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = nn.HybridSequential()
            with self.conv1.name_scope():
                self.conv1.add(nn.BatchNorm(scale=False, center=False))
                self.conv1.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3, use_bias=False))
                self.conv1.add(nn.BatchNorm())
                self.conv1.add(nn.Activation('relu'))
            self.mpool1 = nn.MaxPool2D(pool_size=3, strides=2, padding=1)
            # 112 x 112
            self.residual_block0 = ResidualBlock(128, in_channels=64)
            self.attention_module0 = AttentionModule_stage0(128)
            self.residual_block1 = ResidualBlock(256, in_channels=128, stride=2)
            # 56 x 56
            self.attention_module1 = AttentionModule_stage1(256)
            self.residual_block2 = ResidualBlock(512, in_channels=256, stride=2)
            self.attention_module2 = AttentionModule_stage2(512)
            self.attention_module2_2 = AttentionModule_stage2(512)
            self.residual_block3 = ResidualBlock(1024, in_channels=512, stride=2)
            self.attention_module3 = AttentionModule_stage3(1024)
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
            self.fc = nn.Conv2D(10, kernel_size=1)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.conv1(x)
        x = self.mpool1(x)
        x = self.residual_block0(x)
        x = self.attention_module0(x)

        x = self.residual_block1(x)
        x = self.attention_module1(x)
        x = self.residual_block2(x)
        x = self.attention_module2(x)
        x = self.attention_module2_2(x)
        x = self.residual_block3(x)

        x = self.attention_module3(x)
        x = self.attention_module3_2(x)
        x = self.attention_module3_3(x)
        x = self.residual_block4(x)
        x = self.residual_block5(x)
        x = self.residual_block6(x)
        x = self.mpool2(x)
        x = self.fc(x)
        x = F.Flatten(x)

        return x


class ResidualAttentionModel_92(nn.HybridBlock):

    def __init__(self, classes=1000, **kwargs):
        super(ResidualAttentionModel_92, self).__init__(**kwargs)
        r"""AttentionModel 92 model from
            `"Residual Attention Network for Image Classification"
            <https://arxiv.org/pdf/1704.06904.pdf>`_ paper.
            
            Input size must be 224.
            
            Parameters
            ----------
            classes : int, default 1000
                Number of classification classes.
        """
        with self.name_scope():
            self.conv1 = nn.HybridSequential()
            with self.conv1.name_scope():
                self.conv1.add(nn.BatchNorm(scale=False, center=False))
                self.conv1.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3, use_bias=False))
                self.conv1.add(nn.BatchNorm())
                self.conv1.add(nn.Activation('relu'))
            self.mpool1 = nn.MaxPool2D(pool_size=3, strides=2, padding=1)

            self.residual_block1 = ResidualBlock(256, in_channels=64)
            self.attention_module1 = AttentionModule_stage1(256)
            self.residual_block2 = ResidualBlock(512, in_channels=256, stride=2)
            self.attention_module2 = AttentionModule_stage2(512)
            self.attention_module2_2 = AttentionModule_stage2(512)
            self.residual_block3 = ResidualBlock(1024, in_channels=512, stride=2)
            self.attention_module3 = AttentionModule_stage3(1024)
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

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.conv1(x)
        x = self.mpool1(x)

        x = self.residual_block1(x)
        x = self.attention_module1(x)
        x = self.residual_block2(x)
        x = self.attention_module2(x)
        x = self.attention_module2_2(x)
        x = self.residual_block3(x)

        x = self.attention_module3(x)
        x = self.attention_module3_2(x)
        x = self.attention_module3_3(x)
        x = self.residual_block4(x)
        x = self.residual_block5(x)
        x = self.residual_block6(x)
        x = self.mpool2(x)
        x = self.fc(x)
        x = F.Flatten(x)

        return x


class ResidualAttentionModel_56(nn.HybridBlock):

    def __init__(self, classes=1000, **kwargs):
        super(ResidualAttentionModel_56, self).__init__(**kwargs)
        r"""AttentionModel 56 model from
            `"Residual Attention Network for Image Classification"
            <https://arxiv.org/pdf/1704.06904.pdf>`_ paper.
            
            Input size must be 224.
            
            Parameters
            ----------
            classes : int, default 1000
                Number of classification classes.
        """
        with self.name_scope():
            self.conv1 = nn.HybridSequential()
            with self.conv1.name_scope():
                self.conv1.add(nn.BatchNorm(scale=False, center=False))
                self.conv1.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3, use_bias=False))
                self.conv1.add(nn.BatchNorm())
                self.conv1.add(nn.Activation('relu'))
            self.mpool1 = nn.MaxPool2D(pool_size=3, strides=2, padding=1)

            self.residual_block1 = ResidualBlock(256, in_channels=64)
            self.attention_module1 = AttentionModule_stage1(256)
            self.residual_block2 = ResidualBlock(512, in_channels=256, stride=2)
            self.attention_module2 = AttentionModule_stage2(512)
            self.residual_block3 = ResidualBlock(1024, in_channels=512, stride=2)
            self.attention_module3 = AttentionModule_stage3(1024)
            self.residual_block4 = ResidualBlock(2048, in_channels=1024, stride=2)
            self.residual_block5 = ResidualBlock(2048)
            self.residual_block6 = ResidualBlock(2048)
            self.mpool2 = nn.HybridSequential()
            with self.mpool2.name_scope():
                self.mpool2.add(nn.BatchNorm())
                self.mpool2.add(nn.Activation('relu'))
                self.mpool2.add(nn.AvgPool2D(pool_size=7, strides=1))
            self.fc = nn.Conv2D(classes, kernel_size=1)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.conv1(x)
        x = self.mpool1(x)

        x = self.residual_block1(x)
        x = self.attention_module1(x)
        x = self.residual_block2(x)
        x = self.attention_module2(x)
        x = self.residual_block3(x)

        x = self.attention_module3(x)
        x = self.residual_block4(x)
        x = self.residual_block5(x)
        x = self.residual_block6(x)
        x = self.mpool2(x)
        x = self.fc(x)
        x = F.Flatten(x)
        return x


class ResidualAttentionModel_56_32input(nn.HybridBlock):
    def __init__(self, classes=10, **kwargs):
        super(ResidualAttentionModel_56_32input, self).__init__(**kwargs)
        r"""AttentionModel 56 model from
            `"Residual Attention Network for Image Classification"
            <https://arxiv.org/pdf/1704.06904.pdf>`_ paper.

            Input size must be 32.

            Parameters
            ----------
            classes : int, default 10
                Number of classification classes.
        """
        with self.name_scope():
            self.conv1 = nn.HybridSequential()
            with self.conv1.name_scope():
                self.conv1.add(nn.BatchNorm(scale=False, center=False))
                self.conv1.add(nn.Conv2D(32, kernel_size=3, strides=1, padding=1, use_bias=False))
                self.conv1.add(nn.BatchNorm())
                self.conv1.add(nn.Activation('relu'))
            # 32 x 32
            # self.mpool1 = nn.MaxPool2D(pool_size=2, strides=2, padding=0)

            self.residual_block1 = ResidualBlock(128, in_channels=32)
            self.attention_module1 = AttentionModule_stage1_cifar(128, size1=32, size2=16)
            self.residual_block2 = ResidualBlock(256, in_channels=128, stride=2)
            self.attention_module2 = AttentionModule_stage2_cifar(256, size1=16)
            self.residual_block3 = ResidualBlock(512, in_channels=256, stride=2)
            self.attention_module3 = AttentionModule_stage3_cifar(512)
            self.residual_block4 = ResidualBlock(1024, in_channels=512)
            self.residual_block5 = ResidualBlock(1024)
            self.residual_block6 = ResidualBlock(1024)
            self.mpool2 = nn.HybridSequential()
            with self.mpool2.name_scope():
                self.mpool2.add(nn.BatchNorm())
                self.mpool2.add(nn.Activation('relu'))
                self.mpool2.add(nn.AvgPool2D(pool_size=8, strides=1))
            self.fc = nn.Conv2D(classes, kernel_size=1)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.conv1(x)
        x = self.residual_block1(x)
        x = self.attention_module1(x)
        x = self.residual_block2(x)
        x = self.attention_module2(x)
        x = self.residual_block3(x)
        x = self.attention_module3(x)
        x = self.residual_block4(x)
        x = self.residual_block5(x)
        x = self.residual_block6(x)
        x = self.mpool2(x)
        x = self.fc(x)
        x = F.Flatten(x)

        return x


class ResidualAttentionModel_92_32input(nn.HybridBlock):
    def __init__(self, classes=10, **kwargs):
        super(ResidualAttentionModel_92_32input, self).__init__(**kwargs)
        r"""AttentionModel 92 model from
            `"Residual Attention Network for Image Classification"
            <https://arxiv.org/pdf/1704.06904.pdf>`_ paper.
            
            Input size must be 32.
            
            Parameters
            ----------
            classes : int, default 10
                Number of classification classes.
        """
        with self.name_scope():
            self.conv1 = nn.HybridSequential()
            with self.conv1.name_scope():
                self.conv1.add(nn.BatchNorm(scale=False, center=False))
                self.conv1.add(nn.Conv2D(32, kernel_size=3, strides=1, padding=1, use_bias=False))
                self.conv1.add(nn.BatchNorm())
                self.conv1.add(nn.Activation('relu'))
            # 32 x 32
            # self.mpool1 = nn.MaxPool2D(pool_size=2, strides=2, padding=0)

            self.residual_block1 = ResidualBlock(128, in_channels=32)
            self.attention_module1 = AttentionModule_stage1_cifar(128, size1=32, size2=16)
            self.residual_block2 = ResidualBlock(256, in_channels=128, stride=2)
            self.attention_module2 = AttentionModule_stage2_cifar(256, size1=16)
            self.attention_module2_2 = AttentionModule_stage2_cifar(256, size1=16)
            self.residual_block3 = ResidualBlock(512, in_channels=256, stride=2)
            self.attention_module3 = AttentionModule_stage3_cifar(512)
            self.attention_module3_2 = AttentionModule_stage3_cifar(512)
            self.attention_module3_3 = AttentionModule_stage3_cifar(512)
            self.residual_block4 = ResidualBlock(1024, in_channels=512)
            self.residual_block5 = ResidualBlock(1024)
            self.residual_block6 = ResidualBlock(1024)
            self.mpool2 = nn.HybridSequential()
            with self.mpool2.name_scope():
                self.mpool2.add(nn.BatchNorm())
                self.mpool2.add(nn.Activation('relu'))
                self.mpool2.add(nn.AvgPool2D(pool_size=8, strides=1))
            self.fc = nn.Conv2D(classes, kernel_size=1)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.conv1(x)
        x = self.residual_block1(x)
        x = self.attention_module1(x)
        x = self.residual_block2(x)
        x = self.attention_module2(x)
        x = self.attention_module2_2(x)
        x = self.residual_block3(x)
        x = self.attention_module3(x)
        x = self.attention_module3_2(x)
        x = self.attention_module3_3(x)
        x = self.residual_block4(x)
        x = self.residual_block5(x)
        x = self.residual_block6(x)
        x = self.mpool2(x)
        x = self.fc(x)
        x = F.Flatten(x)

        return x
