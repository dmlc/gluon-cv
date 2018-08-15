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
# pylint: disable= arguments-differ,missing-docstring,unused-argument
"""NASNet, implemented in Gluon."""
from __future__ import division

__all__ = ['get_nasnet', 'nasnet_4_1056', 'nasnet_5_1538', 'nasnet_7_1920', 'nasnet_6_4032']

import os
from mxnet import cpu
from mxnet.gluon import nn
from mxnet.gluon.block import HybridBlock

class MaxPoolPad(HybridBlock):

    def __init__(self):
        super(MaxPoolPad, self).__init__()
        self.pool = nn.MaxPool2D(3, strides=2, padding=1)

    def hybrid_forward(self, F, x):
        x = F.pad(x, pad_width=(0, 0, 0, 0, 1, 0, 1, 0),
                  mode='constant', constant_value=0)
        x = self.pool(x)
        x = F.slice(x, begin=(0, 0, 1, 1), end=(None, None, None, None))
        return x

class AvgPoolPad(HybridBlock):

    def __init__(self, stride=2, padding=1):
        super(AvgPoolPad, self).__init__()
        # There's no 'count_include_pad' parameter, which makes it different
        self.pool = nn.AvgPool2D(3, strides=stride, padding=padding, count_include_pad=False)

    def hybrid_forward(self, F, x):
        x = F.pad(x, pad_width=(0, 0, 0, 0, 1, 0, 1, 0),
                  mode='constant', constant_value=0)
        x = self.pool(x)
        x = F.slice(x, begin=(0, 0, 1, 1), end=(None, None, None, None))
        return x

class SeparableConv2d(HybridBlock):

    def __init__(self, in_channels, channels, dw_kernel, dw_stride, dw_padding,
                 use_bias=False):
        super(SeparableConv2d, self).__init__()
        self.body = nn.HybridSequential(prefix='')
        self.body.add(nn.Conv2D(in_channels, kernel_size=dw_kernel,
                                strides=dw_stride, padding=dw_padding,
                                use_bias=use_bias,
                                groups=in_channels))
        self.body.add(nn.Conv2D(channels, kernel_size=1, strides=1, use_bias=use_bias))

    def hybrid_forward(self, F, x):
        x = self.body(x)
        return x

class BranchSeparables(HybridBlock):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_bias=False):
        super(BranchSeparables, self).__init__()
        self.body = nn.HybridSequential(prefix='')

        self.body.add(nn.Activation('relu'))
        self.body.add(SeparableConv2d(in_channels, in_channels, kernel_size,
                                      stride, padding, use_bias=use_bias))
        self.body.add(nn.BatchNorm(momentum=0.1, epsilon=0.001))
        self.body.add(nn.Activation('relu'))
        self.body.add(SeparableConv2d(in_channels, out_channels, kernel_size,
                                      1, padding, use_bias=use_bias))
        self.body.add(nn.BatchNorm(momentum=0.1, epsilon=0.001))

    def hybrid_forward(self, F, x):
        x = self.body(x)
        return(x)

class BranchSeparablesStem(HybridBlock):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_bias=False):
        super(BranchSeparablesStem, self).__init__()
        self.body = nn.HybridSequential(prefix='')

        self.body.add(nn.Activation('relu'))
        self.body.add(SeparableConv2d(in_channels, out_channels, kernel_size,
                                      stride, padding, use_bias=use_bias))
        self.body.add(nn.BatchNorm(momentum=0.1, epsilon=0.001))
        self.body.add(nn.Activation('relu'))
        self.body.add(SeparableConv2d(out_channels, out_channels, kernel_size,
                                      1, padding, use_bias=use_bias))
        self.body.add(nn.BatchNorm(momentum=0.1, epsilon=0.001))

    def hybrid_forward(self, F, x):
        x = self.body(x)
        return(x)

class BranchSeparablesReduction(HybridBlock):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 z_padding=1, use_bias=False):
        super(BranchSeparablesReduction, self).__init__()

        self.z_padding = z_padding
        self.separable = SeparableConv2d(in_channels, in_channels, kernel_size,
                                         stride, padding, use_bias=use_bias)

        self.body = nn.HybridSequential(prefix='')
        self.body.add(nn.BatchNorm(momentum=0.1, epsilon=0.001))
        self.body.add(nn.Activation('relu'))
        self.body.add(SeparableConv2d(in_channels, out_channels, kernel_size,
                                      1, padding, use_bias=use_bias))
        self.body.add(nn.BatchNorm(momentum=0.1, epsilon=0.001))

    def hybrid_forward(self, F, x):
        x = F.Activation(x, act_type='relu')
        x = F.pad(x, pad_width=(0, 0, 0, 0, self.z_padding, 0, self.z_padding, 0),
                  mode='constant', constant_value=0)
        x = self.separable(x)
        x = F.slice(x, begin=(0, 0, 1, 1), end=(None, None, None, None))
        x = self.body(x)
        return(x)

class CellStem0(HybridBlock):

    def __init__(self, stem_filters, num_filters=42):
        super(CellStem0, self).__init__()

        self.conv_1x1 = nn.HybridSequential(prefix='')
        self.conv_1x1.add(nn.Activation('relu'))
        self.conv_1x1.add(nn.Conv2D(num_filters, 1, strides=1, use_bias=False))
        self.conv_1x1.add(nn.BatchNorm(momentum=0.1, epsilon=0.001))

        self.comb_iter_0_left = BranchSeparables(num_filters, num_filters, 5, 2, 2)
        self.comb_iter_0_right = BranchSeparablesStem(stem_filters, num_filters, 7, 2, 3)

        self.comb_iter_1_left = nn.MaxPool2D(3, strides=2, padding=1)
        self.comb_iter_1_right = BranchSeparablesStem(stem_filters, num_filters, 7, 2, 3)

        self.comb_iter_2_left = nn.AvgPool2D(3, strides=2, padding=1, count_include_pad=False)
        self.comb_iter_2_right = BranchSeparablesStem(stem_filters, num_filters, 5, 2, 2)

        self.comb_iter_3_right = nn.AvgPool2D(3, strides=1, padding=1, count_include_pad=False)

        self.comb_iter_4_left = BranchSeparables(num_filters, num_filters, 3, 1, 1)
        self.comb_iter_4_right = nn.MaxPool2D(3, strides=2, padding=1)

    def hybrid_forward(self, F, x):
        x1 = self.conv_1x1(x)

        x_comb_iter_0_left = self.comb_iter_0_left(x1)
        x_comb_iter_0_right = self.comb_iter_0_right(x)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x1)
        x_comb_iter_1_right = self.comb_iter_1_right(x)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x1)
        x_comb_iter_2_right = self.comb_iter_2_right(x)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right

        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1

        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x1)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right

        x_out = F.concat(x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4, dim=1)

        return x_out

class CellStem1(HybridBlock):

    def __init__(self, num_filters):
        super(CellStem1, self).__init__()

        self.conv_1x1 = nn.HybridSequential(prefix='')
        self.conv_1x1.add(nn.Activation('relu'))
        self.conv_1x1.add(nn.Conv2D(num_filters, 1, strides=1, use_bias=False))
        self.conv_1x1.add(nn.BatchNorm(momentum=0.1, epsilon=0.001))

        self.path_1 = nn.HybridSequential(prefix='')
        self.path_1.add(nn.AvgPool2D(1, strides=2, count_include_pad=False))
        self.path_1.add(nn.Conv2D(num_filters//2, 1, strides=1, use_bias=False))

        self.path_2 = nn.HybridSequential(prefix='')
        # No nn.ZeroPad2D in gluon
        self.path_2.add(nn.AvgPool2D(1, strides=2, count_include_pad=False))
        self.path_2.add(nn.Conv2D(num_filters//2, 1, strides=1, use_bias=False))

        self.final_path_bn = nn.BatchNorm(momentum=0.1, epsilon=0.001)

        self.comb_iter_0_left = BranchSeparables(num_filters, num_filters, 5, 2, 2)
        self.comb_iter_0_right = BranchSeparables(num_filters, num_filters, 7, 2, 3)

        self.comb_iter_1_left = nn.MaxPool2D(3, strides=2, padding=1)
        self.comb_iter_1_right = BranchSeparables(num_filters, num_filters, 7, 2, 3)

        self.comb_iter_2_left = nn.AvgPool2D(3, strides=2, padding=1, count_include_pad=False)
        self.comb_iter_2_right = BranchSeparables(num_filters, num_filters, 5, 2, 2)

        self.comb_iter_3_right = nn.AvgPool2D(3, strides=1, padding=1, count_include_pad=False)

        self.comb_iter_4_left = BranchSeparables(num_filters, num_filters, 3, 1, 1)
        self.comb_iter_4_right = nn.MaxPool2D(3, strides=2, padding=1)

    def hybrid_forward(self, F, x_conv0, x_stem_0):
        x_left = self.conv_1x1(x_stem_0)

        x_relu = F.Activation(x_conv0, act_type='relu')
        x_path1 = self.path_1(x_relu)
        x_path2 = F.pad(x_relu, pad_width=(0, 0, 0, 0, 0, 1, 0, 1),
                        mode='constant', constant_value=0)
        x_path2 = F.slice(x_path2, begin=(0, 0, 1, 1), end=(None, None, None, None))
        x_path2 = self.path_2(x_path2)
        x_right = self.final_path_bn(F.concat(x_path1, x_path2, dim=1))

        x_comb_iter_0_left = self.comb_iter_0_left(x_left)
        x_comb_iter_0_right = self.comb_iter_0_right(x_right)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_right)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_left)
        x_comb_iter_2_right = self.comb_iter_2_right(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right

        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1

        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_left)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right

        x_out = F.concat(x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4, dim=1)

        return x_out


class FirstCell(HybridBlock):

    def __init__(self, out_channels_left, out_channels_right):
        super(FirstCell, self).__init__()

        self.conv_1x1 = nn.HybridSequential(prefix='')
        self.conv_1x1.add(nn.Activation('relu'))
        self.conv_1x1.add(nn.Conv2D(out_channels_right, 1, strides=1, use_bias=False))
        self.conv_1x1.add(nn.BatchNorm(momentum=0.1, epsilon=0.001))

        self.path_1 = nn.HybridSequential(prefix='')
        self.path_1.add(nn.AvgPool2D(1, strides=2, count_include_pad=False))
        self.path_1.add(nn.Conv2D(out_channels_left, 1, strides=1, use_bias=False))

        self.path_2 = nn.HybridSequential(prefix='')
        # No nn.ZeroPad2D in gluon
        self.path_2.add(nn.AvgPool2D(1, strides=2, count_include_pad=False))
        self.path_2.add(nn.Conv2D(out_channels_left, 1, strides=1, use_bias=False))

        self.final_path_bn = nn.BatchNorm(momentum=0.1, epsilon=0.001)

        self.comb_iter_0_left = BranchSeparables(out_channels_right, out_channels_right, 5, 1, 2)
        self.comb_iter_0_right = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1)

        self.comb_iter_1_left = BranchSeparables(out_channels_right, out_channels_right, 5, 1, 2)
        self.comb_iter_1_right = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1)

        self.comb_iter_2_left = nn.AvgPool2D(3, strides=1, padding=1, count_include_pad=False)

        self.comb_iter_3_left = nn.AvgPool2D(3, strides=1, padding=1, count_include_pad=False)
        self.comb_iter_3_right = nn.AvgPool2D(3, strides=1, padding=1, count_include_pad=False)

        self.comb_iter_4_left = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1)

    def hybrid_forward(self, F, x, x_prev):
        x_relu = F.Activation(x_prev, act_type='relu')
        x_path1 = self.path_1(x_relu)
        x_path2 = F.pad(x_relu, pad_width=(0, 0, 0, 0, 0, 1, 0, 1),
                        mode='constant', constant_value=0)
        x_path2 = F.slice(x_path2, begin=(0, 0, 1, 1), end=(None, None, None, None))
        x_path2 = self.path_2(x_path2)
        x_left = self.final_path_bn(F.concat(x_path1, x_path2, dim=1))

        x_right = self.conv_1x1(x)

        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_left

        x_comb_iter_3_left = self.comb_iter_3_left(x_left)
        x_comb_iter_3_right = self.comb_iter_3_right(x_left)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right

        x_comb_iter_4_left = self.comb_iter_4_left(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_right

        x_out = F.concat(x_left, x_comb_iter_0, x_comb_iter_1, x_comb_iter_2,
                         x_comb_iter_3, x_comb_iter_4, dim=1)

        return x_out, x


class NormalCell(HybridBlock):

    def __init__(self, out_channels_left, out_channels_right):
        super(NormalCell, self).__init__()

        self.conv_prev_1x1 = nn.HybridSequential(prefix='')
        self.conv_prev_1x1.add(nn.Activation('relu'))
        self.conv_prev_1x1.add(nn.Conv2D(out_channels_left, 1, strides=1, use_bias=False))
        self.conv_prev_1x1.add(nn.BatchNorm(momentum=0.1, epsilon=0.001))

        self.conv_1x1 = nn.HybridSequential(prefix='')
        self.conv_1x1.add(nn.Activation('relu'))
        self.conv_1x1.add(nn.Conv2D(out_channels_right, 1, strides=1, use_bias=False))
        self.conv_1x1.add(nn.BatchNorm(momentum=0.1, epsilon=0.001))

        self.comb_iter_0_left = BranchSeparables(out_channels_right, out_channels_right, 5, 1, 2)
        self.comb_iter_0_right = BranchSeparables(out_channels_left, out_channels_left, 3, 1, 1)

        self.comb_iter_1_left = BranchSeparables(out_channels_left, out_channels_left, 5, 1, 2)
        self.comb_iter_1_right = BranchSeparables(out_channels_left, out_channels_left, 3, 1, 1)

        self.comb_iter_2_left = nn.AvgPool2D(3, strides=1, padding=1, count_include_pad=False)

        self.comb_iter_3_left = nn.AvgPool2D(3, strides=1, padding=1, count_include_pad=False)
        self.comb_iter_3_right = nn.AvgPool2D(3, strides=1, padding=1, count_include_pad=False)

        self.comb_iter_4_left = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1)

    def hybrid_forward(self, F, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)

        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_left)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2 = x_comb_iter_2_left + x_left

        x_comb_iter_3_left = self.comb_iter_3_left(x_left)
        x_comb_iter_3_right = self.comb_iter_3_right(x_left)
        x_comb_iter_3 = x_comb_iter_3_left + x_comb_iter_3_right

        x_comb_iter_4_left = self.comb_iter_4_left(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_right

        x_out = F.concat(x_left, x_comb_iter_0, x_comb_iter_1, x_comb_iter_2,
                         x_comb_iter_3, x_comb_iter_4, dim=1)

        return x_out, x

class ReductionCell0(HybridBlock):

    def __init__(self, out_channels_left, out_channels_right):
        super(ReductionCell0, self).__init__()

        self.conv_prev_1x1 = nn.HybridSequential(prefix='')
        self.conv_prev_1x1.add(nn.Activation('relu'))
        self.conv_prev_1x1.add(nn.Conv2D(out_channels_left, 1, strides=1, use_bias=False))
        self.conv_prev_1x1.add(nn.BatchNorm(momentum=0.1, epsilon=0.001))

        self.conv_1x1 = nn.HybridSequential(prefix='')
        self.conv_1x1.add(nn.Activation('relu'))
        self.conv_1x1.add(nn.Conv2D(out_channels_right, 1, strides=1, use_bias=False))
        self.conv_1x1.add(nn.BatchNorm(momentum=0.1, epsilon=0.001))

        self.comb_iter_0_left = BranchSeparablesReduction(out_channels_right, out_channels_right,
                                                          5, 2, 2)
        self.comb_iter_0_right = BranchSeparablesReduction(out_channels_right, out_channels_right,
                                                           7, 2, 3)

        self.comb_iter_1_left = MaxPoolPad()
        self.comb_iter_1_right = BranchSeparablesReduction(out_channels_right, out_channels_right,
                                                           7, 2, 3)

        self.comb_iter_2_left = AvgPoolPad()
        self.comb_iter_2_right = BranchSeparablesReduction(out_channels_right, out_channels_right,
                                                           5, 2, 2)

        self.comb_iter_3_right = nn.AvgPool2D(3, strides=1, padding=1, count_include_pad=False)

        self.comb_iter_4_left = BranchSeparablesReduction(out_channels_right, out_channels_right,
                                                          3, 1, 1)
        self.comb_iter_4_right = MaxPoolPad()

    def hybrid_forward(self, F, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)

        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_left)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right

        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1

        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right

        x_out = F.concat(x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4, dim=1)

        return x_out, x


class ReductionCell1(HybridBlock):

    def __init__(self, out_channels_left, out_channels_right):
        super(ReductionCell1, self).__init__()

        self.conv_prev_1x1 = nn.HybridSequential(prefix='')
        self.conv_prev_1x1.add(nn.Activation('relu'))
        self.conv_prev_1x1.add(nn.Conv2D(out_channels_left, 1, strides=1, use_bias=False))
        self.conv_prev_1x1.add(nn.BatchNorm(momentum=0.1, epsilon=0.001))

        self.conv_1x1 = nn.HybridSequential(prefix='')
        self.conv_1x1.add(nn.Activation('relu'))
        self.conv_1x1.add(nn.Conv2D(out_channels_right, 1, strides=1, use_bias=False))
        self.conv_1x1.add(nn.BatchNorm(momentum=0.1, epsilon=0.001))

        self.comb_iter_0_left = BranchSeparables(out_channels_right, out_channels_right, 5, 2, 2)
        self.comb_iter_0_right = BranchSeparables(out_channels_right, out_channels_right, 7, 2, 3)

        self.comb_iter_1_left = nn.MaxPool2D(3, strides=2, padding=1)
        self.comb_iter_1_right = BranchSeparables(out_channels_right, out_channels_right, 7, 2, 3)

        self.comb_iter_2_left = nn.AvgPool2D(3, strides=2, padding=1, count_include_pad=False)
        self.comb_iter_2_right = BranchSeparables(out_channels_right, out_channels_right, 5, 2, 2)

        self.comb_iter_3_right = nn.AvgPool2D(3, strides=1, padding=1, count_include_pad=False)

        self.comb_iter_4_left = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1)
        self.comb_iter_4_right = nn.MaxPool2D(3, strides=2, padding=1)

    def hybrid_forward(self, F, x, x_prev):
        x_left = self.conv_prev_1x1(x_prev)
        x_right = self.conv_1x1(x)

        x_comb_iter_0_left = self.comb_iter_0_left(x_right)
        x_comb_iter_0_right = self.comb_iter_0_right(x_left)
        x_comb_iter_0 = x_comb_iter_0_left + x_comb_iter_0_right

        x_comb_iter_1_left = self.comb_iter_1_left(x_right)
        x_comb_iter_1_right = self.comb_iter_1_right(x_left)
        x_comb_iter_1 = x_comb_iter_1_left + x_comb_iter_1_right

        x_comb_iter_2_left = self.comb_iter_2_left(x_right)
        x_comb_iter_2_right = self.comb_iter_2_right(x_left)
        x_comb_iter_2 = x_comb_iter_2_left + x_comb_iter_2_right

        x_comb_iter_3_right = self.comb_iter_3_right(x_comb_iter_0)
        x_comb_iter_3 = x_comb_iter_3_right + x_comb_iter_1

        x_comb_iter_4_left = self.comb_iter_4_left(x_comb_iter_0)
        x_comb_iter_4_right = self.comb_iter_4_right(x_right)
        x_comb_iter_4 = x_comb_iter_4_left + x_comb_iter_4_right

        x_out = F.concat(x_comb_iter_1, x_comb_iter_2, x_comb_iter_3, x_comb_iter_4, dim=1)

        return x_out, x


class NASNetALarge(HybridBlock):
    r"""NASNet A model from
    `"Learning Transferable Architectures for Scalable Image Recognition"
    <https://arxiv.org/abs/1707.07012>`_ paper

    Parameters
    ----------
    repeat : int
        Number of cell repeats
    penultimate_filters : int
        Number of filters in the penultimate layer of the network
    stem_filters : int
        Number of filters in stem layers
    filters_multiplier : int
        The filter multiplier for stem layers
    classes: int, default 1000
        Number of classification classes
    use_aux : bool, default True
        Whether to use auxiliary classifier when training
    """
    def __init__(self, repeat=6, penultimate_filters=4032, stem_filters=96,
                 filters_multiplier=2, classes=1000, use_aux=True):
        super(NASNetALarge, self).__init__()

        filters = penultimate_filters // 24

        self.conv0 = nn.HybridSequential(prefix='')
        self.conv0.add(nn.Conv2D(stem_filters, 3, padding=0, strides=2, use_bias=False))
        self.conv0.add(nn.BatchNorm(momentum=0.1, epsilon=0.001))

        self.cell_stem_0 = CellStem0(stem_filters, num_filters=filters // (filters_multiplier ** 2))
        self.cell_stem_1 = CellStem1(num_filters=filters // filters_multiplier)

        self.norm_1 = nn.HybridSequential(prefix='')
        self.norm_1.add(FirstCell(out_channels_left=filters//2, out_channels_right=filters))
        for _ in range(repeat - 1):
            self.norm_1.add(NormalCell(out_channels_left=filters, out_channels_right=filters))

        self.reduction_cell_0 = ReductionCell0(out_channels_left=2*filters,
                                               out_channels_right=2*filters)

        self.norm_2 = nn.HybridSequential(prefix='')
        self.norm_2.add(FirstCell(out_channels_left=filters, out_channels_right=2*filters))
        for _ in range(repeat - 1):
            self.norm_2.add(NormalCell(out_channels_left=2*filters, out_channels_right=2*filters))

        if use_aux:
            self.out_aux = nn.HybridSequential(prefix='')
            self.out_aux.add(nn.Conv2D(filters // 3, kernel_size=1, use_bias=False))
            self.out_aux.add(nn.BatchNorm(epsilon=0.001))
            self.out_aux.add(nn.Activation('relu'))
            self.out_aux.add(nn.Conv2D(2*filters, kernel_size=5, use_bias=False))
            self.out_aux.add(nn.BatchNorm(epsilon=0.001))
            self.out_aux.add(nn.Activation('relu'))
            self.out_aux.add(nn.Dense(classes))
        else:
            self.out_aux = None

        self.reduction_cell_1 = ReductionCell1(out_channels_left=4*filters,
                                               out_channels_right=4*filters)

        self.norm_3 = nn.HybridSequential(prefix='')
        self.norm_3.add(FirstCell(out_channels_left=2*filters, out_channels_right=4*filters))
        for _ in range(repeat - 1):
            self.norm_3.add(NormalCell(out_channels_left=4*filters, out_channels_right=4*filters))

        self.out = nn.HybridSequential(prefix='')
        self.out.add(nn.Activation('relu'))
        self.out.add(nn.GlobalAvgPool2D())
        self.out.add(nn.Dropout(0.5))
        self.out.add(nn.Dense(classes))

    def hybrid_forward(self, F, x):
        x_conv0 = self.conv0(x)
        x_stem_0 = self.cell_stem_0(x_conv0)
        x_stem_1 = self.cell_stem_1(x_conv0, x_stem_0)

        x = x_stem_1
        x_prev = x_stem_0
        for cell in self.norm_1._children.values():
            x, x_prev = cell(x, x_prev)
        x, x_prev = self.reduction_cell_0(x, x_prev)
        for cell in self.norm_2._children.values():
            x, x_prev = cell(x, x_prev)
        if self.out_aux:
            x_aux = F.contrib.AdaptiveAvgPooling2D(x, output_size=5)
            x_aux = self.out_aux(x_aux)
        x, x_prev = self.reduction_cell_1(x, x_prev)
        for cell in self.norm_3._children.values():
            x, x_prev = cell(x, x_prev)

        x = self.out(x)
        if self.out_aux:
            return x, x_aux
        else:
            return x

def get_nasnet(repeat=6, penultimate_filters=4032,
               pretrained=False, ctx=cpu(),
               root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    r"""NASNet A model from
    `"Learning Transferable Architectures for Scalable Image Recognition"
    <https://arxiv.org/abs/1707.07012>`_ paper

    Parameters
    ----------
    repeat : int
        Number of cell repeats
    penultimate_filters : int
        Number of filters in the penultimate layer of the network
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    assert repeat >= 2, \
        "Invalid number of repeat: %d. It should be at least two"%(repeat)
    net = NASNetALarge(repeat=repeat, penultimate_filters=penultimate_filters, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        net.load_params(get_model_file('nasnet_%d_%d'%(repeat, penultimate_filters),
                                       root=root), ctx=ctx)
    return net

def nasnet_4_1056(**kwargs):
    r"""NASNet A model from
    `"Learning Transferable Architectures for Scalable Image Recognition"
    <https://arxiv.org/abs/1707.07012>`_ paper

    Parameters
    ----------
    repeat : int
        Number of cell repeats
    penultimate_filters : int
        Number of filters in the penultimate layer of the network
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_nasnet(repeat=4, penultimate_filters=1056, **kwargs)

def nasnet_5_1538(**kwargs):
    r"""NASNet A model from
    `"Learning Transferable Architectures for Scalable Image Recognition"
    <https://arxiv.org/abs/1707.07012>`_ paper

    Parameters
    ----------
    repeat : int
        Number of cell repeats
    penultimate_filters : int
        Number of filters in the penultimate layer of the network
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_nasnet(repeat=5, penultimate_filters=1538, **kwargs)


def nasnet_7_1920(**kwargs):
    r"""NASNet A model from
    `"Learning Transferable Architectures for Scalable Image Recognition"
    <https://arxiv.org/abs/1707.07012>`_ paper

    Parameters
    ----------
    repeat : int
        Number of cell repeats
    penultimate_filters : int
        Number of filters in the penultimate layer of the network
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_nasnet(repeat=7, penultimate_filters=1920, **kwargs)


def nasnet_6_4032(**kwargs):
    r"""NASNet A model from
    `"Learning Transferable Architectures for Scalable Image Recognition"
    <https://arxiv.org/abs/1707.07012>`_ paper

    Parameters
    ----------
    repeat : int
        Number of cell repeats
    penultimate_filters : int
        Number of filters in the penultimate layer of the network
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_nasnet(repeat=6, penultimate_filters=4032, **kwargs)
