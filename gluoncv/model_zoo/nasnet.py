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
# pylint: disable= arguments-differ,missing-docstring
"""NASNet, implemented in Gluon."""
from __future__ import division

__all__ = ['NASNet', 'get_nasnet']

import os
import math
from mxnet import cpu
from mxnet.gluon import nn
from mxnet.gluon.block import HybridBlock

class MaxPoolPad(HybridBlock):

    def __init(self):
        super(MaxPoolPad, self).__init__()
        self.pool = nn.MaxPool2D(3, strides=2, padding=1)

    def hybrid_forward(self, F, x):
        x = F.pad(x, pad_width = (0,0,0,0,1,0,1,0), 
                  mode='constant', constant_value=0)
        x = self.pool(x)
        x = F.slice(x, begin=(0,0,1,1), end=(None, None, None, None))
        return x

class AvgPoolPad(HybridBlock):

    def __init(self, stride=2, padding=1):
        super(MaxPoolPad, self).__init__()
        # There's no 'count_include_pad' parameter, which makes it different
        self.pool = nn.AvgPool2D(3, strides=stride, padding=padding)

    def hybrid_forward(self, F, x):
        x = F.pad(x, pad_width = (0,0,0,0,1,0,1,0), 
                  mode='constant', constant_value=0)
        x = self.pool(x)
        x = F.slice(x, begin=(0,0,1,1), end=(None, None, None, None))
        return x

class SeparableConv2d(HybridBlock):

    def __init__(self, in_channels, channels, dw_kernel, dw_stride, dw_padding,
                 use_bias=False):
       super(SeparableConv2d, self).__init__()
       self.body = nn.HybridSequential(prefix='')
       self.body.add(nn.Conv2D(in_channels, kernel_size=dw_kernel,
                               strides=dw_stride, padding=dw_padding,
                               use_bias=use_bias,
                               groups=in_channels)
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
                                     strides=1, padding=padding, use_bias=use_bias))
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
                                     strides=1, padding=padding, use_bias=use_bias))
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
                                         strides, padding, use_bias=use_bias))

        self.body = nn.HybridSequential(prefix='')
        self.body.add(nn.BatchNorm(momentum=0.1, epsilon=0.001))
        self.body.add(nn.Activation('relu'))
        self.body.add(SeparableConv2d(in_channels, out_channels, kernel_size,
                                      strides=1, padding=padding, use_bias=use_bias))
        self.body.add(nn.BatchNorm(momentum=0.1, epsilon=0.001))

    def hybrid_forward(self, F, x):
        x = F.Activation(x, act_type='relu')
        x = F.pad(x, pad_width = (0,0,0,0,self.z_padding,0,self.z_padding,0),
                  mode='constant', constant_value=0)
        x = self.separable(x)
        x = F.slice(x, begin=(0,0,1,1), end=(None, None, None, None))
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

        self.comb_iter_2_left = nn.AvgPool2D(3, strides=2, padding=1)
        self.comb_iter_2_right = BranchSeparablesStem(stem_filters, num_filters, 5, 2, 2)

        self.comb_iter_3_right = nn.AvgPool2D(3, strides=1, padding=1)

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

    def __init__(self, stem_filters, num_filters):
        super(CellStem1, self).__init__()

        self.conv_1x1 = nn.HybridSequential(prefix='')
        self.conv_1x1.add(nn.Activation('relu'))
        self.conv_1x1.add(nn.Conv2D(num_filters, 1, strides=1, use_bias=False))
        self.conv_1x1.add(nn.BatchNorm(momentum=0.1, epsilon=0.001))

        self.path_1 = nn.HybridSequential(prefix='')
        self.path_1.add(nn.AvgPool2D(1, strides=2))
        self.path_1.add(nn.Conv2D(num_filters//2, 1, strides=1, use_bias=False))

        self.path_2 = nn.HybridSequential(prefix='')
        # No nn.ZeroPad2D in gluon
        self.path_2.add(nn.AvgPool2D(1, strides=2))
        self.path_2.add(nn.Conv2D(num_filters//2, 1, strides=1, use_bias=False))

        self.final_path_bn = nn.BatchNorm(momentum=0.1, epsilon=0.001)

        self.comb_iter_0_left = BranchSeparables(num_filters, num_filters, 5, 2, 2)
        self.comb_iter_0_right = BranchSeparables(stem_filters, num_filters, 7, 2, 3)

        self.comb_iter_1_left = nn.MaxPool2D(3, strides=2, padding=1)
        self.comb_iter_1_right = BranchSeparables(stem_filters, num_filters, 7, 2, 3)

        self.comb_iter_2_left = nn.AvgPool2D(3, strides=2, padding=1)
        self.comb_iter_2_right = BranchSeparables(stem_filters, num_filters, 5, 2, 2)

        self.comb_iter_3_right = nn.AvgPool2D(3, strides=1, padding=1)

        self.comb_iter_4_left = BranchSeparables(num_filters, num_filters, 3, 1, 1)
        self.comb_iter_4_right = nn.MaxPool2D(3, strides=2, padding=1)

    def hybrid_forward(self, F, x_conv0, x_stem_0):
        x_left = self.conv_1x1(x_stem_0)

        x_relu = F.Activation(x_conv0, act_type='relu')
        x_path1 = self.path_1(x_relu)
        x_path2 = F.pad(x_relu, pad_width = (0,0,0,0,0,1,0,1), 
                  mode='constant', constant_value=0)
        x_path2 = F.slice(x, begin=(0,0,1,1), end=(None, None, None, None))
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

    def __init__(self, in_channels_left, out_channels_left,
                 in_channels_right, out_channels_right):
        super(FirstCell, self).__init__()

        self.conv_1x1 = nn.HybridSequential(prefix='')
        self.conv_1x1.add(nn.Activation('relu'))
        self.conv_1x1.add(nn.Conv2D(out_channels_right, 1, strides=1, use_bias=False))
        self.conv_1x1.add(nn.BatchNorm(momentum=0.1, epsilon=0.001))

        self.path_1 = nn.HybridSequential(prefix='')
        self.path_1.add(nn.AvgPool2D(1, strides=2))
        self.path_1.add(nn.Conv2D(out_channels_left, 1, strides=1, use_bias=False))

        self.path_2 = nn.HybridSequential(prefix='')
        # No nn.ZeroPad2D in gluon
        self.path_2.add(nn.AvgPool2D(1, strides=2))
        self.path_2.add(nn.Conv2D(out_channels_left, 1, strides=1, use_bias=False))

        self.final_path_bn = nn.BatchNorm(momentum=0.1, epsilon=0.001)

        self.comb_iter_0_left = BranchSeparables(out_channels_right, out_channels_right, 5, 1, 2)
        self.comb_iter_0_right = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1)

        self.comb_iter_1_left = BranchSeparables(out_channels_right, out_channels_right, 5, 1, 2)
        self.comb_iter_1_right = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1)

        self.comb_iter_2_left = nn.AvgPool2D(3, strides=1, padding=1)

        self.comb_iter_3_left = nn.AvgPool2D(3, strides=1, padding=1)
        self.comb_iter_3_right = nn.AvgPool2D(3, strides=1, padding=1)

        self.comb_iter_4_left = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1)

    def hybrid_forward(self, F, x_list):
        x = x_list[0]
        x_prev = x_list[1]

        x_relu = F.Activation(x_prev, act_type='relu')
        x_path1 = self.path_1(x_relu)
        x_path2 = F.pad(x_relu, pad_width = (0,0,0,0,0,1,0,1), 
                  mode='constant', constant_value=0)
        x_path2 = F.slice(x, begin=(0,0,1,1), end=(None, None, None, None))
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

        return [x_out, x]


class NormalCell(HybridBlock):

    def __init__(self, in_channels_left, out_channels_left,
                 in_channels_right, out_channels_right):
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

        self.comb_iter_2_left = nn.AvgPool2D(3, strides=1, padding=1)

        self.comb_iter_3_left = nn.AvgPool2D(3, strides=1, padding=1)
        self.comb_iter_3_right = nn.AvgPool2D(3, strides=1, padding=1)

        self.comb_iter_4_left = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1)

    def hybrid_forward(self, F, x_list):
        x = x_list[0]
        x_prev = x_list[1]

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

        return [x_out, x]

class ReductionCell0(HybridBlock):

    def __init__(self, in_channels_left, out_channels_left,
                 in_channels_right, out_channels_right):
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

        self.comb_iter_3_right = nn.AvgPool2D(3, strides=1, padding=1)

        self.comb_iter_4_left = BranchSeparablesReduction(out_channels_right, out_channels_right,
                                                          3, 1, 1)
        self.comb_iter_4_left = MaxPoolPad()

    def hybrid_forward(self, F, x_list):
        x = x_list[0]
        x_prev = x_list[1]

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

        return [x_out, x]


class ReductionCell1(HybridBlock):

    def __init__(self, in_channels_left, out_channels_left,
                 in_channels_right, out_channels_right):
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

        self.comb_iter_1_left = MaxPoolPad(3, strides=2, padding=1)
        self.comb_iter_1_right = BranchSeparables(out_channels_right, out_channels_right, 7, 2, 3)

        self.comb_iter_2_left = AvgPoolPad(3, strides=2, padding=1)
        self.comb_iter_2_right = BranchSeparables(out_channels_right, out_channels_right, 5, 2, 2)

        self.comb_iter_3_right = nn.AvgPool2D(3, strides=1, padding=1)

        self.comb_iter_4_left = BranchSeparables(out_channels_right, out_channels_right, 3, 1, 1)
        self.comb_iter_4_left = MaxPoolPad(3, stride=2, padding=1)

    def hybrid_forward(self, F, x_list):
        x = x_list[0]
        x_prev = x_list[1]

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

        return [x_out, x]


class NASNetALarge(HybridBlock):

    def __init__(self, num_classes, stem_filters=96, penultimate_filters=4032, filters_multiplier=2):
        super(NASNetALarge, self).__init__()

        filters = penultimate_filters // 24

        self.conv0 = nn.HybridSequential(prefix='')
        self.conv0.add(nn.Conv2D(steam_filters, 3, padding=0, strides=2, use_bias=False))
        self.conv0.add(nn.BatchNorm(momentum=0.1, epsilon=0.001))

        self.cell_stem_0 = CellStem0(stem_filters, num_filters=filters // (filters_multiplier ** 2))
        self.cell_stem_1 = CellStem1(stem_filters, num_filters=filters // filters_multiplier)

        self.norm_1 = nn.HybridSequential(prefix='')
        self.norm_1.add(FirstCell(in_channels_left=filters, out_channels_left=filters//2,
                                  in_channels_right=2*filters, out_channels_right=filters))
        self.norm_1.add(NormalCell(in_channels_left=2*filters, out_channels_left=filters,
                                   in_channels_right=6*filters, out_channels_right=filters))
        for i in range(4):
            self.norm_1.add(NormalCell(in_channels_left=6*filters, out_channels_left=filters,
                                       in_channels_right=6*filters, out_channels_right=filters))

        self.reduction_cell_0 = ReductionCell0(in_channels_left=6*filters, out_channels_left=2*filters,
                                               in_channels_right=6*filters, out_channels_right=2*filters)

        self.norm_2 = nn.HybridSequential(prefix='')
        self.norm_2.add(FirstCell(in_channels_left=6*filters, out_channels_left=filters,
                                  in_channels_right=8*filters, out_channels_right=2*filters))
        self.norm_2.add(NormalCell(in_channels_left=8*filters, out_channels_left=2*filters,
                                   in_channels_right=12*filters, out_channels_right=2*filters))
        for i in range(4):
            self.norm_2.add(NormalCell(in_channels_left=12*filters, out_channels_left=2*filters,
                                       in_channels_right=12*filters, out_channels_right=2*filters))

        self.reduction_cell_1 = ReductionCell1(in_channels_left=12*filters, out_channels_left=4*filters,
                                               in_channels_right=12*filters, out_channels_right=4*filters)

        self.norm_3 = nn.HybridSequential(prefix='')
        self.norm_3.add(FirstCell(in_channels_left=12*filters, out_channels_left=2*filters,
                                  in_channels_right=16*filters, out_channels_right=4*filters))
        self.norm_3.add(NormalCell(in_channels_left=16*filters, out_channels_left=4*filters,
                                   in_channels_right=24*filters, out_channels_right=4*filters))
        for i in range(4):
            self.norm_3.add(NormalCell(in_channels_left=24*filters, out_channels_left=4*filters,
                                       in_channels_right=24*filters, out_channels_right=4*filters))

        self.out = nn.HybridSequential(prefix='')
        self.out.add(nn.Activation('relu'))
        self.out.add(nn.AvgPool2D(11, strides=1, padding=0))
        self.out.add(nn.Dropout())
        self.out.add(nn.Dense(num_classes))

    def hybrid_forward(self, F, x):
        x_conv0 = self.conv0(x)
        x_stem_0 = self.cell_stem_0(x_conv0)
        x_stem_1 = self.cell_stem_1(x_conv0, x_stem_0)

        x_norm_1 = self.norm_1([x_stem_1, x_stem_0])
        x_reduction_cell_0 = self.reduction_cell_0(x_norm_1)
        x_norm_2 = self.norm_2(x_reduction_cell_0)
        x_reduction_cell_1 = self.reduction_cell_1(x_norm_2)
        x_norm_3 = self.norm_3(x_reduction_cell_1)

        x = self.out(x_norm_3[0])
        return x

def nasnetalarge(num_classes=1001)
    model = NASNetALarge(num_classes=num_classes)
    return model
