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
# pylint: disable= arguments-differ,unused-argument,unused-variable,missing-docstring
"""HRNet, implemented in Gluon."""
__all__ = ['get_hrnet', 'hrnet_w18_c', 'hrnet_w18_small_v1_c', 'hrnet_w18_small_v2_c',
           'hrnet_w30_c', 'hrnet_w32_c', 'hrnet_w40_c', 'hrnet_w44_c', 'hrnet_w48_c',
           'hrnet_w64_c', 'hrnet_w18_small_v1_s', 'hrnet_w18_small_v2_s', 'hrnet_w48_s']

import numpy as np
from mxnet.gluon import contrib
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm
from mxnet.context import cpu
from .resnet import BasicBlockV1, BottleneckV1, _conv3x3

class HRBasicBlock(BasicBlockV1):
    r"""BasicBlock V1 from `"Deep Residual Learning for Image Recognition"
    """
    expansion = 1

class HRBottleneck(BottleneckV1):
    '''
    warning: It's mxnet compatable bottleneck, the orginal implementation is different
    from this bottleneck as its all convolutions are no bias
    '''
    expansion = 4

class OrigHRBottleneck(nn.HybridBlock):
    r"""Modified Bottleneck V1 from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    This is used for ResNet V1 for 50, 101, 152 layers. Its all convolutions are
    no bias to match with the original hrnet implementation.

    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    last_gamma : bool, default False
        Whether to initialize the gamma of the last BatchNorm layer in each bottleneck to zero.
    use_se : bool, default False
        Whether to use Squeeze-and-Excitation module
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    expansion = 4
    def __init__(self, channels, stride, downsample=False, in_channels=0,
                 last_gamma=False, use_se=False, norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(OrigHRBottleneck, self).__init__(**kwargs)
        self.body = nn.HybridSequential(prefix='')
        # add use_bias=False here to match with the original implementation
        self.body.add(nn.Conv2D(channels//4, kernel_size=1, strides=1, use_bias=False))
        self.body.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        self.body.add(nn.Activation('relu'))
        self.body.add(_conv3x3(channels//4, stride, channels//4))
        self.body.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        self.body.add(nn.Activation('relu'))
        # add use_bias=False here to match with the original implementation
        self.body.add(nn.Conv2D(channels, kernel_size=1, strides=1, use_bias=False))

        if use_se:
            self.se = nn.HybridSequential(prefix='')
            self.se.add(nn.Dense(channels // 16, use_bias=False))
            self.se.add(nn.Activation('relu'))
            self.se.add(nn.Dense(channels, use_bias=False))
            self.se.add(nn.Activation('sigmoid'))
        else:
            self.se = None

        if not last_gamma:
            self.body.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        else:
            self.body.add(norm_layer(gamma_initializer='zeros',
                                     **({} if norm_kwargs is None else norm_kwargs)))

        if downsample:
            self.downsample = nn.HybridSequential(prefix='')
            self.downsample.add(nn.Conv2D(channels, kernel_size=1, strides=stride,
                                          use_bias=False, in_channels=in_channels))
            self.downsample.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x

        x = self.body(x)

        if self.se:
            w = F.contrib.AdaptiveAvgPooling2D(x, output_size=1)
            w = self.se(w)
            x = F.broadcast_mul(x, w.expand_dims(axis=2).expand_dims(axis=2))

        if self.downsample:
            residual = self.downsample(residual)

        x = F.Activation(x + residual, act_type='relu')
        return x

class HighResolutionModule(nn.HybridBlock):
    '''
    interp_type can be 'nearest'/'bilinear'/'bilinear_like'
    '''
    def __init__(self, num_branches, blocks, num_blocks, num_channels, fuse_method,
                 num_inchannels=None, multi_scale_output=True, interp_type='nearest',
                 norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(HighResolutionModule, self).__init__(**kwargs)

        if num_inchannels is not None:
            self.num_inchannels = num_inchannels
        else:
            self.num_inchannels = num_channels

        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output
        self.interp_type = interp_type

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers(norm_layer=norm_layer, norm_kwargs=norm_kwargs)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = stride != 1 or self.num_inchannels[branch_index] != \
            num_channels[branch_index] * block.expansion

        layers = nn.HybridSequential()
        layers.add(block(num_channels[branch_index]* block.expansion, stride,
                         downsample, self.num_inchannels[branch_index]))
        self.num_inchannels[branch_index] = \
                num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.add(block(num_channels[branch_index]* block.expansion, 1,
                             False, self.num_inchannels[branch_index]))

        return layers


    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = nn.HybridSequential()

        for i in range(num_branches):
            branches.add(
                self._make_one_branch(i, block, num_blocks, num_channels)
            )
        return branches

    def _make_fuse_layers(self, norm_layer=BatchNorm, norm_kwargs=None):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = nn.HybridSequential()
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = nn.HybridSequential()
            for j in range(num_branches):
                if j > i:
                    seq = nn.HybridSequential()

                    seq.add(
                        nn.Conv2D(num_inchannels[i], 1, 1, 0, use_bias=False),
                        norm_layer(**({} if norm_kwargs is None else norm_kwargs))
                    )
                    fuse_layer.add(seq)
                elif j == i:
                    fuse_layer.add(contrib.nn.Identity())
                else:
                    conv3x3s = nn.HybridSequential()
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.add(
                                nn.Conv2D(num_outchannels_conv3x3, 3, 2, 1, use_bias=False),
                                norm_layer(**({} if norm_kwargs is None else norm_kwargs))
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.add(
                                nn.Conv2D(num_outchannels_conv3x3, 3, 2, 1, use_bias=False),
                                norm_layer(**({} if norm_kwargs is None else norm_kwargs)),
                                nn.Activation('relu')
                            )
                    fuse_layer.add(conv3x3s)
            fuse_layers.add(fuse_layer)

        return fuse_layers

    def get_num_inchannels(self):
        return self.num_inchannels

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.branches[0](x)
        if self.num_branches == 1:
            return [x]

        X = []
        X.append(x)

        for i in range(1, self.num_branches):
            X.append(self.branches[i](args[i-1]))

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = X[0] if i == 0 else self.fuse_layers[i][0](X[0])
            for j in range(1, self.num_branches):
                if j > i:
                    if self.interp_type == 'nearest':
                        y = F.broadcast_add(y, F.UpSampling(
                            self.fuse_layers[i][j](X[j]),
                            scale=2**(j-i),
                            sample_type='nearest'))
                    elif self.interp_type == 'bilinear':
                        y = F.broadcast_add(y, F.contrib.BilinearResize2D(
                            self.fuse_layers[i][j](X[j]),
                            scale_height=2**(j-i),
                            scale_width=2**(j-i),
                            align_corners=False
                        ))
                    elif self.interp_type == 'bilinear_like':
                        y = F.broadcast_add(y, F.contrib.BilinearResize2D(
                            self.fuse_layers[i][j](X[j]),
                            like=X[i],
                            mode='like',
                            align_corners=False
                        ))
                    else:
                        raise NotImplementedError

                else:
                    y = y + self.fuse_layers[i][j](X[j])
            x_fuse.append(F.relu(y))

        return x_fuse

# TODO: Now, We use OrigHRBottleneck to match with the origial implementation. You
#       can also replace it with the mxnet compatable HRBottleneck.
BLOCKS_DICT = {
    'BASIC': HRBasicBlock,
    'BOTTLENECK': OrigHRBottleneck
}

class HighResolutionBaseNet(nn.HybridBlock):
    r"""Base class for classification and segmentation
    """
    def __init__(self, cfg, stage_interp_type='nearst', norm_layer=BatchNorm, \
                 norm_kwargs=None, **kwargs):
        self.stage_interp_type = stage_interp_type
        super(HighResolutionBaseNet, self).__init__()

        self.conv1 = nn.Conv2D(64, kernel_size=3, strides=2, padding=1,
                               use_bias=False)
        self.bn1 = norm_layer(**({} if norm_kwargs is None else norm_kwargs))
        self.conv2 = nn.Conv2D(64, kernel_size=3, strides=2, padding=1,
                               use_bias=False)
        self.bn2 = norm_layer(**({} if norm_kwargs is None else norm_kwargs))

        self.stage1_cfg = cfg[0]
        num_channels = self.stage1_cfg[3][0]
        block = BLOCKS_DICT[self.stage1_cfg[1]]
        num_blocks = self.stage1_cfg[2][0]
        self.layer1 = self._make_layer(block, num_channels, num_blocks, inplanes=64)
        stage1_out_channel = block.expansion*num_channels

        self.stage2_cfg = cfg[1]
        num_channels = self.stage2_cfg[3]
        block = BLOCKS_DICT[self.stage2_cfg[1]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels, norm_layer, norm_kwargs)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = cfg[2]
        num_channels = self.stage3_cfg[3]
        block = BLOCKS_DICT[self.stage3_cfg[1]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels, norm_layer, norm_kwargs)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = cfg[3]
        num_channels = self.stage4_cfg[3]
        block = BLOCKS_DICT[self.stage4_cfg[1]]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels, norm_layer, norm_kwargs)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=True)

        self.pre_stage_channels = pre_stage_channels

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer,
                               norm_layer=BatchNorm, norm_kwargs=None):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = nn.HybridSequential()

        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layer = nn.HybridSequential()
                    transition_layer.add(
                        nn.Conv2D(num_channels_cur_layer[i], 3, 1, 1,
                                  use_bias=False, in_channels=num_channels_pre_layer[i]),
                        norm_layer(**({} if norm_kwargs is None else norm_kwargs)),
                        nn.Activation('relu')
                    )
                    transition_layers.add(transition_layer)
                else:
                    transition_layers.add(contrib.nn.Identity())
            else:
                conv3x3s = nn.HybridSequential()
                for j in range(i+1-num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    cba = nn.HybridSequential()
                    cba.add(
                        nn.Conv2D(outchannels, 3, 2, 1,
                                  use_bias=False, in_channels=inchannels),
                        norm_layer(**({} if norm_kwargs is None else norm_kwargs)),
                        nn.Activation('relu')
                    )
                    conv3x3s.add(cba)
                transition_layers.add(conv3x3s)

        return transition_layers

    def _make_layer(self, block, planes, blocks, inplanes=0, stride=1):
        downsample = stride != 1 or inplanes != planes * block.expansion

        layers = nn.HybridSequential()
        layers.add(block(planes* block.expansion, stride, downsample, inplanes))
        for i in range(1, blocks):
            layers.add(block(planes* block.expansion, 1, False, inplanes))

        return layers

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config[0]
        num_blocks = layer_config[2]
        num_branches = len(num_blocks)
        num_channels = layer_config[3]
        block = BLOCKS_DICT[layer_config[1]]
        fuse_method = layer_config[4]

        blocks = nn.HybridSequential()
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            hrm = HighResolutionModule(num_branches,
                                       block,
                                       num_blocks,
                                       num_channels,
                                       fuse_method,
                                       num_inchannels,
                                       reset_multi_scale_output,
                                       self.stage_interp_type)
            blocks.add(hrm)
            num_inchannels = hrm.get_num_inchannels()

        return blocks, num_inchannels

    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.layer1(x)
        x_list = []

        for i in range(len(self.stage2_cfg[2])):
            x_list.append(self.transition1[i](x))

        y_list = x_list
        for s in self.stage2:
            y_list = s(*y_list)

        x_list = []
        for i in range(len(self.stage3_cfg[2])):
            if i < len(self.stage2_cfg[2]):
                x_list.append(self.transition2[i](y_list[i]))
            else:
                x_list.append(self.transition2[i](y_list[-1]))

        y_list = x_list
        for s in self.stage3:
            y_list = s(*y_list)

        x_list = []
        for i in range(len(self.stage4_cfg[2])):
            if i < len(self.stage3_cfg[2]):
                x_list.append(self.transition3[i](y_list[i]))
            else:
                x_list.append(self.transition3[i](y_list[-1]))

        y_list = x_list
        for s in self.stage4:
            y_list = s(*y_list)

        return y_list

class HighResolutionClsNet(HighResolutionBaseNet):
    r"""HRNet for Classification
    """
    def __init__(self, config, stage_interp_type='nearest', norm_layer=BatchNorm, norm_kwargs=None,
                 num_classes=1000, **kwargs):
        super(HighResolutionClsNet, self).__init__(config, stage_interp_type=stage_interp_type,
                                                   norm_layer=norm_layer, norm_kwargs=norm_kwargs)

        # Classification Head
        self.incre_blocks, self.downsamp_blocks, \
            self.final_layer = self._make_head(self.pre_stage_channels, norm_layer, norm_kwargs)

        self.avg = nn.GlobalAvgPool2D()

        self.classifier = nn.Dense(num_classes, in_units=2048)

    def hybrid_forward(self, F, x):
        y_list = super(HighResolutionClsNet, self).hybrid_forward(F, x)

        # Classification Head
        y = self.incre_blocks[0](y_list[0])
        for i in range(len(self.downsamp_blocks)):
            y = self.incre_blocks[i+1](y_list[i+1]) + \
                        self.downsamp_blocks[i](y)

        y = self.final_layer(y)
        y = self.avg(y)
        y = self.classifier(y)

        return y


    def _make_head(self, pre_stage_channels, norm_layer=BatchNorm, norm_kwargs=None):
        head_block = BLOCKS_DICT['BOTTLENECK']
        head_channels = [32, 64, 128, 256]

        incre_blocks = nn.HybridSequential()
        for i, channels in enumerate(pre_stage_channels):
            incre_block = self._make_layer(head_block,
                                           head_channels[i],
                                           1,
                                           channels,
                                           stride=1)
            incre_blocks.add(incre_block)

        downsamp_blocks = nn.HybridSequential()
        for i in range(len(pre_stage_channels)-1):
            in_channels = head_channels[i] * head_block.expansion
            out_channels = head_channels[i+1] * head_block.expansion

            downsamp_block = nn.HybridSequential()
            downsamp_block.add(
                nn.Conv2D(out_channels, 3, 2, 1, in_channels=in_channels),
                norm_layer(**({} if norm_kwargs is None else norm_kwargs)),
                nn.Activation('relu')
            )
            downsamp_blocks.add(downsamp_block)

        final_layer = nn.HybridSequential()
        final_layer.add(
            nn.Conv2D(2048, 1, 1, 0, in_channels=head_channels[3] * head_block.expansion),
            norm_layer(**({} if norm_kwargs is None else norm_kwargs)),
            nn.Activation('relu')
        )

        return incre_blocks, downsamp_blocks, final_layer

class HighResolutionSegNet(HighResolutionBaseNet):
    r"""HRNet for Segmentation
    """
    def __init__(self, config, stage_interp_type='bilinear_like', norm_layer=BatchNorm,
                 norm_kwargs=None, num_classes=19, **kwargs):
        super(HighResolutionSegNet, self).__init__(config, stage_interp_type=stage_interp_type,
                                                   norm_layer=norm_layer, norm_kwargs=norm_kwargs)

        self.last_layer = self._make_head(self.pre_stage_channels,
                                          norm_layer=norm_layer,
                                          norm_kwargs=norm_kwargs,
                                          num_classes=num_classes,
                                          **kwargs)

    def hybrid_forward(self, F, x):
        y_list = super(HighResolutionSegNet, self).hybrid_forward(F, x)

        # Upsampling
        x1 = F.contrib.BilinearResize2D(y_list[1], like=y_list[0], mode='like', align_corners=False)
        x2 = F.contrib.BilinearResize2D(y_list[2], like=y_list[0], mode='like', align_corners=False)
        x3 = F.contrib.BilinearResize2D(y_list[3], like=y_list[0], mode='like', align_corners=False)

        y = F.concat(y_list[0], x1, x2, x3, dim=1)

        y = self.last_layer(y)

        return y

    def _make_head(self, pre_stage_channels, norm_layer=BatchNorm, norm_kwargs=None,
                   num_classes=19, final_conv_kernel=1):
        last_inp_channels = np.int(np.sum(pre_stage_channels))

        last_layer = nn.HybridSequential()
        last_layer.add(
            nn.Conv2D(
                last_inp_channels,
                1,
                1,
                0,
                in_channels=last_inp_channels),
            norm_layer(**({} if norm_kwargs is None else norm_kwargs)),
            nn.Activation('relu'),
            nn.Conv2D(
                num_classes,
                final_conv_kernel,
                1,
                padding=1 if final_conv_kernel == 3 else 0)
        )
        return last_layer

# pylint: disable=C0326
HRNET_SPEC = {}
HRNET_SPEC['w18'] = [
    #modules, block_type, blocks, channels, fuse_method
    (1,    'BOTTLENECK', [4],           [64],              'SUM'),
    (1,    'BASIC',      [4]*2,         [18, 36],          'SUM'),
    (4,    'BASIC',      [4]*3,         [18, 36, 72],      'SUM'),
    (3,    'BASIC',      [4]*4,         [18, 36, 72, 144], 'SUM')
]

HRNET_SPEC['w18_small_v1'] = [
    #modules, block_type, blocks, channels, fuse_method
    (1,    'BOTTLENECK', [1],           [32],              'SUM'),
    (1,    'BASIC',      [2]*2,         [16, 32],          'SUM'),
    (1,    'BASIC',      [2]*3,         [16, 32, 64],      'SUM'),
    (1,    'BASIC',      [2]*4,         [16, 32, 64, 128], 'SUM')
]

HRNET_SPEC['w18_small_v2'] = [
    #modules, block_type, blocks, channels, fuse_method
    (1,    'BOTTLENECK', [2],           [64],               'SUM'),
    (1,    'BASIC',      [2]*2,         [18, 36],           'SUM'),
    (3,    'BASIC',      [2]*3,         [18, 36, 72],       'SUM'),
    (2,    'BASIC',      [2]*4,         [18, 36, 72, 144],  'SUM')
]

HRNET_SPEC['w30'] = [
    #modules, block_type, blocks, channels, fuse_method
    (1,    'BOTTLENECK', [4],           [64],               'SUM'),
    (1,    'BASIC',      [4]*2,         [30, 60],           'SUM'),
    (4,    'BASIC',      [4]*3,         [30, 60, 120],      'SUM'),
    (3,    'BASIC',      [4]*4,         [30, 60, 120, 240], 'SUM')
]

HRNET_SPEC['w32'] = [
    #modules, block_type, blocks, channels, fuse_method
    (1,    'BOTTLENECK', [4],           [64],               'SUM'),
    (1,    'BASIC',      [4]*2,         [32, 64],           'SUM'),
    (4,    'BASIC',      [4]*3,         [32, 64, 128],      'SUM'),
    (3,    'BASIC',      [4]*4,         [32, 64, 128, 256], 'SUM')
]


HRNET_SPEC['w40'] = [
    #modules, block_type, blocks, channels, fuse_method
    (1,    'BOTTLENECK', [4],           [64],               'SUM'),
    (1,    'BASIC',      [4]*2,         [40, 80],           'SUM'),
    (4,    'BASIC',      [4]*3,         [40, 80, 160],      'SUM'),
    (3,    'BASIC',      [4]*4,         [40, 80, 160, 320], 'SUM')
]

HRNET_SPEC['w44'] = [
    #modules, block_type, blocks, channels, fuse_method
    (1,    'BOTTLENECK', [4],           [64],               'SUM'),
    (1,    'BASIC',      [4]*2,         [44, 88],           'SUM'),
    (4,    'BASIC',      [4]*3,         [44, 88, 176],      'SUM'),
    (3,    'BASIC',      [4]*4,         [44, 88, 176, 352], 'SUM')
]

HRNET_SPEC['w48'] = [
    #modules, block_type, blocks, channels, fuse_method
    (1,    'BOTTLENECK', [4],           [64],               'SUM'),
    (1,    'BASIC',      [4]*2,         [48, 96],           'SUM'),
    (4,    'BASIC',      [4]*3,         [48, 96, 192],      'SUM'),
    (3,    'BASIC',      [4]*4,         [48, 96, 192, 384], 'SUM')
]

HRNET_SPEC['w64'] = [
    #modules, block_type, blocks, channels, fuse_method
    (1,    'BOTTLENECK', [4],           [64],                'SUM'),
    (1,    'BASIC',      [4]*2,         [64, 128],           'SUM'),
    (4,    'BASIC',      [4]*3,         [64, 128, 256],      'SUM'),
    (3,    'BASIC',      [4]*4,         [64, 128, 256, 512], 'SUM')
]
# pylint: enable=C0326

def get_hrnet(model_name, stage_interp_type='nearest', purpose='cls', pretrained=False, ctx=cpu(),
              root='~/.mxnet/models', norm_layer=BatchNorm, norm_kwargs=None, num_classes=1000,
              **kwargs):
    r"""HRNet model from the
    `"Deep High-Resolution Representation Learning for Visual Recognition"
    <https://arxiv.org/pdf/1908.07919>`_ paper.

    Parameters
    ----------
    model_name : string
        The name of hrnet models: w18_small_v1/w18_small_v2/w30/w32/w40/w42/w48.
    stage_interp_type : string
        The interpolation type for upsample in each stage, nearest, bilinear and
        bilinear_like are supported.
    purpose: string
        The purpose of model, cls and seg are supported.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    if model_name not in HRNET_SPEC.keys():
        raise NotImplementedError

    spec = HRNET_SPEC[model_name]

    if purpose == 'cls':
        net = HighResolutionClsNet(spec, stage_interp_type, norm_layer,
                                   norm_kwargs, num_classes, **kwargs)
        from ..data import ImageNet1kAttr
        attrib = ImageNet1kAttr()
        net.synset = attrib.synset
        net.classes = attrib.classes
        net.classes_long = attrib.classes_long
    elif purpose == 'seg':
        net = HighResolutionSegNet(spec, stage_interp_type, norm_layer,
                                   norm_kwargs, num_classes, **kwargs)
    else:
        raise NotImplementedError

    if pretrained:
        from .model_store import get_model_file
        net.load_parameters(get_model_file('hrnet_%s_%s'%(model_name, purpose),
                                           tag=pretrained, root=root), ctx=ctx)
    return net

def hrnet_w18_c(**kwargs):
    r"""hrnet_w18 for Imagenet classification
    """
    return get_hrnet('w18', **kwargs)

def hrnet_w18_small_v1_c(**kwargs):
    r"""hhrnet_w18_small_v1 for Imagenet classification
    """
    return get_hrnet('w18_small_v1', **kwargs)

def hrnet_w18_small_v2_c(**kwargs):
    r"""hhrnet_w18_small_v2 for Imagenet classification
    """
    return get_hrnet('w18_small_v2', **kwargs)

def hrnet_w30_c(**kwargs):
    r"""hhrnet_w30 for Imagenet classification
    """
    return get_hrnet('w30', **kwargs)

def hrnet_w32_c(**kwargs):
    r"""hhrnet_w32 for Imagenet classification
    """
    return get_hrnet('w32', **kwargs)

def hrnet_w40_c(**kwargs):
    r"""hhrnet_w40 for Imagenet classification
    """
    return get_hrnet('w40', **kwargs)

def hrnet_w44_c(**kwargs):
    r"""hhrnet_w44 for Imagenet classification
    """
    return get_hrnet('w44', **kwargs)

def hrnet_w48_c(**kwargs):
    r"""hhrnet_w48 for Imagenet classification
    """
    return get_hrnet('w48', **kwargs)

def hrnet_w64_c(**kwargs):
    r"""hhrnet_w64 for Imagenet classification
    """
    return get_hrnet('w64', **kwargs)

def hrnet_w18_small_v1_s(**kwargs):
    r"""hrnet_w18_small_v1 for cityscapes segmentation
    """
    return get_hrnet('w18_small_v1', stage_interp_type='bilinear_like', purpose='seg',
                     norm_kwargs={'momentum': 0.99}, num_classes=19, **kwargs)

def hrnet_w18_small_v2_s(**kwargs):
    r"""hrnet_w18_small_v2 for cityscapes segmentation
    """
    return get_hrnet('w18_small_v2', stage_interp_type='bilinear_like', purpose='seg',
                     norm_kwargs={'momentum': 0.99}, num_classes=19, **kwargs)

def hrnet_w48_s(**kwargs):
    r"""hrnet_w48 for cityscapes segmentation
    """
    return get_hrnet('w48', stage_interp_type='bilinear_like', purpose='seg',
                     norm_kwargs={'momentum': 0.99}, num_classes=19, **kwargs)
