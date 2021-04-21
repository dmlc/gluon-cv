# pylint: disable=missing-docstring, unused-argument, line-too-long, too-many-lines
"""
Temporal Pyramid Network for Action Recognition
CVPR 2020, https://arxiv.org/pdf/2004.03548.pdf
Code adapted from https://github.com/decisionforce/TPN
"""
import numpy as np

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import torch.nn.functional as F


__all__ = ['TPN', 'TPNet', 'tpn_resnet50_f8s8_kinetics400', 'tpn_resnet50_f16s4_kinetics400',
           'tpn_resnet50_f32s2_kinetics400', 'tpn_resnet101_f8s8_kinetics400',
           'tpn_resnet101_f16s4_kinetics400', 'tpn_resnet101_f32s2_kinetics400',
           'tpn_resnet50_f32s2_custom']


def rgetattr(obj, attr, *args):
    import functools
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


def rhasattr(obj, attr, *args):
    import functools
    def _hasattr(obj, attr):
        if hasattr(obj, attr):
            return getattr(obj, attr)
        else:
            return None

    return functools.reduce(_hasattr, [obj] + attr.split('.')) is not None


def conv3x3x3(in_planes, out_planes, spatial_stride=1, temporal_stride=1, dilation=1):
    "3x3x3 convolution with padding"
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=(temporal_stride, spatial_stride, spatial_stride),
        padding=dilation,
        dilation=dilation,
        bias=False)


def conv1x3x3(in_planes, out_planes, spatial_stride=1, temporal_stride=1, dilation=1):
    "1x3x3 convolution with padding"
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(1, 3, 3),
        stride=(temporal_stride, spatial_stride, spatial_stride),
        padding=(0, dilation, dilation),
        dilation=dilation,
        bias=False)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ConvModule(nn.Module):
    def __init__(
            self,
            inplanes,
            planes,
            kernel_size,
            stride,
            padding,
            bias=False,
            groups=1,
    ):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv3d(inplanes, planes, kernel_size, stride, padding, bias=bias, groups=groups)
        self.bn = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


class AuxHead(nn.Module):
    def __init__(self, inplanes, planes, loss_weight=0.5):
        super(AuxHead, self).__init__()
        self.convs = ConvModule(inplanes, inplanes * 2, kernel_size=(1, 3, 3),
                                stride=(1, 2, 2), padding=(0, 1, 1), bias=False)
        self.loss_weight = loss_weight
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(inplanes * 2, planes)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Conv3d):
                pass
            if isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.fill_(0)

    def forward(self, x, target=None):
        if target is None:
            return None
        loss = dict()
        x = self.convs(x)
        x = F.adaptive_avg_pool3d(x, 1).squeeze(-1).squeeze(-1).squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)

        loss['loss_aux'] = self.loss_weight * F.cross_entropy(x, target)
        return loss


class TemporalModulation(nn.Module):
    def __init__(self, inplanes, planes, downsample_scale=8):
        super(TemporalModulation, self).__init__()

        self.conv = nn.Conv3d(inplanes, planes, (3, 1, 1), (1, 1, 1), (1, 0, 0), bias=False, groups=32)
        self.pool = nn.MaxPool3d((downsample_scale, 1, 1), (downsample_scale, 1, 1), (0, 0, 0), ceil_mode=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class Upsampling(nn.Module):
    def __init__(self, scale=(2, 1, 1)):
        super(Upsampling, self).__init__()
        self.scale = scale

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode='nearest')
        return x


class Downampling(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size=(3, 1, 1),
                 stride=(1, 1, 1),
                 padding=(1, 0, 0),
                 bias=False,
                 groups=1,
                 norm=False,
                 activation=False,
                 downsample_position='after',
                 downsample_scale=(1, 2, 2),
                 ):
        super(Downampling, self).__init__()

        self.conv = nn.Conv3d(inplanes, planes, kernel_size, stride, padding, bias=bias, groups=groups)
        self.norm = nn.BatchNorm3d(planes) if norm else None
        self.relu = nn.ReLU(inplace=True) if activation else None
        assert (downsample_position in ['before', 'after'])
        self.downsample_position = downsample_position
        self.pool = nn.MaxPool3d(downsample_scale, downsample_scale, (0, 0, 0), ceil_mode=True)

    def forward(self, x):
        # pylint: disable=not-callable
        if self.downsample_position == 'before':
            x = self.pool(x)
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.downsample_position == 'after':
            x = self.pool(x)

        return x


class LevelFusion(nn.Module):
    def __init__(self,
                 in_channels=(1024, 1024),
                 mid_channels=(1024, 1024),
                 out_channels=2048,
                 ds_scales=((1, 1, 1), (1, 1, 1)),
                 ):
        super(LevelFusion, self).__init__()
        self.ops = nn.ModuleList()
        num_ins = len(in_channels)
        for i in range(num_ins):
            op = Downampling(in_channels[i], mid_channels[i], kernel_size=(1, 1, 1), stride=(1, 1, 1),
                             padding=(0, 0, 0), bias=False, groups=32, norm=True, activation=True,
                             downsample_position='before', downsample_scale=ds_scales[i])
            self.ops.append(op)

        in_dims = np.sum(mid_channels)
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(in_dims, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, inputs):
        out = [self.ops[i](feature) for i, feature in enumerate(inputs)]
        out = torch.cat(out, 1)
        out = self.fusion_conv(out)
        return out


class SpatialModulation(nn.Module):
    def __init__(self, inplanes=(1024, 2048), planes=2048):
        super(SpatialModulation, self).__init__()

        self.spatial_modulation = nn.ModuleList()
        for _, dim in enumerate(inplanes):
            op = nn.ModuleList()
            ds_factor = planes // dim
            ds_num = int(np.log2(ds_factor))
            if ds_num < 1:
                op = Identity()
            else:
                for dsi in range(ds_num):
                    in_factor = 2 ** dsi
                    out_factor = 2 ** (dsi + 1)
                    op.append(ConvModule(dim * in_factor, dim * out_factor, kernel_size=(1, 3, 3), stride=(1, 2, 2),
                                         padding=(0, 1, 1), bias=False))
            self.spatial_modulation.append(op)

    def forward(self, inputs):
        out = []
        for i, _ in enumerate(inputs):
            if isinstance(self.spatial_modulation[i], nn.ModuleList):
                out_ = inputs[i]
                for _, op in enumerate(self.spatial_modulation[i]):
                    out_ = op(out_)
                out.append(out_)
            else:
                out.append(self.spatial_modulation[i](inputs[i]))
        return out


class TPN(nn.Module):

    def __init__(self,
                 in_channels=(256, 512, 1024, 2048),
                 out_channels=256,
                 spatial_modulation_config=None,
                 temporal_modulation_config=None,
                 upsampling_config=None,
                 downsampling_config=None,
                 level_fusion_config=None,
                 aux_head_config=None,
                 ):
        super(TPN, self).__init__()
        assert isinstance(out_channels, int)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)

        self.temporal_modulation_ops = nn.ModuleList()
        self.upsampling_ops = nn.ModuleList()
        self.downsampling_ops = nn.ModuleList()
        self.level_fusion_op = LevelFusion(**level_fusion_config)
        self.spatial_modulation = SpatialModulation(**spatial_modulation_config)
        for i in range(0, self.num_ins, 1):
            inplanes = in_channels[-1]
            planes = out_channels

            if temporal_modulation_config is not None:
                # overwrite the temporal_modulation_config
                temporal_modulation_config['param']['downsample_scale'] = temporal_modulation_config['scales'][i]
                temporal_modulation_config['param']['inplanes'] = inplanes
                temporal_modulation_config['param']['planes'] = planes
                temporal_modulation = TemporalModulation(**temporal_modulation_config['param'])
                self.temporal_modulation_ops.append(temporal_modulation)

            if i < self.num_ins - 1:
                if upsampling_config is not None:
                    # overwrite the upsampling_config
                    upsampling = Upsampling(**upsampling_config)
                    self.upsampling_ops.append(upsampling)

                if downsampling_config is not None:
                    # overwrite the downsampling_config
                    downsampling_config['param']['inplanes'] = planes
                    downsampling_config['param']['planes'] = planes
                    downsampling_config['param']['downsample_scale'] = downsampling_config['scales']
                    downsampling = Downampling(**downsampling_config['param'])
                    self.downsampling_ops.append(downsampling)

        out_dims = level_fusion_config['out_channels']

        # Two pyramids
        self.level_fusion_op2 = LevelFusion(**level_fusion_config)

        self.pyramid_fusion_op = nn.Sequential(
            nn.Conv3d(out_dims * 2, 2048, 1, 1, 0, bias=False),
            nn.BatchNorm3d(2048),
            nn.ReLU(inplace=True)
        )

        # overwrite aux_head_config
        if aux_head_config is not None:
            aux_head_config['inplanes'] = self.in_channels[-2]
            self.aux_head = AuxHead(**aux_head_config)
        else:
            self.aux_head = None

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                pass
            if isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.fill_(0)

        if self.aux_head is not None:
            self.aux_head.init_weights()

    def forward(self, inputs, target=None):
        loss = None

        # Auxiliary loss
        if self.aux_head is not None:
            loss = self.aux_head(inputs[-2], target)

        # Spatial Modulation
        outs = self.spatial_modulation(inputs)

        # Temporal Modulation
        outs = [temporal_modulation(outs[i]) for i, temporal_modulation in enumerate(self.temporal_modulation_ops)]

        temporal_modulation_outs = outs

        # Build top-down flow - upsampling operation
        if self.upsampling_ops is not None:
            for i in range(self.num_ins - 1, 0, -1):
                outs[i - 1] = outs[i - 1] + self.upsampling_ops[i - 1](outs[i])

        # Get top-down outs
        topdownouts = self.level_fusion_op2(outs)
        outs = temporal_modulation_outs

        # Build bottom-up flow - downsampling operation
        if self.downsampling_ops is not None:
            for i in range(0, self.num_ins - 1, 1):
                outs[i + 1] = outs[i + 1] + self.downsampling_ops[i](outs[i])

        # Get bottom-up outs
        outs = self.level_fusion_op(outs)

        # fuse two pyramid outs
        outs = self.pyramid_fusion_op(torch.cat([topdownouts, outs], 1))

        return outs, loss


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 spatial_stride=1,
                 temporal_stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 if_inflate=True,
                 inflate_style='3x1x1',
                 if_nonlocal=True,
                 nonlocal_cfg=None,
                 with_cp=False):
        """Bottleneck block for ResNet.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']
        assert inflate_style in ['3x1x1', '3x3x3']
        self.inplanes = inplanes
        self.planes = planes

        if style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = spatial_stride
            self.conv1_stride_t = 1
            self.conv2_stride_t = temporal_stride
        else:
            self.conv1_stride = spatial_stride
            self.conv2_stride = 1
            self.conv1_stride_t = temporal_stride
            self.conv2_stride_t = 1
        if if_inflate:
            if inflate_style == '3x1x1':
                self.conv1 = nn.Conv3d(
                    inplanes,
                    planes,
                    kernel_size=(3, 1, 1),
                    stride=(self.conv1_stride_t, self.conv1_stride, self.conv1_stride),
                    padding=(1, 0, 0),
                    bias=False)
                self.conv2 = nn.Conv3d(
                    planes,
                    planes,
                    kernel_size=(1, 3, 3),
                    stride=(self.conv2_stride_t, self.conv2_stride, self.conv2_stride),
                    padding=(0, dilation, dilation),
                    dilation=(1, dilation, dilation),
                    bias=False)
            else:
                self.conv1 = nn.Conv3d(
                    inplanes,
                    planes,
                    kernel_size=1,
                    stride=(self.conv1_stride_t, self.conv1_stride, self.conv1_stride),
                    bias=False)
                self.conv2 = nn.Conv3d(
                    planes,
                    planes,
                    kernel_size=3,
                    stride=(self.conv2_stride_t, self.conv2_stride, self.conv2_stride),
                    padding=(1, dilation, dilation),
                    dilation=(1, dilation, dilation),
                    bias=False)
        else:
            self.conv1 = nn.Conv3d(
                inplanes,
                planes,
                kernel_size=1,
                stride=(1, self.conv1_stride, self.conv1_stride),
                bias=False)
            self.conv2 = nn.Conv3d(
                planes,
                planes,
                kernel_size=(1, 3, 3),
                stride=(1, self.conv2_stride, self.conv2_stride),
                padding=(0, dilation, dilation),
                dilation=(1, dilation, dilation),
                bias=False)

        self.bn1 = nn.BatchNorm3d(planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(
            planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.spatial_tride = spatial_stride
        self.temporal_tride = temporal_stride
        self.dilation = dilation
        self.with_cp = with_cp

        if if_nonlocal and nonlocal_cfg is not None:
            nonlocal_cfg_ = nonlocal_cfg.copy()
            nonlocal_cfg_['in_channels'] = planes * self.expansion
            self.nonlocal_block = build_nonlocal_block(nonlocal_cfg_)
        else:
            self.nonlocal_block = None

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        if self.nonlocal_block is not None:
            out = self.nonlocal_block(out)

        return out


def make_res_layer(block,
                   inplanes,
                   planes,
                   blocks,
                   spatial_stride=1,
                   temporal_stride=1,
                   dilation=1,
                   style='pytorch',
                   inflate_freq=1,
                   inflate_style='3x1x1',
                   nonlocal_freq=1,
                   nonlocal_cfg=None,
                   with_cp=False):
    inflate_freq = inflate_freq if not isinstance(inflate_freq, int) else (inflate_freq,) * blocks
    nonlocal_freq = nonlocal_freq if not isinstance(nonlocal_freq, int) else (nonlocal_freq,) * blocks
    assert len(inflate_freq) == blocks
    assert len(nonlocal_freq) == blocks
    downsample = None
    if spatial_stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv3d(
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=(temporal_stride, spatial_stride, spatial_stride),
                bias=False),
            nn.BatchNorm3d(planes * block.expansion),
        )

    layers = []
    layers.append(
        block(
            inplanes,
            planes,
            spatial_stride,
            temporal_stride,
            dilation,
            downsample,
            style=style,
            if_inflate=(inflate_freq[0] == 1),
            inflate_style=inflate_style,
            if_nonlocal=(nonlocal_freq[0] == 1),
            nonlocal_cfg=nonlocal_cfg,
            with_cp=with_cp))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(
            block(inplanes,
                  planes,
                  1, 1,
                  dilation,
                  style=style,
                  if_inflate=(inflate_freq[i] == 1),
                  inflate_style=inflate_style,
                  if_nonlocal=(nonlocal_freq[i] == 1),
                  nonlocal_cfg=nonlocal_cfg,
                  with_cp=with_cp))

    return nn.Sequential(*layers)


class TPNet(nn.Module):
    """ResNe(x)t_SlowFast backbone.
    Args:
        depth (int): Depth of resnet, from {50, 101, 152}.
        num_stages (int): Resnet stages, normally 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        bn_eval (bool): Whether to set BN layers to eval mode, namely, freeze
            running stats (mean and var).
        bn_frozen (bool): Whether to freeze weight and bias of BN layers.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
    """

    arch_settings = {
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 TPN_neck=None,
                 num_classes=400,
                 pretrained=None,
                 pretrained_base=True,
                 feat_ext=False,
                 num_stages=4,
                 spatial_strides=(1, 2, 2, 2),
                 temporal_strides=(1, 1, 1, 1),
                 dilations=(1, 1, 1, 1),
                 out_indices=(2, 3),
                 conv1_kernel_t=1,
                 conv1_stride_t=1,
                 pool1_kernel_t=1,
                 pool1_stride_t=1,
                 style='pytorch',
                 frozen_stages=-1,
                 inflate_freq=(0, 0, 1, 1),
                 inflate_stride=(1, 1, 1, 1),
                 inflate_style='3x1x1',
                 nonlocal_stages=(-1,),
                 nonlocal_freq=(0, 0, 0, 0),
                 nonlocal_cfg=None,
                 bn_eval=False,
                 bn_frozen=False,
                 partial_bn=False,
                 with_cp=False,
                 dropout_ratio=0.5,
                 init_std=0.01):
        super(TPNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        self.depth = depth
        self.pretrained = pretrained
        self.pretrained_base = pretrained_base
        self.num_classes = num_classes
        self.num_stages = num_stages
        assert 1 <= num_stages <= 4
        self.spatial_strides = spatial_strides
        self.temporal_strides = temporal_strides
        self.dilations = dilations
        assert len(spatial_strides) == len(temporal_strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.frozen_stages = frozen_stages
        self.inflate_freqs = inflate_freq if not isinstance(inflate_freq, int) else (inflate_freq,) * num_stages
        self.inflate_style = inflate_style
        self.nonlocal_stages = nonlocal_stages
        self.nonlocal_freqs = nonlocal_freq if not isinstance(nonlocal_freq, int) else (nonlocal_freq,) * num_stages
        self.nonlocal_cfg = nonlocal_cfg
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen
        self.partial_bn = partial_bn
        self.with_cp = with_cp
        self.feat_ext = feat_ext

        self.dropout_ratio = dropout_ratio
        self.init_std = init_std

        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = 64

        self.conv1 = nn.Conv3d(
            3, 64, kernel_size=(conv1_kernel_t, 7, 7), stride=(conv1_stride_t, 2, 2),
            padding=((conv1_kernel_t - 1) // 2, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(pool1_kernel_t, 3, 3), stride=(pool1_stride_t, 2, 2),
                                    padding=(pool1_kernel_t // 2, 1, 1))

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            spatial_stride = spatial_strides[i]
            temporal_stride = temporal_strides[i]
            dilation = dilations[i]
            planes = 64 * 2 ** i
            res_layer = make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                spatial_stride=spatial_stride,
                temporal_stride=temporal_stride,
                dilation=dilation,
                style=self.style,
                inflate_freq=self.inflate_freqs[i],
                inflate_style=self.inflate_style,
                nonlocal_freq=self.nonlocal_freqs[i],
                nonlocal_cfg=self.nonlocal_cfg if i in self.nonlocal_stages else None,
                with_cp=with_cp)
            self.inplanes = planes * self.block.expansion
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self.feat_dim = self.block.expansion * 64 * 2 ** (len(self.stage_blocks) - 1)

        self.TPN_neck = TPN_neck

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(2048, self.num_classes)
        nn.init.normal_(self.fc.weight, 0, self.init_std)
        nn.init.constant_(self.fc.bias, 0)
        self.init_weights()

    def init_weights(self):
        if not self.pretrained_base:
            raise RuntimeError("TPN models need to be inflated. Please set PRETRAINED_BASE to True in config.")

        if self.pretrained_base and not self.pretrained:
            import torchvision
            if self.depth == 50:
                R2D = torchvision.models.resnet50(pretrained=True, progress=True)
            elif self.depth == 101:
                R2D = torchvision.models.resnet101(pretrained=True, progress=True)
            else:
                raise RuntimeError("We only support ResNet50 and ResNet101 for TPN models at this moment.")

            for name, module in self.named_modules():
                if isinstance(module, nn.Conv3d) and rhasattr(R2D, name):
                    new_weight = rgetattr(R2D, name).weight.data.unsqueeze(2).expand_as(module.weight) / \
                                 module.weight.data.shape[2]
                    module.weight.data.copy_(new_weight)
                    if hasattr(module, 'bias') and module.bias is not None:
                        new_bias = rgetattr(R2D, name).bias.data
                        module.bias.data.copy_(new_bias)
                elif isinstance(module, nn.BatchNorm3d) and rhasattr(R2D, name):
                    for attr in ['weight', 'bias', 'running_mean', 'running_var']:
                        setattr(module, attr, getattr(rgetattr(R2D, name), attr))
            print('TPN weights inflated from pretrained C2D.')

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        feat = [x3, x4]
        tpn_feat = self.TPN_neck(feat)

        x = self.avg_pool(tpn_feat[0])
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)

        if self.feat_ext:
            return x

        out = self.fc(x)
        return out


def tpn_resnet50_f8s8_kinetics400(cfg):
    neck = TPN(in_channels=[1024, 2048],
               out_channels=1024,
               spatial_modulation_config=dict(
                   inplanes=[1024, 2048],
                   planes=2048, ),
               temporal_modulation_config=dict(
                   scales=(8, 8),
                   param=dict(
                       inplanes=-1,
                       planes=-1,
                       downsample_scale=-1,
                   )),
               upsampling_config=dict(
                   scale=(1, 1, 1),
               ),
               downsampling_config=dict(
                   scales=(1, 1, 1),
                   param=dict(
                       inplanes=-1,
                       planes=-1,
                       downsample_scale=-1,
                   )),
               level_fusion_config=dict(
                   in_channels=[1024, 1024],
                   mid_channels=[1024, 1024],
                   out_channels=2048,
                   ds_scales=[(1, 1, 1), (1, 1, 1)],
               ),
               aux_head_config=dict(
                   inplanes=-1,
                   planes=cfg.CONFIG.DATA.NUM_CLASSES,
                   loss_weight=0.5
               ))

    model = TPNet(depth=50,
                  TPN_neck=neck,
                  num_classes=cfg.CONFIG.DATA.NUM_CLASSES,
                  pretrained=cfg.CONFIG.MODEL.PRETRAINED,
                  pretrained_base=cfg.CONFIG.MODEL.PRETRAINED_BASE,
                  feat_ext=cfg.CONFIG.INFERENCE.FEAT,
                  bn_eval=cfg.CONFIG.MODEL.BN_EVAL,
                  partial_bn=cfg.CONFIG.MODEL.PARTIAL_BN,
                  bn_frozen=cfg.CONFIG.MODEL.BN_FROZEN)

    if cfg.CONFIG.MODEL.PRETRAINED:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('tpn_resnet50_f8s8_kinetics400',
                                                        tag=cfg.CONFIG.MODEL.PRETRAINED)))
    return model


def tpn_resnet50_f16s4_kinetics400(cfg):
    neck = TPN(in_channels=[1024, 2048],
               out_channels=1024,
               spatial_modulation_config=dict(
                   inplanes=[1024, 2048],
                   planes=2048, ),
               temporal_modulation_config=dict(
                   scales=(16, 16),
                   param=dict(
                       inplanes=-1,
                       planes=-1,
                       downsample_scale=-1,
                   )),
               upsampling_config=dict(
                   scale=(1, 1, 1),
               ),
               downsampling_config=dict(
                   scales=(1, 1, 1),
                   param=dict(
                       inplanes=-1,
                       planes=-1,
                       downsample_scale=-1,
                   )),
               level_fusion_config=dict(
                   in_channels=[1024, 1024],
                   mid_channels=[1024, 1024],
                   out_channels=2048,
                   ds_scales=[(1, 1, 1), (1, 1, 1)],
               ),
               aux_head_config=dict(
                   inplanes=-1,
                   planes=cfg.CONFIG.DATA.NUM_CLASSES,
                   loss_weight=0.5
               ))

    model = TPNet(depth=50,
                  TPN_neck=neck,
                  num_classes=cfg.CONFIG.DATA.NUM_CLASSES,
                  pretrained=cfg.CONFIG.MODEL.PRETRAINED,
                  pretrained_base=cfg.CONFIG.MODEL.PRETRAINED_BASE,
                  feat_ext=cfg.CONFIG.INFERENCE.FEAT,
                  bn_eval=cfg.CONFIG.MODEL.BN_EVAL,
                  partial_bn=cfg.CONFIG.MODEL.PARTIAL_BN,
                  bn_frozen=cfg.CONFIG.MODEL.BN_FROZEN)

    if cfg.CONFIG.MODEL.PRETRAINED:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('tpn_resnet50_f16s4_kinetics400',
                                                        tag=cfg.CONFIG.MODEL.PRETRAINED)))
    return model


def tpn_resnet50_f32s2_kinetics400(cfg):
    neck = TPN(in_channels=[1024, 2048],
               out_channels=1024,
               spatial_modulation_config=dict(
                   inplanes=[1024, 2048],
                   planes=2048, ),
               temporal_modulation_config=dict(
                   scales=(32, 32),
                   param=dict(
                       inplanes=-1,
                       planes=-1,
                       downsample_scale=-1,
                   )),
               upsampling_config=dict(
                   scale=(1, 1, 1),
               ),
               downsampling_config=dict(
                   scales=(1, 1, 1),
                   param=dict(
                       inplanes=-1,
                       planes=-1,
                       downsample_scale=-1,
                   )),
               level_fusion_config=dict(
                   in_channels=[1024, 1024],
                   mid_channels=[1024, 1024],
                   out_channels=2048,
                   ds_scales=[(1, 1, 1), (1, 1, 1)],
               ),
               aux_head_config=dict(
                   inplanes=-1,
                   planes=cfg.CONFIG.DATA.NUM_CLASSES,
                   loss_weight=0.5
               ))

    model = TPNet(depth=50,
                  TPN_neck=neck,
                  num_classes=cfg.CONFIG.DATA.NUM_CLASSES,
                  pretrained=cfg.CONFIG.MODEL.PRETRAINED,
                  pretrained_base=cfg.CONFIG.MODEL.PRETRAINED_BASE,
                  feat_ext=cfg.CONFIG.INFERENCE.FEAT,
                  bn_eval=cfg.CONFIG.MODEL.BN_EVAL,
                  partial_bn=cfg.CONFIG.MODEL.PARTIAL_BN,
                  bn_frozen=cfg.CONFIG.MODEL.BN_FROZEN)

    if cfg.CONFIG.MODEL.PRETRAINED:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('tpn_resnet50_f32s2_kinetics400',
                                                        tag=cfg.CONFIG.MODEL.PRETRAINED)))
    return model


def tpn_resnet101_f8s8_kinetics400(cfg):
    neck = TPN(in_channels=[1024, 2048],
               out_channels=1024,
               spatial_modulation_config=dict(
                   inplanes=[1024, 2048],
                   planes=2048, ),
               temporal_modulation_config=dict(
                   scales=(4, 8),
                   param=dict(
                       inplanes=-1,
                       planes=-1,
                       downsample_scale=-1,
                   )),
               upsampling_config=dict(
                   scale=(1, 1, 1),
               ),
               downsampling_config=dict(
                   scales=(2, 1, 1),
                   param=dict(
                       inplanes=-1,
                       planes=-1,
                       downsample_scale=-1,
                   )),
               level_fusion_config=dict(
                   in_channels=[1024, 1024],
                   mid_channels=[1024, 1024],
                   out_channels=2048,
                   ds_scales=[(2, 1, 1), (1, 1, 1)],
               ),
               aux_head_config=dict(
                   inplanes=-1,
                   planes=cfg.CONFIG.DATA.NUM_CLASSES,
                   loss_weight=0.5
               ))

    model = TPNet(depth=101,
                  TPN_neck=neck,
                  num_classes=cfg.CONFIG.DATA.NUM_CLASSES,
                  pretrained=cfg.CONFIG.MODEL.PRETRAINED,
                  pretrained_base=cfg.CONFIG.MODEL.PRETRAINED_BASE,
                  feat_ext=cfg.CONFIG.INFERENCE.FEAT,
                  bn_eval=cfg.CONFIG.MODEL.BN_EVAL,
                  partial_bn=cfg.CONFIG.MODEL.PARTIAL_BN,
                  bn_frozen=cfg.CONFIG.MODEL.BN_FROZEN)

    if cfg.CONFIG.MODEL.PRETRAINED:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('tpn_resnet101_f8s8_kinetics400',
                                                        tag=cfg.CONFIG.MODEL.PRETRAINED)))
    return model


def tpn_resnet101_f16s4_kinetics400(cfg):
    neck = TPN(in_channels=[1024, 2048],
               out_channels=1024,
               spatial_modulation_config=dict(
                   inplanes=[1024, 2048],
                   planes=2048, ),
               temporal_modulation_config=dict(
                   scales=(8, 16),
                   param=dict(
                       inplanes=-1,
                       planes=-1,
                       downsample_scale=-1,
                   )),
               upsampling_config=dict(
                   scale=(1, 1, 1),
               ),
               downsampling_config=dict(
                   scales=(2, 1, 1),
                   param=dict(
                       inplanes=-1,
                       planes=-1,
                       downsample_scale=-1,
                   )),
               level_fusion_config=dict(
                   in_channels=[1024, 1024],
                   mid_channels=[1024, 1024],
                   out_channels=2048,
                   ds_scales=[(2, 1, 1), (1, 1, 1)],
               ),
               aux_head_config=dict(
                   inplanes=-1,
                   planes=cfg.CONFIG.DATA.NUM_CLASSES,
                   loss_weight=0.5
               ))

    model = TPNet(depth=101,
                  TPN_neck=neck,
                  num_classes=cfg.CONFIG.DATA.NUM_CLASSES,
                  pretrained=cfg.CONFIG.MODEL.PRETRAINED,
                  pretrained_base=cfg.CONFIG.MODEL.PRETRAINED_BASE,
                  feat_ext=cfg.CONFIG.INFERENCE.FEAT,
                  bn_eval=cfg.CONFIG.MODEL.BN_EVAL,
                  partial_bn=cfg.CONFIG.MODEL.PARTIAL_BN,
                  bn_frozen=cfg.CONFIG.MODEL.BN_FROZEN)

    if cfg.CONFIG.MODEL.PRETRAINED:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('tpn_resnet101_f16s4_kinetics400',
                                                        tag=cfg.CONFIG.MODEL.PRETRAINED)))
    return model


def tpn_resnet101_f32s2_kinetics400(cfg):
    neck = TPN(in_channels=[1024, 2048],
               out_channels=1024,
               spatial_modulation_config=dict(
                   inplanes=[1024, 2048],
                   planes=2048, ),
               temporal_modulation_config=dict(
                   scales=(16, 32),
                   param=dict(
                       inplanes=-1,
                       planes=-1,
                       downsample_scale=-1,
                   )),
               upsampling_config=dict(
                   scale=(1, 1, 1),
               ),
               downsampling_config=dict(
                   scales=(2, 1, 1),
                   param=dict(
                       inplanes=-1,
                       planes=-1,
                       downsample_scale=-1,
                   )),
               level_fusion_config=dict(
                   in_channels=[1024, 1024],
                   mid_channels=[1024, 1024],
                   out_channels=2048,
                   ds_scales=[(2, 1, 1), (1, 1, 1)],
               ),
               aux_head_config=dict(
                   inplanes=-1,
                   planes=cfg.CONFIG.DATA.NUM_CLASSES,
                   loss_weight=0.5
               ))

    model = TPNet(depth=101,
                  TPN_neck=neck,
                  num_classes=cfg.CONFIG.DATA.NUM_CLASSES,
                  pretrained=cfg.CONFIG.MODEL.PRETRAINED,
                  pretrained_base=cfg.CONFIG.MODEL.PRETRAINED_BASE,
                  feat_ext=cfg.CONFIG.INFERENCE.FEAT,
                  bn_eval=cfg.CONFIG.MODEL.BN_EVAL,
                  partial_bn=cfg.CONFIG.MODEL.PARTIAL_BN,
                  bn_frozen=cfg.CONFIG.MODEL.BN_FROZEN)

    if cfg.CONFIG.MODEL.PRETRAINED:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('tpn_resnet101_f32s2_kinetics400',
                                                        tag=cfg.CONFIG.MODEL.PRETRAINED)))
    return model


def tpn_resnet50_f32s2_custom(cfg):
    neck = TPN(in_channels=[1024, 2048],
               out_channels=1024,
               spatial_modulation_config=dict(
                   inplanes=[1024, 2048],
                   planes=2048, ),
               temporal_modulation_config=dict(
                   scales=(32, 32),
                   param=dict(
                       inplanes=-1,
                       planes=-1,
                       downsample_scale=-1,
                   )),
               upsampling_config=dict(
                   scale=(1, 1, 1),
               ),
               downsampling_config=dict(
                   scales=(1, 1, 1),
                   param=dict(
                       inplanes=-1,
                       planes=-1,
                       downsample_scale=-1,
                   )),
               level_fusion_config=dict(
                   in_channels=[1024, 1024],
                   mid_channels=[1024, 1024],
                   out_channels=2048,
                   ds_scales=[(1, 1, 1), (1, 1, 1)],
               ),
               aux_head_config=dict(
                   inplanes=-1,
                   planes=cfg.CONFIG.DATA.NUM_CLASSES,
                   loss_weight=0.5
               ))

    model = TPNet(depth=50,
                  TPN_neck=neck,
                  num_classes=cfg.CONFIG.DATA.NUM_CLASSES,
                  pretrained=cfg.CONFIG.MODEL.PRETRAINED,
                  pretrained_base=cfg.CONFIG.MODEL.PRETRAINED_BASE,
                  feat_ext=cfg.CONFIG.INFERENCE.FEAT,
                  bn_eval=cfg.CONFIG.MODEL.BN_EVAL,
                  partial_bn=cfg.CONFIG.MODEL.PARTIAL_BN,
                  bn_frozen=cfg.CONFIG.MODEL.BN_FROZEN)

    if cfg.CONFIG.MODEL.PRETRAINED:
        from ..model_store import get_model_file
        state_dict = torch.load(get_model_file('tpn_resnet50_f32s2_kinetics400', tag=cfg.CONFIG.MODEL.PRETRAINED))
        for k in list(state_dict.keys()):
            # retain only backbone up to before the classification layer
            if k.startswith('fc'):
                del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {'fc.weight', 'fc.bias'}
        print("=> initialized from a SlowFast4x16 model pretrained on Kinetcis400 dataset")
    return model
