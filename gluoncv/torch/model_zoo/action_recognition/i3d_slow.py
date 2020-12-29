# pylint: disable=missing-function-docstring, line-too-long
"""
SlowFast Networks for Video Recognition
ICCV 2019, https://arxiv.org/abs/1812.03982
Code adapted from https://github.com/open-mmlab/mmaction and
https://github.com/decisionforce/TPN
"""
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from .non_local import build_nonlocal_block


__all__ = ['ResNet_SlowFast', 'i3d_slow_resnet50_f32s2_kinetics400',
           'i3d_slow_resnet50_f16s4_kinetics400', 'i3d_slow_resnet50_f8s8_kinetics400',
           'i3d_slow_resnet101_f32s2_kinetics400', 'i3d_slow_resnet101_f16s4_kinetics400',
           'i3d_slow_resnet101_f8s8_kinetics400', 'i3d_slow_resnet50_f32s2_custom', 'i3d_slow_resnet101_f16s4_kinetics700']


def conv3x3x3(in_planes, out_planes, spatial_stride=1, temporal_stride=1, dilation=1):
    "3x3x3 convolution with padding"
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=(temporal_stride, spatial_stride, spatial_stride),
                     padding=dilation,
                     dilation=dilation,
                     bias=False)


def conv1x3x3(in_planes, out_planes, spatial_stride=1, temporal_stride=1, dilation=1):
    "1x3x3 convolution with padding"
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=(1, 3, 3),
                     stride=(temporal_stride, spatial_stride, spatial_stride),
                     padding=(0, dilation, dilation),
                     dilation=dilation,
                     bias=False)


class Bottleneck(nn.Module):
    """Bottleneck block for ResNet.
    If style is "pytorch", the stride-two layer is the 3x3 conv layer,
    if it is "caffe", the stride-two layer is the first 1x1 conv layer.
    """
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


class ResNet_SlowFast(nn.Module):
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
                 num_classes,
                 depth,
                 pretrained=None,
                 pretrained_base=True,
                 feat_ext=False,
                 num_stages=4,
                 spatial_strides=(1, 2, 2, 2),
                 temporal_strides=(1, 1, 1, 1),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
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
        super(ResNet_SlowFast, self).__init__()

        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))

        self.num_classes = num_classes
        self.depth = depth
        self.pretrained = pretrained
        self.pretrained_base = pretrained_base
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

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(in_features=2048, out_features=num_classes)
        if not self.pretrained:
            nn.init.normal_(self.fc.weight, 0, self.init_std)
            nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = x.view(x.size(0), -1)

        if self.feat_ext:
            return x

        out = self.fc(x)
        return out


def i3d_slow_resnet50_f32s2_kinetics400(cfg):
    model = ResNet_SlowFast(num_classes=cfg.CONFIG.DATA.NUM_CLASSES,
                            depth=50,
                            pretrained=cfg.CONFIG.MODEL.PRETRAINED,
                            pretrained_base=cfg.CONFIG.MODEL.PRETRAINED_BASE,
                            feat_ext=cfg.CONFIG.INFERENCE.FEAT,
                            bn_eval=cfg.CONFIG.MODEL.BN_EVAL,
                            partial_bn=cfg.CONFIG.MODEL.PARTIAL_BN,
                            bn_frozen=cfg.CONFIG.MODEL.BN_FROZEN)

    if cfg.CONFIG.MODEL.PRETRAINED:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('i3d_slow_resnet50_f32s2_kinetics400',
                                                        tag=cfg.CONFIG.MODEL.PRETRAINED)))
    return model


def i3d_slow_resnet50_f16s4_kinetics400(cfg):
    model = ResNet_SlowFast(num_classes=cfg.CONFIG.DATA.NUM_CLASSES,
                            depth=50,
                            pretrained=cfg.CONFIG.MODEL.PRETRAINED,
                            pretrained_base=cfg.CONFIG.MODEL.PRETRAINED_BASE,
                            feat_ext=cfg.CONFIG.INFERENCE.FEAT,
                            bn_eval=cfg.CONFIG.MODEL.BN_EVAL,
                            partial_bn=cfg.CONFIG.MODEL.PARTIAL_BN,
                            bn_frozen=cfg.CONFIG.MODEL.BN_FROZEN)

    if cfg.CONFIG.MODEL.PRETRAINED:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('i3d_slow_resnet50_f16s4_kinetics400',
                                                        tag=cfg.CONFIG.MODEL.PRETRAINED)))
    return model


def i3d_slow_resnet50_f8s8_kinetics400(cfg):
    model = ResNet_SlowFast(num_classes=cfg.CONFIG.DATA.NUM_CLASSES,
                            depth=50,
                            pretrained=cfg.CONFIG.MODEL.PRETRAINED,
                            pretrained_base=cfg.CONFIG.MODEL.PRETRAINED_BASE,
                            feat_ext=cfg.CONFIG.INFERENCE.FEAT,
                            bn_eval=cfg.CONFIG.MODEL.BN_EVAL,
                            partial_bn=cfg.CONFIG.MODEL.PARTIAL_BN,
                            bn_frozen=cfg.CONFIG.MODEL.BN_FROZEN)

    if cfg.CONFIG.MODEL.PRETRAINED:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('i3d_slow_resnet50_f8s8_kinetics400',
                                                        tag=cfg.CONFIG.MODEL.PRETRAINED)))
    return model


def i3d_slow_resnet101_f32s2_kinetics400(cfg):
    model = ResNet_SlowFast(num_classes=cfg.CONFIG.DATA.NUM_CLASSES,
                            depth=101,
                            pretrained=cfg.CONFIG.MODEL.PRETRAINED,
                            pretrained_base=cfg.CONFIG.MODEL.PRETRAINED_BASE,
                            feat_ext=cfg.CONFIG.INFERENCE.FEAT,
                            bn_eval=cfg.CONFIG.MODEL.BN_EVAL,
                            partial_bn=cfg.CONFIG.MODEL.PARTIAL_BN,
                            bn_frozen=cfg.CONFIG.MODEL.BN_FROZEN)

    if cfg.CONFIG.MODEL.PRETRAINED:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('i3d_slow_resnet101_f32s2_kinetics400',
                                                        tag=cfg.CONFIG.MODEL.PRETRAINED)))
    return model


def i3d_slow_resnet101_f16s4_kinetics400(cfg):
    model = ResNet_SlowFast(num_classes=cfg.CONFIG.DATA.NUM_CLASSES,
                            depth=101,
                            pretrained=cfg.CONFIG.MODEL.PRETRAINED,
                            pretrained_base=cfg.CONFIG.MODEL.PRETRAINED_BASE,
                            feat_ext=cfg.CONFIG.INFERENCE.FEAT,
                            bn_eval=cfg.CONFIG.MODEL.BN_EVAL,
                            partial_bn=cfg.CONFIG.MODEL.PARTIAL_BN,
                            bn_frozen=cfg.CONFIG.MODEL.BN_FROZEN)

    if cfg.CONFIG.MODEL.PRETRAINED:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('i3d_slow_resnet101_f16s4_kinetics400',
                                                        tag=cfg.CONFIG.MODEL.PRETRAINED)))
    return model

def i3d_slow_resnet101_f16s4_kinetics700(cfg):
    model = ResNet_SlowFast(num_classes=cfg.CONFIG.DATA.NUM_CLASSES,
                            depth=101,
                            pretrained=cfg.CONFIG.MODEL.PRETRAINED,
                            pretrained_base=cfg.CONFIG.MODEL.PRETRAINED_BASE,
                            feat_ext=cfg.CONFIG.INFERENCE.FEAT,
                            bn_eval=cfg.CONFIG.MODEL.BN_EVAL,
                            partial_bn=cfg.CONFIG.MODEL.PARTIAL_BN,
                            bn_frozen=cfg.CONFIG.MODEL.BN_FROZEN)

    if cfg.CONFIG.MODEL.PRETRAINED:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('i3d_slow_resnet101_f16s4_kinetics700',
                                                        tag=cfg.CONFIG.MODEL.PRETRAINED)))
    return model

def i3d_slow_resnet101_f8s8_kinetics400(cfg):
    model = ResNet_SlowFast(num_classes=cfg.CONFIG.DATA.NUM_CLASSES,
                            depth=101,
                            pretrained=cfg.CONFIG.MODEL.PRETRAINED,
                            pretrained_base=cfg.CONFIG.MODEL.PRETRAINED_BASE,
                            feat_ext=cfg.CONFIG.INFERENCE.FEAT,
                            bn_eval=cfg.CONFIG.MODEL.BN_EVAL,
                            partial_bn=cfg.CONFIG.MODEL.PARTIAL_BN,
                            bn_frozen=cfg.CONFIG.MODEL.BN_FROZEN)

    if cfg.CONFIG.MODEL.PRETRAINED:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('i3d_slow_resnet101_f8s8_kinetics400',
                                                        tag=cfg.CONFIG.MODEL.PRETRAINED)))
    return model


def i3d_slow_resnet50_f32s2_custom(cfg):
    model = ResNet_SlowFast(num_classes=cfg.CONFIG.DATA.NUM_CLASSES,
                            depth=50,
                            pretrained=cfg.CONFIG.MODEL.PRETRAINED,
                            pretrained_base=cfg.CONFIG.MODEL.PRETRAINED_BASE,
                            feat_ext=cfg.CONFIG.INFERENCE.FEAT,
                            bn_eval=cfg.CONFIG.MODEL.BN_EVAL,
                            partial_bn=cfg.CONFIG.MODEL.PARTIAL_BN,
                            bn_frozen=cfg.CONFIG.MODEL.BN_FROZEN)

    if cfg.CONFIG.MODEL.PRETRAINED:
        from ..model_store import get_model_file
        state_dict = torch.load(get_model_file('i3d_slow_resnet50_f32s2_kinetics400', tag=cfg.CONFIG.MODEL.PRETRAINED))
        for k in list(state_dict.keys()):
            # retain only backbone up to before the classification layer
            if k.startswith('fc'):
                del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {'fc.weight', 'fc.bias'}
        print("=> Initialized from a I3D_slow model pretrained on Kinetcis400 dataset")
    return model
