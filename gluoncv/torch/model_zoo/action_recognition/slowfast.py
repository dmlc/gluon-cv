# pylint: disable=missing-function-docstring, line-too-long, unused-argument
"""
SlowFast Networks for Video Recognition
ICCV 2019, https://arxiv.org/abs/1812.03982
Code adapted from https://github.com/r1ch88/SlowFastNetworks
"""
import torch
import torch.nn as nn
from torch.nn import BatchNorm3d


__all__ = ['SlowFast', 'slowfast_4x16_resnet50_kinetics400', 'slowfast_8x8_resnet50_kinetics400',
           'slowfast_4x16_resnet101_kinetics400', 'slowfast_8x8_resnet101_kinetics400',
           'slowfast_16x8_resnet101_kinetics400', 'slowfast_16x8_resnet101_50_50_kinetics400',
           'slowfast_16x8_resnet50_sthsthv2']


class Bottleneck(nn.Module):
    """Bottleneck block for ResNet-SlowFast.
    """
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 strides=1,
                 downsample=None,
                 head_conv=1,
                 norm_layer=BatchNorm3d,
                 norm_kwargs=None,
                 layer_name=''):
        super(Bottleneck, self).__init__()

        if head_conv == 1:
            self.conv1 = nn.Conv3d(in_channels=inplanes, out_channels=planes, kernel_size=1, bias=False)
            self.bn1 = norm_layer(num_features=planes, **({} if norm_kwargs is None else norm_kwargs))
        elif head_conv == 3:
            self.conv1 = nn.Conv3d(in_channels=inplanes, out_channels=planes, kernel_size=(3, 1, 1), padding=(1, 0, 0),
                                   bias=False)
            self.bn1 = norm_layer(num_features=planes, **({} if norm_kwargs is None else norm_kwargs))
        else:
            raise ValueError("Unsupported head_conv!")
        self.conv2 = nn.Conv3d(in_channels=planes, out_channels=planes, kernel_size=(1, 3, 3),
                               stride=(1, strides, strides), padding=(0, 1, 1), bias=False)
        self.bn2 = norm_layer(num_features=planes, **({} if norm_kwargs is None else norm_kwargs))
        self.conv3 = nn.Conv3d(in_channels=planes, out_channels=planes * self.expansion, kernel_size=1, stride=1,
                               bias=False)
        self.bn3 = norm_layer(num_features=planes * self.expansion, **({} if norm_kwargs is None else norm_kwargs))
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
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
        out = self.relu(out + identity)
        return out


class SlowFast(nn.Module):
    """SlowFast networks (SlowFast) from
    `"SlowFast Networks for Video Recognition"
    <https://arxiv.org/abs/1812.03982>`_ paper.
    """

    def __init__(self,
                 num_classes,
                 block=Bottleneck,
                 layers=None,
                 num_block_temp_kernel_fast=None,
                 num_block_temp_kernel_slow=None,
                 pretrained=False,
                 pretrained_base=False,
                 feat_ext=False,
                 num_segment=1,
                 num_crop=1,
                 bn_eval=True,
                 bn_frozen=False,
                 partial_bn=False,
                 frozen_stages=-1,
                 dropout_ratio=0.5,
                 init_std=0.01,
                 alpha=8,
                 beta_inv=8,
                 fusion_conv_channel_ratio=2,
                 fusion_kernel_size=5,
                 width_per_group=64,
                 num_groups=1,
                 slow_temporal_stride=16,
                 fast_temporal_stride=2,
                 slow_frames=4,
                 fast_frames=32,
                 norm_layer=BatchNorm3d,
                 norm_kwargs=None,
                 ctx=None,
                 **kwargs):
        super(SlowFast, self).__init__()
        self.num_segment = num_segment
        self.num_crop = num_crop
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.alpha = alpha
        self.beta_inv = beta_inv
        self.fusion_conv_channel_ratio = fusion_conv_channel_ratio
        self.fusion_kernel_size = fusion_kernel_size
        self.width_per_group = width_per_group
        self.num_groups = num_groups
        self.dim_inner = self.num_groups * self.width_per_group
        self.out_dim_ratio = self.beta_inv // self.fusion_conv_channel_ratio
        self.slow_temporal_stride = slow_temporal_stride
        self.fast_temporal_stride = fast_temporal_stride
        self.slow_frames = slow_frames
        self.fast_frames = fast_frames
        self.feat_ext = feat_ext

        # build fast pathway
        self.fast_conv1 = nn.Conv3d(in_channels=3, out_channels=self.width_per_group // self.beta_inv,
                                    kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False)
        self.fast_bn1 = norm_layer(num_features=self.width_per_group // self.beta_inv,
                                   **({} if norm_kwargs is None else norm_kwargs))
        self.fast_relu = nn.ReLU(inplace=True)
        self.fast_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.fast_res2 = self._make_layer_fast(inplanes=self.width_per_group // self.beta_inv,
                                               planes=self.dim_inner // self.beta_inv,
                                               num_blocks=layers[0],
                                               head_conv=3,
                                               norm_layer=norm_layer,
                                               norm_kwargs=norm_kwargs,
                                               layer_name='fast_res2_')
        self.fast_res3 = self._make_layer_fast(inplanes=self.width_per_group * 4 // self.beta_inv,
                                               planes=self.dim_inner * 2 // self.beta_inv,
                                               num_blocks=layers[1],
                                               strides=2,
                                               head_conv=3,
                                               norm_layer=norm_layer,
                                               norm_kwargs=norm_kwargs,
                                               layer_name='fast_res3_')
        self.fast_res4 = self._make_layer_fast(inplanes=self.width_per_group * 8 // self.beta_inv,
                                               planes=self.dim_inner * 4 // self.beta_inv,
                                               num_blocks=layers[2],
                                               num_block_temp_kernel_fast=num_block_temp_kernel_fast,
                                               strides=2,
                                               head_conv=3,
                                               norm_layer=norm_layer,
                                               norm_kwargs=norm_kwargs,
                                               layer_name='fast_res4_')
        self.fast_res5 = self._make_layer_fast(inplanes=self.width_per_group * 16 // self.beta_inv,
                                               planes=self.dim_inner * 8 // self.beta_inv,
                                               num_blocks=layers[3],
                                               strides=2,
                                               head_conv=3,
                                               norm_layer=norm_layer,
                                               norm_kwargs=norm_kwargs,
                                               layer_name='fast_res5_')

        # build lateral connections
        self.lateral_p1 = nn.Sequential(
            nn.Conv3d(in_channels=self.width_per_group // self.beta_inv,
                      out_channels=self.width_per_group // self.beta_inv * self.fusion_conv_channel_ratio,
                      kernel_size=(self.fusion_kernel_size, 1, 1),
                      stride=(self.alpha, 1, 1),
                      padding=(self.fusion_kernel_size // 2, 0, 0),
                      bias=False),
            norm_layer(num_features=self.width_per_group // self.beta_inv * self.fusion_conv_channel_ratio,
                       **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(inplace=True)
        )

        self.lateral_res2 = nn.Sequential(
            nn.Conv3d(in_channels=self.width_per_group * 4 // self.beta_inv,
                      out_channels=self.width_per_group * 4 // self.beta_inv * self.fusion_conv_channel_ratio,
                      kernel_size=(self.fusion_kernel_size, 1, 1),
                      stride=(self.alpha, 1, 1),
                      padding=(self.fusion_kernel_size // 2, 0, 0),
                      bias=False),
            norm_layer(num_features=self.width_per_group * 4 // self.beta_inv * self.fusion_conv_channel_ratio,
                       **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(inplace=True)
        )

        self.lateral_res3 = nn.Sequential(
            nn.Conv3d(in_channels=self.width_per_group * 8 // self.beta_inv,
                      out_channels=self.width_per_group * 8 // self.beta_inv * self.fusion_conv_channel_ratio,
                      kernel_size=(self.fusion_kernel_size, 1, 1),
                      stride=(self.alpha, 1, 1),
                      padding=(self.fusion_kernel_size // 2, 0, 0),
                      bias=False),
            norm_layer(num_features=self.width_per_group * 8 // self.beta_inv * self.fusion_conv_channel_ratio,
                       **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(inplace=True)
        )

        self.lateral_res4 = nn.Sequential(
            nn.Conv3d(in_channels=self.width_per_group * 16 // self.beta_inv,
                      out_channels=self.width_per_group * 16 // self.beta_inv * self.fusion_conv_channel_ratio,
                      kernel_size=(self.fusion_kernel_size, 1, 1),
                      stride=(self.alpha, 1, 1),
                      padding=(self.fusion_kernel_size // 2, 0, 0),
                      bias=False),
            norm_layer(num_features=self.width_per_group * 16 // self.beta_inv * self.fusion_conv_channel_ratio,
                       **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(inplace=True)
        )

        # build slow pathway
        self.slow_conv1 = nn.Conv3d(in_channels=3, out_channels=self.width_per_group,
                                    kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.slow_bn1 = norm_layer(num_features=self.width_per_group,
                                   **({} if norm_kwargs is None else norm_kwargs))
        self.slow_relu = nn.ReLU(inplace=True)
        self.slow_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.slow_res2 = self._make_layer_slow(
            inplanes=self.width_per_group + self.width_per_group // self.out_dim_ratio,
            planes=self.dim_inner,
            num_blocks=layers[0],
            head_conv=1,
            norm_layer=norm_layer,
            norm_kwargs=norm_kwargs,
            layer_name='slow_res2_')
        self.slow_res3 = self._make_layer_slow(
            inplanes=self.width_per_group * 4 + self.width_per_group * 4 // self.out_dim_ratio,
            planes=self.dim_inner * 2,
            num_blocks=layers[1],
            strides=2,
            head_conv=1,
            norm_layer=norm_layer,
            norm_kwargs=norm_kwargs,
            layer_name='slow_res3_')
        self.slow_res4 = self._make_layer_slow(
            inplanes=self.width_per_group * 8 + self.width_per_group * 8 // self.out_dim_ratio,
            planes=self.dim_inner * 4,
            num_blocks=layers[2],
            num_block_temp_kernel_slow=num_block_temp_kernel_slow,
            strides=2,
            head_conv=3,
            norm_layer=norm_layer,
            norm_kwargs=norm_kwargs,
            layer_name='slow_res4_')
        self.slow_res5 = self._make_layer_slow(
            inplanes=self.width_per_group * 16 + self.width_per_group * 16 // self.out_dim_ratio,
            planes=self.dim_inner * 8,
            num_blocks=layers[3],
            strides=2,
            head_conv=3,
            norm_layer=norm_layer,
            norm_kwargs=norm_kwargs,
            layer_name='slow_res5_')

        # build classifier
        self.avg = nn.AdaptiveAvgPool3d(1)
        self.dp = nn.Dropout(p=self.dropout_ratio)
        self.feat_dim = self.width_per_group * 32 // self.beta_inv + self.width_per_group * 32
        self.fc = nn.Linear(in_features=self.feat_dim, out_features=num_classes, bias=True)

    def forward(self, x):
        fast_input = x
        slow_input = x[:, :, ::self.slow_temporal_stride // 2, :, :]

        fast, lateral = self.FastPath(fast_input)
        slow = self.SlowPath(slow_input, lateral)
        x = torch.cat((slow, fast), dim=1)  # bx2304

        if self.feat_ext:
            return x

        x = self.dp(x)
        x = self.fc(x)  # bxnum_classes
        return x

    def SlowPath(self, x, lateral):
        x = self.slow_conv1(x)  # bx64x4x112x112, input is bx3x4x224x224
        x = self.slow_bn1(x)
        x = self.slow_relu(x)
        pool1 = self.slow_maxpool(x)  # bx64x4x56x56
        pool1_lat = torch.cat((pool1, lateral[0]), dim=1)  # bx80x4x56x56

        res2 = self.slow_res2(pool1_lat)  # bx256x4x56x56
        res2_lat = torch.cat((res2, lateral[1]), dim=1)  # bx320x4x56x56

        res3 = self.slow_res3(res2_lat)  # bx512x4x28x28
        res3_lat = torch.cat((res3, lateral[2]), dim=1)  # bx640x4x28x28

        res4 = self.slow_res4(res3_lat)  # bx1024x4x14x14
        res4_lat = torch.cat((res4, lateral[3]), dim=1)  # bx1280x4x14x14

        res5 = self.slow_res5(res4_lat)  # bx2048x4x7x7
        out = self.avg(res5)  # bx2048x1x1x1
        out = out.view(out.shape[0], out.shape[1])  # bx2048
        return out

    def FastPath(self, x):
        lateral = []
        x = self.fast_conv1(x)  # bx8x32x112x112, input is bx3x32x224x224
        x = self.fast_bn1(x)
        x = self.fast_relu(x)
        pool1 = self.fast_maxpool(x)  # bx8x32x56x56
        lateral_p = self.lateral_p1(pool1)  # bx16x4x56x56
        lateral.append(lateral_p)

        res2 = self.fast_res2(pool1)  # bx32x32x56x56
        lateral_res2 = self.lateral_res2(res2)  # bx64x4x56x56
        lateral.append(lateral_res2)

        res3 = self.fast_res3(res2)  # bx64x32x28x28
        lateral_res3 = self.lateral_res3(res3)  # bx128x4x28x28
        lateral.append(lateral_res3)

        res4 = self.fast_res4(res3)  # bx128x32x14x14
        lateral_res4 = self.lateral_res4(res4)  # bx256x4x14x14
        lateral.append(lateral_res4)

        res5 = self.fast_res5(res4)  # bx256x32x7x7
        out = self.avg(res5)  # bx256x1x1x1
        out = out.view(out.shape[0], out.shape[1])  # bx256
        return out, lateral

    def _make_layer_fast(self,
                         inplanes,
                         planes,
                         num_blocks,
                         num_block_temp_kernel_fast=None,
                         block=Bottleneck,
                         strides=1,
                         head_conv=1,
                         norm_layer=BatchNorm3d,
                         norm_kwargs=None,
                         layer_name=''):
        downsample = None
        if strides != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(in_channels=inplanes,
                          out_channels=planes * block.expansion,
                          kernel_size=1,
                          stride=(1, strides, strides),
                          bias=False),
                norm_layer(num_features=planes * block.expansion, **({} if norm_kwargs is None else norm_kwargs))
            )

        layers = []
        cnt = 0
        layers.append(block(inplanes=inplanes,
                            planes=planes,
                            strides=strides,
                            downsample=downsample,
                            head_conv=head_conv,
                            layer_name='block%d_' % cnt))
        inplanes = planes * block.expansion
        cnt += 1
        for _ in range(1, num_blocks):
            if num_block_temp_kernel_fast is not None:
                if cnt < num_block_temp_kernel_fast:
                    layers.append(block(inplanes=inplanes,
                                        planes=planes,
                                        head_conv=head_conv,
                                        layer_name='block%d_' % cnt))
                else:
                    layers.append(block(inplanes=inplanes,
                                        planes=planes,
                                        head_conv=1,
                                        layer_name='block%d_' % cnt))
            else:
                layers.append(block(inplanes=inplanes,
                                    planes=planes,
                                    head_conv=head_conv,
                                    layer_name='block%d_' % cnt))
            cnt += 1
        return nn.Sequential(*layers)

    def _make_layer_slow(self,
                         inplanes,
                         planes,
                         num_blocks,
                         num_block_temp_kernel_slow=None,
                         block=Bottleneck,
                         strides=1,
                         head_conv=1,
                         norm_layer=BatchNorm3d,
                         norm_kwargs=None,
                         layer_name=''):
        downsample = None
        if strides != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(in_channels=inplanes,
                          out_channels=planes * block.expansion,
                          kernel_size=1,
                          stride=(1, strides, strides),
                          bias=False),
                norm_layer(num_features=planes * block.expansion, **({} if norm_kwargs is None else norm_kwargs))
            )

        layers = []
        cnt = 0
        layers.append(block(inplanes=inplanes,
                            planes=planes,
                            strides=strides,
                            downsample=downsample,
                            head_conv=head_conv,
                            layer_name='block%d_' % cnt))
        inplanes = planes * block.expansion
        cnt += 1
        for _ in range(1, num_blocks):
            if num_block_temp_kernel_slow is not None:
                if cnt < num_block_temp_kernel_slow:
                    layers.append(block(inplanes=inplanes,
                                        planes=planes,
                                        head_conv=head_conv,
                                        layer_name='block%d_' % cnt))
                else:
                    layers.append(block(inplanes=inplanes,
                                        planes=planes,
                                        head_conv=1,
                                        layer_name='block%d_' % cnt))
            else:
                layers.append(block(inplanes=inplanes,
                                    planes=planes,
                                    head_conv=head_conv,
                                    layer_name='block%d_' % cnt))
            cnt += 1
        return nn.Sequential(*layers)


def slowfast_4x16_resnet50_kinetics400(cfg):
    model = SlowFast(num_classes=cfg.CONFIG.DATA.NUM_CLASSES,
                     layers=[3, 4, 6, 3],
                     pretrained=cfg.CONFIG.MODEL.PRETRAINED,
                     pretrained_base=cfg.CONFIG.MODEL.PRETRAINED_BASE,
                     feat_ext=cfg.CONFIG.INFERENCE.FEAT,
                     num_segment=cfg.CONFIG.DATA.NUM_SEGMENT,
                     num_crop=cfg.CONFIG.DATA.NUM_CROP,
                     alpha=8,
                     beta_inv=8,
                     fusion_conv_channel_ratio=2,
                     fusion_kernel_size=5,
                     width_per_group=64,
                     num_groups=1,
                     slow_temporal_stride=16,
                     fast_temporal_stride=2,
                     slow_frames=4,
                     fast_frames=32,
                     bn_eval=cfg.CONFIG.MODEL.BN_EVAL,
                     partial_bn=cfg.CONFIG.MODEL.PARTIAL_BN,
                     bn_frozen=cfg.CONFIG.MODEL.BN_FROZEN)

    if cfg.CONFIG.MODEL.PRETRAINED:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('slowfast_4x16_resnet50_kinetics400',
                                                        tag=cfg.CONFIG.MODEL.PRETRAINED)))
    return model


def slowfast_8x8_resnet50_kinetics400(cfg):
    model = SlowFast(num_classes=cfg.CONFIG.DATA.NUM_CLASSES,
                     layers=[3, 4, 6, 3],
                     pretrained=cfg.CONFIG.MODEL.PRETRAINED,
                     pretrained_base=cfg.CONFIG.MODEL.PRETRAINED_BASE,
                     feat_ext=cfg.CONFIG.INFERENCE.FEAT,
                     num_segment=cfg.CONFIG.DATA.NUM_SEGMENT,
                     num_crop=cfg.CONFIG.DATA.NUM_CROP,
                     alpha=4,
                     beta_inv=8,
                     fusion_conv_channel_ratio=2,
                     fusion_kernel_size=7,
                     width_per_group=64,
                     num_groups=1,
                     slow_temporal_stride=8,
                     fast_temporal_stride=2,
                     slow_frames=8,
                     fast_frames=32,
                     bn_eval=cfg.CONFIG.MODEL.BN_EVAL,
                     partial_bn=cfg.CONFIG.MODEL.PARTIAL_BN,
                     bn_frozen=cfg.CONFIG.MODEL.BN_FROZEN)

    if cfg.CONFIG.MODEL.PRETRAINED:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('slowfast_8x8_resnet50_kinetics400',
                                                        tag=cfg.CONFIG.MODEL.PRETRAINED)))
    return model


def slowfast_4x16_resnet101_kinetics400(cfg):
    model = SlowFast(num_classes=cfg.CONFIG.DATA.NUM_CLASSES,
                     layers=[3, 4, 23, 3],
                     pretrained=cfg.CONFIG.MODEL.PRETRAINED,
                     pretrained_base=cfg.CONFIG.MODEL.PRETRAINED_BASE,
                     feat_ext=cfg.CONFIG.INFERENCE.FEAT,
                     num_segment=cfg.CONFIG.DATA.NUM_SEGMENT,
                     num_crop=cfg.CONFIG.DATA.NUM_CROP,
                     alpha=8,
                     beta_inv=8,
                     fusion_conv_channel_ratio=2,
                     fusion_kernel_size=5,
                     width_per_group=64,
                     num_groups=1,
                     slow_temporal_stride=16,
                     fast_temporal_stride=2,
                     slow_frames=4,
                     fast_frames=32,
                     bn_eval=cfg.CONFIG.MODEL.BN_EVAL,
                     partial_bn=cfg.CONFIG.MODEL.PARTIAL_BN,
                     bn_frozen=cfg.CONFIG.MODEL.BN_FROZEN)
    return model


def slowfast_8x8_resnet101_kinetics400(cfg):
    model = SlowFast(num_classes=cfg.CONFIG.DATA.NUM_CLASSES,
                     layers=[3, 4, 23, 3],
                     pretrained=cfg.CONFIG.MODEL.PRETRAINED,
                     pretrained_base=cfg.CONFIG.MODEL.PRETRAINED_BASE,
                     feat_ext=cfg.CONFIG.INFERENCE.FEAT,
                     num_segment=cfg.CONFIG.DATA.NUM_SEGMENT,
                     num_crop=cfg.CONFIG.DATA.NUM_CROP,
                     alpha=4,
                     beta_inv=8,
                     fusion_conv_channel_ratio=2,
                     fusion_kernel_size=5,
                     width_per_group=64,
                     num_groups=1,
                     slow_temporal_stride=8,
                     fast_temporal_stride=2,
                     slow_frames=8,
                     fast_frames=32,
                     bn_eval=cfg.CONFIG.MODEL.BN_EVAL,
                     partial_bn=cfg.CONFIG.MODEL.PARTIAL_BN,
                     bn_frozen=cfg.CONFIG.MODEL.BN_FROZEN)

    if cfg.CONFIG.MODEL.PRETRAINED:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('slowfast_8x8_resnet101_kinetics400',
                                                        tag=cfg.CONFIG.MODEL.PRETRAINED)))
    return model


def slowfast_16x8_resnet101_kinetics400(cfg):
    model = SlowFast(num_classes=cfg.CONFIG.DATA.NUM_CLASSES,
                     layers=[3, 4, 23, 3],
                     pretrained=cfg.CONFIG.MODEL.PRETRAINED,
                     pretrained_base=cfg.CONFIG.MODEL.PRETRAINED_BASE,
                     feat_ext=cfg.CONFIG.INFERENCE.FEAT,
                     num_segment=cfg.CONFIG.DATA.NUM_SEGMENT,
                     num_crop=cfg.CONFIG.DATA.NUM_CROP,
                     alpha=4,
                     beta_inv=8,
                     fusion_conv_channel_ratio=2,
                     fusion_kernel_size=5,
                     width_per_group=64,
                     num_groups=1,
                     slow_temporal_stride=8,
                     fast_temporal_stride=2,
                     slow_frames=16,
                     fast_frames=64,
                     bn_eval=cfg.CONFIG.MODEL.BN_EVAL,
                     partial_bn=cfg.CONFIG.MODEL.PARTIAL_BN,
                     bn_frozen=cfg.CONFIG.MODEL.BN_FROZEN)
    return model


def slowfast_16x8_resnet101_50_50_kinetics400(cfg):
    model = SlowFast(num_classes=cfg.CONFIG.DATA.NUM_CLASSES,
                     layers=[3, 4, 23, 3],
                     pretrained=cfg.CONFIG.MODEL.PRETRAINED,
                     pretrained_base=cfg.CONFIG.MODEL.PRETRAINED_BASE,
                     feat_ext=cfg.CONFIG.INFERENCE.FEAT,
                     num_segment=cfg.CONFIG.DATA.NUM_SEGMENT,
                     num_crop=cfg.CONFIG.DATA.NUM_CROP,
                     num_block_temp_kernel_fast=6,
                     num_block_temp_kernel_slow=6,
                     alpha=4,
                     beta_inv=8,
                     fusion_conv_channel_ratio=2,
                     fusion_kernel_size=5,
                     width_per_group=64,
                     num_groups=1,
                     slow_temporal_stride=8,
                     fast_temporal_stride=2,
                     slow_frames=16,
                     fast_frames=64,
                     bn_eval=cfg.CONFIG.MODEL.BN_EVAL,
                     partial_bn=cfg.CONFIG.MODEL.PARTIAL_BN,
                     bn_frozen=cfg.CONFIG.MODEL.BN_FROZEN)
    return model


def slowfast_16x8_resnet50_sthsthv2(cfg):
    model = SlowFast(num_classes=cfg.CONFIG.DATA.NUM_CLASSES,
                     layers=[3, 4, 6, 3],
                     pretrained=cfg.CONFIG.MODEL.PRETRAINED,
                     pretrained_base=cfg.CONFIG.MODEL.PRETRAINED_BASE,
                     feat_ext=cfg.CONFIG.INFERENCE.FEAT,
                     num_segment=cfg.CONFIG.DATA.NUM_SEGMENT,
                     num_crop=cfg.CONFIG.DATA.NUM_CROP,
                     alpha=4,
                     beta_inv=8,
                     fusion_conv_channel_ratio=2,
                     fusion_kernel_size=7,
                     width_per_group=64,
                     num_groups=1,
                     slow_temporal_stride=8,
                     fast_temporal_stride=2,
                     slow_frames=16,
                     fast_frames=64,
                     bn_eval=cfg.CONFIG.MODEL.BN_EVAL,
                     partial_bn=cfg.CONFIG.MODEL.PARTIAL_BN,
                     bn_frozen=cfg.CONFIG.MODEL.BN_FROZEN)

    if cfg.CONFIG.MODEL.PRETRAINED:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('slowfast_16x8_resnet50_sthsthv2',
                                                        tag=cfg.CONFIG.MODEL.PRETRAINED)))
    return model
