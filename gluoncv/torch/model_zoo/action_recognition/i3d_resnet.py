# pylint: disable=missing-function-docstring, missing-class-docstring, unused-argument, line-too-long
"""
Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
CVPR 2017, https://arxiv.org/abs/1705.07750
"""
import torch
import torch.nn as nn
from torch.nn import BatchNorm3d

from .non_local import build_nonlocal_block


__all__ = ['I3D_ResNetV1', 'i3d_resnet50_v1_kinetics400', 'i3d_resnet101_v1_kinetics400',
           'i3d_nl5_resnet50_v1_kinetics400', 'i3d_nl10_resnet50_v1_kinetics400',
           'i3d_nl5_resnet101_v1_kinetics400', 'i3d_nl10_resnet101_v1_kinetics400',
           'i3d_resnet50_v1_sthsthv2']


def conv3x3x3(in_planes, out_planes, spatial_stride=1, temporal_stride=1, dilation=1):
    "3x3x3 convolution with padding"
    return nn.Conv3d(in_channels=in_planes,
                     out_channels=out_planes,
                     kernel_size=3,
                     stride=(temporal_stride, spatial_stride, spatial_stride),
                     dilation=dilation,
                     bias=False)


def conv1x3x3(in_planes, out_planes, spatial_stride=1, temporal_stride=1, dilation=1):
    "1x3x3 convolution with padding"
    return nn.Conv3d(in_channels=in_planes,
                     out_channels=out_planes,
                     kernel_size=(1, 3, 3),
                     stride=(temporal_stride, spatial_stride, spatial_stride),
                     padding=(0, dilation, dilation),
                     dilation=dilation,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 spatial_stride=1,
                 temporal_stride=1,
                 dilation=1,
                 downsample=None,
                 if_inflate=True,
                 inflate_style=None,
                 norm_layer=BatchNorm3d,
                 norm_kwargs=None,
                 layer_name='',
                 **kwargs):
        super(BasicBlock, self).__init__()

        if if_inflate:
            self.conv1 = conv3x3x3(inplanes, planes, spatial_stride, temporal_stride, dilation)
        else:
            self.conv1 = conv1x3x3(inplanes, planes, spatial_stride, temporal_stride, dilation)
        self.bn1 = norm_layer(**({} if norm_kwargs is None else norm_kwargs))
        self.relu = nn.ReLU(inplace=False)
        if if_inflate:
            self.conv2 = conv3x3x3(planes, planes)
        else:
            self.conv2 = conv1x3x3(planes, planes)
        self.bn2 = norm_layer(**({} if norm_kwargs is None else norm_kwargs))

        self.downsample = downsample
        self.spatial_stride = spatial_stride
        self.temporal_stride = temporal_stride
        self.dilation = dilation

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 spatial_stride=1,
                 temporal_stride=1,
                 dilation=1,
                 downsample=None,
                 if_inflate=True,
                 inflate_style='3x1x1',
                 if_nonlocal=True,
                 nonlocal_cfg=None,
                 norm_layer=BatchNorm3d,
                 norm_kwargs=None,
                 layer_name='',
                 **kwargs):
        super(Bottleneck, self).__init__()
        assert inflate_style in ['3x1x1', '3x3x3']
        self.inplanes = inplanes
        self.planes = planes

        self.conv1_stride = 1
        self.conv2_stride = spatial_stride
        self.conv1_stride_t = 1
        self.conv2_stride_t = temporal_stride

        if if_inflate:
            if inflate_style == '3x1x1':
                self.conv1 = nn.Conv3d(in_channels=inplanes,
                                       out_channels=planes,
                                       kernel_size=(3, 1, 1),
                                       stride=(self.conv1_stride_t, self.conv1_stride, self.conv1_stride),
                                       padding=(1, 0, 0),
                                       bias=False)
                self.conv2 = nn.Conv3d(in_channels=planes,
                                       out_channels=planes,
                                       kernel_size=(1, 3, 3),
                                       stride=(self.conv2_stride_t, self.conv2_stride, self.conv2_stride),
                                       padding=(0, dilation, dilation),
                                       dilation=(1, dilation, dilation),
                                       bias=False)
            else:
                self.conv1 = nn.Conv3d(in_channels=inplanes,
                                       out_channels=planes,
                                       kernel_size=1,
                                       stride=(self.conv1_stride_t, self.conv1_stride, self.conv1_stride),
                                       bias=False)
                self.conv2 = nn.Conv3d(in_channels=planes,
                                       out_channels=planes,
                                       kernel_size=3,
                                       stride=(self.conv2_stride_t, self.conv2_stride, self.conv2_stride),
                                       padding=(1, dilation, dilation),
                                       dilation=(1, dilation, dilation),
                                       bias=False)
        else:
            self.conv1 = nn.Conv3d(in_channels=inplanes,
                                   out_channels=planes,
                                   kernel_size=1,
                                   stride=(1, self.conv1_stride, self.conv1_stride),
                                   bias=False)
            self.conv2 = nn.Conv3d(in_channels=planes,
                                   out_channels=planes,
                                   kernel_size=(1, 3, 3),
                                   stride=(1, self.conv2_stride, self.conv2_stride),
                                   padding=(0, dilation, dilation),
                                   dilation=(1, dilation, dilation),
                                   bias=False)

        self.bn1 = norm_layer(num_features=planes, **({} if norm_kwargs is None else norm_kwargs))
        self.bn2 = norm_layer(num_features=planes, **({} if norm_kwargs is None else norm_kwargs))
        self.conv3 = nn.Conv3d(in_channels=planes,
                               out_channels=planes * self.expansion,
                               kernel_size=1,
                               bias=False)
        self.bn3 = norm_layer(num_features=planes * self.expansion, **({} if norm_kwargs is None else norm_kwargs))
        self.downsample = downsample

        self.spatial_tride = spatial_stride
        self.temporal_tride = temporal_stride
        self.dilation = dilation

        if if_nonlocal and nonlocal_cfg is not None:
            nonlocal_cfg_ = nonlocal_cfg.copy()
            nonlocal_cfg_['in_channels'] = planes * self.expansion
            self.nonlocal_block = build_nonlocal_block(nonlocal_cfg_)
        else:
            self.nonlocal_block = None

        self.relu = nn.ReLU(inplace=False)

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
        out = out + identity
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
                   inflate_freq=1,
                   inflate_style='3x1x1',
                   nonlocal_freq=1,
                   nonlocal_cfg=None,
                   norm_layer=BatchNorm3d,
                   norm_kwargs=None,
                   layer_name=''):
    inflate_freq = inflate_freq if not isinstance(inflate_freq, int) else (inflate_freq,) * blocks
    nonlocal_freq = nonlocal_freq if not isinstance(nonlocal_freq, int) else (nonlocal_freq,) * blocks
    assert len(inflate_freq) == blocks
    assert len(nonlocal_freq) == blocks

    downsample = None
    if spatial_stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv3d(in_channels=inplanes,
                      out_channels=planes * block.expansion,
                      kernel_size=1,
                      stride=(temporal_stride, spatial_stride, spatial_stride),
                      bias=False),
            norm_layer(num_features=planes * block.expansion, **({} if norm_kwargs is None else norm_kwargs)))

    layers = []
    cnt = 0
    layers.append(block(inplanes=inplanes,
                        planes=planes,
                        spatial_stride=spatial_stride,
                        temporal_stride=temporal_stride,
                        dilation=dilation,
                        downsample=downsample,
                        if_inflate=(inflate_freq[0] == 1),
                        inflate_style=inflate_style,
                        if_nonlocal=(nonlocal_freq[0] == 1),
                        nonlocal_cfg=nonlocal_cfg,
                        layer_name='%d_' % cnt))

    cnt += 1
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes=inplanes,
                            planes=planes,
                            spatial_stride=1,
                            temporal_stride=1,
                            dilation=dilation,
                            if_inflate=(inflate_freq[i] == 1),
                            inflate_style=inflate_style,
                            if_nonlocal=(nonlocal_freq[i] == 1),
                            nonlocal_cfg=nonlocal_cfg,
                            layer_name='%d_' % cnt))
        cnt += 1
    return nn.Sequential(*layers)


class I3D_ResNetV1(nn.Module):
    """ResNet_I3D backbone.
    Inflated 3D model (I3D) from
    `"Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset"
    <https://arxiv.org/abs/1705.07750>`_ paper.
    Args:
        depth (int): Depth of ResNet, from {18, 34, 50, 101, 152}.
        num_stages (int): ResNet stages, normally 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        bn_eval (bool): Whether to set BN layers to eval mode, namely, freeze
            running stats (mean and var).
        bn_frozen (bool): Whether to freeze weight and bias of BN layers.
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 num_classes,
                 depth,
                 num_stages=4,
                 pretrained=False,
                 pretrained_base=True,
                 feat_ext=False,
                 num_segment=1,
                 num_crop=1,
                 spatial_strides=(1, 2, 2, 2),
                 temporal_strides=(1, 1, 1, 1),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 conv1_kernel_t=5,
                 conv1_stride_t=2,
                 pool1_kernel_t=1,
                 pool1_stride_t=2,
                 inflate_freq=(1, 1, 1, 1),
                 inflate_stride=(1, 1, 1, 1),
                 inflate_style='3x1x1',
                 nonlocal_stages=(-1,),
                 nonlocal_freq=(0, 1, 1, 0),
                 nonlocal_cfg=None,
                 bn_eval=True,
                 bn_frozen=False,
                 partial_bn=False,
                 frozen_stages=-1,
                 dropout_ratio=0.5,
                 init_std=0.01,
                 norm_layer=BatchNorm3d,
                 norm_kwargs=None,
                 ctx=None,
                 **kwargs):
        super(I3D_ResNetV1, self).__init__()

        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))

        self.num_classes = num_classes
        self.depth = depth
        self.num_stages = num_stages
        self.pretrained = pretrained
        self.pretrained_base = pretrained_base
        self.feat_ext = feat_ext
        self.num_segment = num_segment
        self.num_crop = num_crop
        self.spatial_strides = spatial_strides
        self.temporal_strides = temporal_strides
        self.dilations = dilations
        assert len(spatial_strides) == len(temporal_strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.inflate_freqs = inflate_freq if not isinstance(inflate_freq, int) else (inflate_freq,) * num_stages
        self.inflate_style = inflate_style
        self.nonlocal_stages = nonlocal_stages
        self.nonlocal_freqs = nonlocal_freq if not isinstance(nonlocal_freq, int) else (nonlocal_freq,) * num_stages
        self.nonlocal_cfg = nonlocal_cfg
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen
        self.partial_bn = partial_bn
        self.frozen_stages = frozen_stages
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std

        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = 64

        if self.bn_frozen:
            if norm_kwargs is not None:
                norm_kwargs['use_global_stats'] = True
            else:
                norm_kwargs = {}
                norm_kwargs['use_global_stats'] = True

        self.first_stage = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(conv1_kernel_t, 7, 7),
                      stride=(conv1_stride_t, 2, 2), padding=((conv1_kernel_t - 1) // 2, 3, 3),
                      bias=False),
            norm_layer(num_features=64, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(inplace=False),
            nn.MaxPool3d(kernel_size=(pool1_kernel_t, 3, 3), stride=(pool1_stride_t, 2, 2),
                         padding=(pool1_kernel_t // 2, 1, 1)))

        self.pool2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        if self.partial_bn:
            if norm_kwargs is not None:
                norm_kwargs['use_global_stats'] = True
            else:
                norm_kwargs = {}
                norm_kwargs['use_global_stats'] = True

        res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            spatial_stride = spatial_strides[i]
            temporal_stride = temporal_strides[i]
            dilation = dilations[i]
            planes = 64 * 2 ** i
            layer_name = 'layer{}_'.format(i + 1)

            res_layer = make_res_layer(self.block,
                                       self.inplanes,
                                       planes,
                                       num_blocks,
                                       spatial_stride=spatial_stride,
                                       temporal_stride=temporal_stride,
                                       dilation=dilation,
                                       inflate_freq=self.inflate_freqs[i],
                                       inflate_style=self.inflate_style,
                                       nonlocal_freq=self.nonlocal_freqs[i],
                                       nonlocal_cfg=self.nonlocal_cfg if i in self.nonlocal_stages else None,
                                       norm_layer=norm_layer,
                                       norm_kwargs=norm_kwargs,
                                       layer_name=layer_name)
            self.inplanes = planes * self.block.expansion
            res_layers.append(res_layer)

        self.res_layers = nn.Sequential(*res_layers)
        self.feat_dim = self.block.expansion * 64 * 2 ** (len(self.stage_blocks) - 1)

        self.st_avg = nn.AdaptiveAvgPool3d(output_size=1)
        self.dp = nn.Dropout(self.dropout_ratio)
        self.fc = nn.Linear(in_features=self.feat_dim, out_features=num_classes)
        nn.init.normal_(self.fc.weight, 0, self.init_std)
        nn.init.constant_(self.fc.bias, 0)
        self.head = nn.Sequential(self.dp, self.fc)

        self.inflate_weights()

    def inflate_weights(self):
        """Inflate I3D network with its 2D ImageNet pretrained weights."""

        if not self.pretrained_base:
            raise RuntimeError("I3D models need to be inflated. Please set PRETRAINED_BASE to True in config.")

        if self.pretrained_base and not self.pretrained:
            import torchvision
            if self.depth == 50:
                R2D = torchvision.models.resnet50(pretrained=True, progress=True)
            elif self.depth == 101:
                R2D = torchvision.models.resnet101(pretrained=True, progress=True)
            else:
                raise RuntimeError("We only support ResNet50 and ResNet101 for I3D models at this moment.")

            # copy conv1
            conv1 = self.first_stage._modules['0']
            conv1_bn = self.first_stage._modules['1']

            conv1.weight.data.copy_(torch.unsqueeze(R2D.conv1.weight.data, dim=2).repeat(1, 1, 5, 1, 1))
            conv1_bn.weight.data.copy_(R2D.bn1.weight.data)
            conv1_bn.bias.data.copy_(R2D.bn1.bias.data)
            conv1_bn.running_mean.data.copy_(R2D.bn1.running_mean.data)
            conv1_bn.running_var.data.copy_(R2D.bn1.running_var.data)

            res2 = self.res_layers._modules['0']
            res3 = self.res_layers._modules['1']
            res4 = self.res_layers._modules['2']
            res5 = self.res_layers._modules['3']

            stages = [res2, res3, res4, res5]

            R2Dlayers = [R2D.layer1, R2D.layer2, R2D.layer3, R2D.layer4]

            for s, _ in enumerate(stages):
                res = stages[s]._modules
                count = 0

                for k, block in res.items():
                    if block.conv1.weight.data.shape[2] > 1:
                        block.conv1.weight.data.copy_(
                            torch.unsqueeze(R2Dlayers[s]._modules[str(k)].conv1.weight.data, dim=2).repeat(1, 1, 3, 1, 1))
                    else:
                        block.conv1.weight.data.copy_(
                            torch.unsqueeze(R2Dlayers[s]._modules[str(k)].conv1.weight.data, dim=2))
                    block.conv2.weight.data.copy_(torch.unsqueeze(R2Dlayers[s]._modules[str(k)].conv2.weight.data, dim=2))
                    block.conv3.weight.data.copy_(torch.unsqueeze(R2Dlayers[s]._modules[str(k)].conv3.weight.data, dim=2))

                    block.bn1.weight.data.copy_(R2Dlayers[s]._modules[str(k)].bn1.weight.data)
                    block.bn1.bias.data.copy_(R2Dlayers[s]._modules[str(k)].bn1.bias.data)
                    block.bn1.running_mean.data.copy_(R2Dlayers[s]._modules[str(k)].bn1.running_mean.data)
                    block.bn1.running_var.data.copy_(R2Dlayers[s]._modules[str(k)].bn1.running_var.data)

                    block.bn2.weight.data.copy_(R2Dlayers[s]._modules[str(k)].bn2.weight.data)
                    block.bn2.bias.data.copy_(R2Dlayers[s]._modules[str(k)].bn2.bias.data)
                    block.bn2.running_mean.data.copy_(R2Dlayers[s]._modules[str(k)].bn2.running_mean.data)
                    block.bn2.running_var.data.copy_(R2Dlayers[s]._modules[str(k)].bn2.running_var.data)

                    block.bn3.weight.data.copy_(R2Dlayers[s]._modules[str(k)].bn3.weight.data)
                    block.bn3.bias.data.copy_(R2Dlayers[s]._modules[str(k)].bn3.bias.data)
                    block.bn3.running_mean.data.copy_(R2Dlayers[s]._modules[str(k)].bn3.running_mean.data)
                    block.bn3.running_var.data.copy_(R2Dlayers[s]._modules[str(k)].bn3.running_var.data)

                    if block.downsample is not None:
                        down_conv = block.downsample._modules["0"]
                        down_bn = block.downsample._modules["1"]

                        down_conv.weight.data.copy_(
                            torch.unsqueeze(R2Dlayers[s]._modules[str(k)].downsample._modules['0'].weight.data, dim=2))
                        down_bn.weight.data.copy_(R2Dlayers[s]._modules[str(k)].downsample._modules['1'].weight.data)
                        down_bn.bias.data.copy_(R2Dlayers[s]._modules[str(k)].downsample._modules['1'].bias.data)
                        down_bn.running_mean.data.copy_(
                            R2Dlayers[s]._modules[str(k)].downsample._modules['1'].running_mean.data)
                        down_bn.running_var.data.copy_(
                            R2Dlayers[s]._modules[str(k)].downsample._modules['1'].running_var.data)
                    count += 1
        print("I3D weights inflated from pretrained C2D.")

    def forward(self, x):
        bs, _, _, _, _ = x.shape
        x = self.first_stage(x)

        for i, res_layer in enumerate(self.res_layers):
            x = res_layer(x)
            if i == 0:
                x = self.pool2(x)

        # spatial temporal average
        pooled_feat = self.st_avg(x)
        x = pooled_feat.view(bs, -1)
        x = self.head(x)
        return x


def i3d_resnet50_v1_kinetics400(cfg):
    model = I3D_ResNetV1(num_classes=cfg.CONFIG.DATA.NUM_CLASSES,
                         depth=50,
                         pretrained=cfg.CONFIG.MODEL.PRETRAINED,
                         pretrained_base=cfg.CONFIG.MODEL.PRETRAINED_BASE,
                         feat_ext=cfg.CONFIG.INFERENCE.FEAT,
                         num_segment=cfg.CONFIG.DATA.NUM_SEGMENT,
                         num_crop=cfg.CONFIG.DATA.NUM_CROP,
                         out_indices=[3],
                         inflate_freq=((1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 1, 0)),
                         bn_eval=cfg.CONFIG.MODEL.BN_EVAL,
                         partial_bn=cfg.CONFIG.MODEL.PARTIAL_BN,
                         bn_frozen=cfg.CONFIG.MODEL.BN_FROZEN)

    if cfg.CONFIG.MODEL.PRETRAINED:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('i3d_resnet50_v1_kinetics400',
                                                        tag=cfg.CONFIG.MODEL.PRETRAINED)))
    return model


def i3d_resnet101_v1_kinetics400(cfg):
    model = I3D_ResNetV1(num_classes=cfg.CONFIG.DATA.NUM_CLASSES,
                         depth=101,
                         pretrained=cfg.CONFIG.MODEL.PRETRAINED,
                         pretrained_base=cfg.CONFIG.MODEL.PRETRAINED_BASE,
                         feat_ext=cfg.CONFIG.INFERENCE.FEAT,
                         num_segment=cfg.CONFIG.DATA.NUM_SEGMENT,
                         num_crop=cfg.CONFIG.DATA.NUM_CROP,
                         out_indices=[3],
                         inflate_freq=((1, 1, 1),
                                       (1, 0, 1, 0),
                                       (1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1),
                                       (0, 1, 0)),
                         bn_eval=cfg.CONFIG.MODEL.BN_EVAL,
                         partial_bn=cfg.CONFIG.MODEL.PARTIAL_BN,
                         bn_frozen=cfg.CONFIG.MODEL.BN_FROZEN)

    if cfg.CONFIG.MODEL.PRETRAINED:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('i3d_resnet101_v1_kinetics400',
                                                        tag=cfg.CONFIG.MODEL.PRETRAINED)))
    return model


def i3d_nl5_resnet50_v1_kinetics400(cfg):
    model = I3D_ResNetV1(num_classes=cfg.CONFIG.DATA.NUM_CLASSES,
                         depth=50,
                         pretrained=cfg.CONFIG.MODEL.PRETRAINED,
                         pretrained_base=cfg.CONFIG.MODEL.PRETRAINED_BASE,
                         feat_ext=cfg.CONFIG.INFERENCE.FEAT,
                         num_segment=cfg.CONFIG.DATA.NUM_SEGMENT,
                         num_crop=cfg.CONFIG.DATA.NUM_CROP,
                         out_indices=[3],
                         inflate_freq=((1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 1, 0)),
                         bn_eval=cfg.CONFIG.MODEL.BN_EVAL,
                         partial_bn=cfg.CONFIG.MODEL.PARTIAL_BN,
                         bn_frozen=cfg.CONFIG.MODEL.BN_FROZEN,
                         nonlocal_stages=(1, 2),
                         nonlocal_cfg=dict(nonlocal_type="gaussian"),
                         nonlocal_freq=((0, 0, 0), (0, 1, 0, 1), (0, 1, 0, 1, 0, 1), (0, 0, 0)))

    if cfg.CONFIG.MODEL.PRETRAINED:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('i3d_nl5_resnet50_v1_kinetics400',
                                                        tag=cfg.CONFIG.MODEL.PRETRAINED)))
    return model


def i3d_nl10_resnet50_v1_kinetics400(cfg):
    model = I3D_ResNetV1(num_classes=cfg.CONFIG.DATA.NUM_CLASSES,
                         depth=50,
                         pretrained=cfg.CONFIG.MODEL.PRETRAINED,
                         pretrained_base=cfg.CONFIG.MODEL.PRETRAINED_BASE,
                         feat_ext=cfg.CONFIG.INFERENCE.FEAT,
                         num_segment=cfg.CONFIG.DATA.NUM_SEGMENT,
                         num_crop=cfg.CONFIG.DATA.NUM_CROP,
                         out_indices=[3],
                         inflate_freq=((1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 1, 0)),
                         bn_eval=cfg.CONFIG.MODEL.BN_EVAL,
                         partial_bn=cfg.CONFIG.MODEL.PARTIAL_BN,
                         bn_frozen=cfg.CONFIG.MODEL.BN_FROZEN,
                         nonlocal_stages=(1, 2),
                         nonlocal_cfg=dict(nonlocal_type="gaussian"),
                         nonlocal_freq=((0, 0, 0), (1, 1, 1, 1), (1, 1, 1, 1, 1, 1), (0, 0, 0)))

    if cfg.CONFIG.MODEL.PRETRAINED:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('i3d_nl10_resnet50_v1_kinetics400',
                                                        tag=cfg.CONFIG.MODEL.PRETRAINED)))
    return model


def i3d_nl5_resnet101_v1_kinetics400(cfg):
    model = I3D_ResNetV1(num_classes=cfg.CONFIG.DATA.NUM_CLASSES,
                         depth=101,
                         pretrained=cfg.CONFIG.MODEL.PRETRAINED,
                         pretrained_base=cfg.CONFIG.MODEL.PRETRAINED_BASE,
                         feat_ext=cfg.CONFIG.INFERENCE.FEAT,
                         num_segment=cfg.CONFIG.DATA.NUM_SEGMENT,
                         num_crop=cfg.CONFIG.DATA.NUM_CROP,
                         out_indices=[3],
                         inflate_freq=((1, 1, 1),
                                       (1, 0, 1, 0),
                                       (1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1),
                                       (0, 1, 0)),
                         bn_eval=cfg.CONFIG.MODEL.BN_EVAL,
                         partial_bn=cfg.CONFIG.MODEL.PARTIAL_BN,
                         bn_frozen=cfg.CONFIG.MODEL.BN_FROZEN,
                         nonlocal_stages=(1, 2),
                         nonlocal_cfg=dict(nonlocal_type="gaussian"),
                         nonlocal_freq=((0, 0, 0),
                                        (0, 1, 0, 1),
                                        (0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0),
                                        (0, 0, 0)))

    if cfg.CONFIG.MODEL.PRETRAINED:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('i3d_nl5_resnet101_v1_kinetics400',
                                                        tag=cfg.CONFIG.MODEL.PRETRAINED)))
    return model


def i3d_nl10_resnet101_v1_kinetics400(cfg):
    model = I3D_ResNetV1(num_classes=cfg.CONFIG.DATA.NUM_CLASSES,
                         depth=101,
                         pretrained=cfg.CONFIG.MODEL.PRETRAINED,
                         pretrained_base=cfg.CONFIG.MODEL.PRETRAINED_BASE,
                         feat_ext=cfg.CONFIG.INFERENCE.FEAT,
                         num_segment=cfg.CONFIG.DATA.NUM_SEGMENT,
                         num_crop=cfg.CONFIG.DATA.NUM_CROP,
                         out_indices=[3],
                         inflate_freq=((1, 1, 1),
                                       (1, 0, 1, 0),
                                       (1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1),
                                       (0, 1, 0)),
                         bn_eval=cfg.CONFIG.MODEL.BN_EVAL,
                         partial_bn=cfg.CONFIG.MODEL.PARTIAL_BN,
                         bn_frozen=cfg.CONFIG.MODEL.BN_FROZEN,
                         nonlocal_stages=(1, 2),
                         nonlocal_cfg=dict(nonlocal_type="gaussian"),
                         nonlocal_freq=((0, 0, 0),
                                        (1, 1, 1, 1),
                                        (0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1),
                                        (0, 0, 0)))
    if cfg.CONFIG.MODEL.PRETRAINED:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('i3d_nl10_resnet101_v1_kinetics400',
                                                        tag=cfg.CONFIG.MODEL.PRETRAINED)))
    return model


def i3d_resnet50_v1_sthsthv2(cfg):
    model = I3D_ResNetV1(num_classes=cfg.CONFIG.DATA.NUM_CLASSES,
                         depth=50,
                         pretrained=cfg.CONFIG.MODEL.PRETRAINED,
                         pretrained_base=cfg.CONFIG.MODEL.PRETRAINED_BASE,
                         feat_ext=cfg.CONFIG.INFERENCE.FEAT,
                         num_segment=cfg.CONFIG.DATA.NUM_SEGMENT,
                         num_crop=cfg.CONFIG.DATA.NUM_CROP,
                         out_indices=[3],
                         inflate_freq=((1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 1, 0)),
                         bn_eval=cfg.CONFIG.MODEL.BN_EVAL,
                         partial_bn=cfg.CONFIG.MODEL.PARTIAL_BN,
                         bn_frozen=cfg.CONFIG.MODEL.BN_FROZEN)

    if cfg.CONFIG.MODEL.PRETRAINED:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('i3d_resnet50_v1_sthsthv2',
                                                        tag=cfg.CONFIG.MODEL.PRETRAINED)))
    return model
