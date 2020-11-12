"""
Video Classification with Channel-Separated Convolutional Networks
ICCV 2019, https://arxiv.org/abs/1904.02811
Large-scale weakly-supervised pre-training for video action recognition
CVPR 2019, https://arxiv.org/abs/1905.00561
"""
# pylint: disable=missing-function-docstring, missing-class-docstring
import torch
import torch.nn as nn


__all__ = ['ResNet_IRCSNv2', 'ircsn_v2_resnet152_f32s2_kinetics400']


eps = 1e-3
bn_mmt = 0.1


class Affine(nn.Module):
    def __init__(self, feature_in):
        super(Affine, self).__init__()
        self.weight = nn.Parameter(torch.randn(feature_in, 1, 1, 1))
        self.bias = nn.Parameter(torch.randn(feature_in, 1, 1, 1))
        self.weight.requires_grad = False
        self.bias.requires_grad = False

    def forward(self, x):
        x = x * self.weight + self.bias
        return x


class Bottleneck_IRCSNv2(nn.Module):
    def __init__(self, in_planes, planes, stride=1, temporal_stride=1,
                 down_sample=None, expansion=2, temporal_kernel=3, use_affine=True):

        super(Bottleneck_IRCSNv2, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=(1, 1, 1), bias=False, stride=(1, 1, 1))

        if use_affine:
            self.bn1 = Affine(planes)
        else:
            self.bn1 = nn.BatchNorm3d(planes, track_running_stats=True, eps=eps, momentum=bn_mmt)

        self.conv3 = nn.Conv3d(planes, planes, kernel_size=(3, 3, 3), bias=False,
                               stride=(temporal_stride, stride, stride),
                               padding=((temporal_kernel - 1) // 2, 1, 1),
                               groups=planes)

        if use_affine:
            self.bn3 = Affine(planes)
        else:
            self.bn3 = nn.BatchNorm3d(planes, track_running_stats=True, eps=eps, momentum=bn_mmt)

        self.conv4 = nn.Conv3d(
            planes, planes * self.expansion, kernel_size=1, bias=False)

        if use_affine:
            self.bn4 = Affine(planes * self.expansion)
        else:
            self.bn4 = nn.BatchNorm3d(planes * self.expansion, track_running_stats=True, eps=eps, momentum=bn_mmt)

        self.relu = nn.ReLU(inplace=True)
        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        out = self.conv4(out)
        out = self.bn4(out)

        if self.down_sample is not None:
            residual = self.down_sample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_IRCSNv2(nn.Module):
    def __init__(self,
                 block,
                 block_nums,
                 num_classes=400,
                 use_affine=True):

        self.use_affine = use_affine
        self.in_planes = 64
        self.num_classes = num_classes

        super(ResNet_IRCSNv2, self).__init__()

        self.conv1 = nn.Conv3d(
            3,
            64,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(1, 3, 3),
            bias=False)
        if use_affine:
            self.bn1 = Affine(64)
        else:
            self.bn1 = nn.BatchNorm3d(64, track_running_stats=True, eps=eps, momentum=bn_mmt)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.layer1 = self._make_layer(block, in_planes=64, planes=64, blocks=block_nums[0],
                                       stride=1, expansion=4)

        self.layer2 = self._make_layer(block, in_planes=256, planes=128, blocks=block_nums[1],
                                       stride=2, temporal_stride=2, expansion=4)

        self.layer3 = self._make_layer(block, in_planes=512, planes=256, blocks=block_nums[2],
                                       stride=2, temporal_stride=2, expansion=4)

        self.layer4 = self._make_layer(block, in_planes=1024, planes=512, blocks=block_nums[3],
                                       stride=2, temporal_stride=2, expansion=4)

        self.avgpool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))

        self.out_fc = nn.Linear(in_features=2048, out_features=num_classes)

    def _make_layer(self,
                    block,
                    in_planes,
                    planes,
                    blocks,
                    stride=1,
                    temporal_stride=1,
                    expansion=4):

        if self.use_affine:
            down_bn = Affine(planes * expansion)
        else:
            down_bn = nn.BatchNorm3d(planes * expansion, track_running_stats=True, eps=eps, momentum=bn_mmt)
        down_sample = nn.Sequential(
            nn.Conv3d(
                in_planes,
                planes * expansion,
                kernel_size=1,
                stride=(temporal_stride, stride, stride),
                bias=False), down_bn)
        layers = []
        layers.append(
            block(in_planes, planes, stride, temporal_stride, down_sample, expansion,
                  temporal_kernel=3, use_affine=self.use_affine))
        for _ in range(1, blocks):
            layers.append(block(planes * expansion, planes, expansion=expansion,
                                temporal_kernel=3, use_affine=self.use_affine))

        return nn.Sequential(*layers)

    def forward(self, x):

        bs, _, _, _, _ = x.size()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(bs, -1)
        logits = self.out_fc(x)

        return logits


def ircsn_v2_resnet152_f32s2_kinetics400(cfg):
    model = ResNet_IRCSNv2(Bottleneck_IRCSNv2,
                           num_classes=cfg.CONFIG.DATA.NUM_CLASSES,
                           block_nums=[3, 8, 36, 3],
                           use_affine=cfg.CONFIG.MODEL.USE_AFFINE)

    if cfg.CONFIG.MODEL.PRETRAINED:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('ircsn_v2_resnet152_f32s2_kinetics400',
                                                        tag=cfg.CONFIG.MODEL.PRETRAINED)))
    return model
