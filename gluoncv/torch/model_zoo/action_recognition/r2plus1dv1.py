# pylint: disable=missing-function-docstring, missing-class-docstring, unused-argument
"""R2Plus1D, https://arxiv.org/abs/1711.11248. Code adapted from
https://github.com/pytorch/vision/blob/master/torchvision/models/video/resnet.py."""
import torch
import torch.nn as nn
from torch.nn import BatchNorm3d

__all__ = ['R2Plus1D', 'r2plus1d_v1_resnet18_kinetics400', 'r2plus1d_v1_resnet34_kinetics400',
           'r2plus1d_v1_resnet50_kinetics400', 'r2plus1d_v1_resnet101_kinetics400',
           'r2plus1d_v1_resnet152_kinetics400', 'r2plus1d_v1_resnet50_custom']


def conv3x1x1(in_planes, out_planes, spatial_stride=1, temporal_stride=1, dilation=1):
    """3x1x1 convolution with padding"""
    return nn.Conv3d(in_channels=in_planes,
                     out_channels=out_planes,
                     kernel_size=(3, 1, 1),
                     stride=(temporal_stride, spatial_stride, spatial_stride),
                     padding=(dilation, 0, 0),
                     dilation=dilation,
                     bias=False)


class Conv2Plus1D(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 midplanes,
                 stride=1,
                 padding=1,
                 norm_layer=BatchNorm3d,
                 norm_kwargs=None,
                 **kwargs):
        super(Conv2Plus1D, self).__init__()

        self.conv1 = nn.Conv3d(in_channels=inplanes,
                               out_channels=midplanes,
                               kernel_size=(1, 3, 3),
                               stride=(1, stride, stride),
                               padding=(0, padding, padding),
                               bias=False)
        self.bn1 = norm_layer(num_features=midplanes,
                              **({} if norm_kwargs is None else norm_kwargs))
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv3d(in_channels=midplanes,
                               out_channels=planes,
                               kernel_size=(3, 1, 1),
                               stride=(stride, 1, 1),
                               padding=(padding, 0, 0),
                               bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 norm_layer=BatchNorm3d, norm_kwargs=None, layer_name='',
                 **kwargs):
        super(BasicBlock, self).__init__()
        self.downsample = downsample

        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)
        self.conv1 = Conv2Plus1D(inplanes, planes, midplanes, stride)
        self.bn1 = norm_layer(num_features=planes,
                              **({} if norm_kwargs is None else norm_kwargs))
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = Conv2Plus1D(planes, planes, midplanes)
        self.bn2 = norm_layer(num_features=planes,
                              **({} if norm_kwargs is None else norm_kwargs))

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

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 norm_layer=BatchNorm3d, norm_kwargs=None, layer_name='',
                 **kwargs):
        super(Bottleneck, self).__init__()
        self.downsample = downsample

        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        # 1x1x1
        self.conv1 = nn.Conv3d(in_channels=inplanes, out_channels=planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(num_features=planes,
                              **({} if norm_kwargs is None else norm_kwargs))
        self.relu = nn.ReLU(inplace=False)

        # Second kernel
        self.conv2 = Conv2Plus1D(planes, planes, midplanes, stride)
        self.bn2 = norm_layer(num_features=planes,
                              **({} if norm_kwargs is None else norm_kwargs))

        self.conv3 = nn.Conv3d(in_channels=planes, out_channels=planes * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = norm_layer(num_features=planes * self.expansion,
                              **({} if norm_kwargs is None else norm_kwargs))

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
        return out


class R2Plus1D(nn.Module):
    r"""The R2+1D network.
    A Closer Look at Spatiotemporal Convolutions for Action Recognition.
    CVPR, 2018. https://arxiv.org/abs/1711.11248
    """
    def __init__(self, num_classes, block, layers, dropout_ratio=0.5,
                 num_segment=1, num_crop=1, feat_ext=False,
                 init_std=0.001, partial_bn=False,
                 norm_layer=BatchNorm3d, norm_kwargs=None, **kwargs):
        super(R2Plus1D, self).__init__()
        self.partial_bn = partial_bn
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.num_segment = num_segment
        self.num_crop = num_crop
        self.feat_ext = feat_ext
        self.inplanes = 64
        self.feat_dim = 512 * block.expansion

        self.conv1 = nn.Conv3d(in_channels=3, out_channels=45, kernel_size=(1, 7, 7),
                               stride=(1, 2, 2), padding=(0, 3, 3), bias=False)
        self.bn1 = norm_layer(num_features=45, **({} if norm_kwargs is None else norm_kwargs))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x1x1(in_planes=45, out_planes=64)
        self.bn2 = norm_layer(num_features=64, **({} if norm_kwargs is None else norm_kwargs))

        if self.partial_bn:
            if norm_kwargs is not None:
                norm_kwargs['use_global_stats'] = True
            else:
                norm_kwargs = {}
                norm_kwargs['use_global_stats'] = True

        self.layer1 = self._make_res_layer(block=block,
                                           planes=64,
                                           blocks=layers[0],
                                           layer_name='layer1_')
        self.layer2 = self._make_res_layer(block=block,
                                           planes=128,
                                           blocks=layers[1],
                                           stride=2,
                                           layer_name='layer2_')
        self.layer3 = self._make_res_layer(block=block,
                                           planes=256,
                                           blocks=layers[2],
                                           stride=2,
                                           layer_name='layer3_')
        self.layer4 = self._make_res_layer(block=block,
                                           planes=512,
                                           blocks=layers[3],
                                           stride=2,
                                           layer_name='layer4_')

        self.avgpool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.dropout = nn.Dropout(self.dropout_ratio)
        self.fc = nn.Linear(in_features=self.feat_dim, out_features=num_classes)
        nn.init.normal_(self.fc.weight, 0, self.init_std)
        nn.init.constant_(self.fc.bias, 0)

    def _make_res_layer(self,
                        block,
                        planes,
                        blocks,
                        stride=1,
                        norm_layer=BatchNorm3d,
                        norm_kwargs=None,
                        layer_name=''):
        """Build each stage of a ResNet"""
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(in_channels=self.inplanes,
                          out_channels=planes * block.expansion,
                          kernel_size=1,
                          stride=(stride, stride, stride),
                          bias=False),
                norm_layer(num_features=planes * block.expansion,
                           **({} if norm_kwargs is None else norm_kwargs)))

        layers = []
        layers.append(block(inplanes=self.inplanes,
                            planes=planes,
                            stride=stride,
                            downsample=downsample))

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes=self.inplanes, planes=planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        bs, _, _, _, _ = x.size()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(bs, -1)

        if self.feat_ext:
            return x

        x = self.fc(self.dropout(x))
        return x


def r2plus1d_v1_resnet18_kinetics400(cfg):
    model = R2Plus1D(num_classes=cfg.CONFIG.DATA.NUM_CLASSES,
                     block=BasicBlock,
                     layers=[2, 2, 2, 2],
                     num_segment=cfg.CONFIG.DATA.NUM_SEGMENT,
                     num_crop=cfg.CONFIG.DATA.NUM_CROP,
                     feat_ext=cfg.CONFIG.INFERENCE.FEAT,
                     partial_bn=cfg.CONFIG.MODEL.PARTIAL_BN)
    if cfg.CONFIG.MODEL.PRETRAINED:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('r2plus1d_v1_resnet18_kinetics400',
                                                        tag=cfg.CONFIG.MODEL.PRETRAINED)))
    return model


def r2plus1d_v1_resnet34_kinetics400(cfg):
    model = R2Plus1D(num_classes=cfg.CONFIG.DATA.NUM_CLASSES,
                     block=BasicBlock,
                     layers=[3, 4, 6, 3],
                     num_segment=cfg.CONFIG.DATA.NUM_SEGMENT,
                     num_crop=cfg.CONFIG.DATA.NUM_CROP,
                     feat_ext=cfg.CONFIG.INFERENCE.FEAT,
                     partial_bn=cfg.CONFIG.MODEL.PARTIAL_BN)

    if cfg.CONFIG.MODEL.PRETRAINED:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('r2plus1d_v1_resnet34_kinetics400',
                                                        tag=cfg.CONFIG.MODEL.PRETRAINED)))
    return model


def r2plus1d_v1_resnet50_kinetics400(cfg):
    model = R2Plus1D(num_classes=cfg.CONFIG.DATA.NUM_CLASSES,
                     block=Bottleneck,
                     layers=[3, 4, 6, 3],
                     num_segment=cfg.CONFIG.DATA.NUM_SEGMENT,
                     num_crop=cfg.CONFIG.DATA.NUM_CROP,
                     feat_ext=cfg.CONFIG.INFERENCE.FEAT,
                     partial_bn=cfg.CONFIG.MODEL.PARTIAL_BN)

    if cfg.CONFIG.MODEL.PRETRAINED:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('r2plus1d_v1_resnet50_kinetics400',
                                                        tag=cfg.CONFIG.MODEL.PRETRAINED)))
    return model


def r2plus1d_v1_resnet101_kinetics400(cfg):
    model = R2Plus1D(num_classes=cfg.CONFIG.DATA.NUM_CLASSES,
                     block=Bottleneck,
                     layers=[3, 4, 23, 3],
                     num_segment=cfg.CONFIG.DATA.NUM_SEGMENT,
                     num_crop=cfg.CONFIG.DATA.NUM_CROP,
                     feat_ext=cfg.CONFIG.INFERENCE.FEAT,
                     partial_bn=cfg.CONFIG.MODEL.PARTIAL_BN)

    if cfg.CONFIG.MODEL.PRETRAINED:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('r2plus1d_v1_resnet101_kinetics400',
                                                        tag=cfg.CONFIG.MODEL.PRETRAINED)))
    return model

def r2plus1d_v1_resnet152_kinetics400(cfg):
    model = R2Plus1D(num_classes=cfg.CONFIG.DATA.NUM_CLASSES,
                     block=Bottleneck,
                     layers=[3, 8, 36, 3],
                     num_segment=cfg.CONFIG.DATA.NUM_SEGMENT,
                     num_crop=cfg.CONFIG.DATA.NUM_CROP,
                     feat_ext=cfg.CONFIG.INFERENCE.FEAT,
                     partial_bn=cfg.CONFIG.MODEL.PARTIAL_BN)

    if cfg.CONFIG.MODEL.PRETRAINED:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('r2plus1d_v1_resnet152_kinetics400',
                                                        tag=cfg.CONFIG.MODEL.PRETRAINED)))
    return model


def r2plus1d_v1_resnet50_custom(cfg):
    model = R2Plus1D(num_classes=cfg.CONFIG.DATA.NUM_CLASSES,
                     block=Bottleneck,
                     layers=[3, 4, 6, 3],
                     num_segment=cfg.CONFIG.DATA.NUM_SEGMENT,
                     num_crop=cfg.CONFIG.DATA.NUM_CROP,
                     feat_ext=cfg.CONFIG.INFERENCE.FEAT,
                     partial_bn=cfg.CONFIG.MODEL.PARTIAL_BN)

    if cfg.CONFIG.MODEL.PRETRAINED:
        from ..model_store import get_model_file
        state_dict = torch.load(get_model_file('r2plus1d_v1_resnet50_kinetics400', tag=cfg.CONFIG.MODEL.PRETRAINED))
        for k in list(state_dict.keys()):
            # retain only backbone up to before the classification layer
            if k.startswith('fc'):
                del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {'fc.weight', 'fc.bias'}
        print("=> initialized from a R2+1D model pretrained on Kinetcis400 dataset")
    return model
