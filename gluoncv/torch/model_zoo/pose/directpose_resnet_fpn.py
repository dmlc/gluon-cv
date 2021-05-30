"""Directpose based on resnet + FPN backbones"""
# pylint: disable=line-too-long, redefined-builtin, missing-class-docstring, unused-variable,unnecessary-pass,arguments-differ
# adpated from https://github.com/aim-uofa/AdelaiDet/blob/master/adet/modeling/backbone
import math
from abc import ABCMeta, abstractmethod

import numpy as np
import torch
import torch.nn.parallel
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

from ...data.structures import ImageList
from ...engine.config import get_cfg_defaults
from ...nn.batch_norm import get_norm
from ...nn.batch_norm import NaiveSyncBatchNorm
from ...nn.shape_spec import ShapeSpec
from ..object_detection.model_utils import detector_postprocess
from .directpose import DirectPose


__all__ = ['directpose_resnet_lpf_fpn', 'directpose_resnet50_lpf_fpn_coco']


def conv3x3(in_planes, out_planes, stride=1, groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:
        Args:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function
        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)

        self.norm = norm
        self.activation = activation

    def forward(self, x):
        if x.numel() == 0 and self.training:
            # https://github.com/pytorch/pytorch/issues/12013
            assert not isinstance(
                self.norm, torch.nn.SyncBatchNorm
            ), "SyncBatchNorm does not support empty inputs!"

        x = super().forward(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, norm_layer=None, filter_size=1):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1:
            raise ValueError('BasicBlock only supports groups=1')
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        if(stride == 1):
            self.conv2 = conv3x3(planes, planes)
        else:
            self.conv2 = nn.Sequential(Downsample(filt_size=filter_size, stride=stride, channels=planes),
                                       conv3x3(planes, planes),)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, norm_layer=None, filter_size=1):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, groups)  # stride moved
        self.bn2 = norm_layer(planes)
        if(stride == 1):
            self.conv3 = conv1x1(planes, planes * self.expansion)
        else:
            self.conv3 = nn.Sequential(Downsample(filt_size=filter_size, stride=stride, channels=planes),
                                       conv1x1(planes, planes * self.expansion))
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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

        out += identity
        out = self.relu(out)

        return out


class Backbone(nn.Module, metaclass=ABCMeta):
    """
    Abstract base class for network backbones.
    """

    def __init__(self):
        """
        The `__init__` method of any subclass can specify its own set of arguments.
        """
        super().__init__()

    @abstractmethod
    def forward(self):
        """
        Subclasses must override this method, but adhere to the same return type.
        Returns:
            dict[str->Tensor]: mapping from feature name (e.g., "res2") to tensor
        """
        pass

    @property
    def size_divisibility(self):
        """
        Some backbones require the input height and width to be divisible by a
        specific integer. This is typically true for encoder / decoder type networks
        with lateral connection (e.g., FPN) for which feature maps need to match
        dimension in the "bottom up" and "top down" paths. Set to 0 if no specific
        input size divisibility is required.
        """
        return 0

    def output_shape(self):
        """
        Returns:
            dict[str->ShapeSpec]
        """
        # this is a backward-compatible default
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }


class ResNetLPF(Backbone):
    def __init__(self, cfg, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, norm_layer=None, filter_size=1,
                 pool_only=True, return_idx=(0, 1, 2, 3)):
        super().__init__()
        self.return_idx = list(return_idx)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        planes = [int(width_per_group * groups * 2 ** i) for i in range(4)]
        self.inplanes = planes[0]

        if(pool_only):
            self.conv1 = nn.Conv2d(3, planes[0], kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, planes[0], kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = norm_layer(planes[0])
        self.relu = nn.ReLU(inplace=True)

        if(pool_only):
            self.maxpool = nn.Sequential(*[nn.MaxPool2d(kernel_size=2, stride=1),
                                           Downsample(filt_size=filter_size, stride=2, channels=planes[0])])
        else:
            self.maxpool = nn.Sequential(*[Downsample(filt_size=filter_size, stride=2, channels=planes[0]),
                                           nn.MaxPool2d(kernel_size=2, stride=1),
                                           Downsample(filt_size=filter_size, stride=2, channels=planes[0])])

        self.layer1 = self._make_layer(
            block, planes[0], layers[0], groups=groups, norm_layer=norm_layer)
        self.layer2 = self._make_layer(
            block, planes[1], layers[1], stride=2, groups=groups, norm_layer=norm_layer, filter_size=filter_size)
        self.layer3 = self._make_layer(
            block, planes[2], layers[2], stride=2, groups=groups, norm_layer=norm_layer, filter_size=filter_size)
        self.layer4 = self._make_layer(
            block, planes[3], layers[3], stride=2, groups=groups, norm_layer=norm_layer, filter_size=filter_size)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if(m.in_channels != m.out_channels or m.out_channels != m.groups or m.bias is not None):
                    # don't want to reinitialize downsample layers, code assuming normal conv layers will not have these characteristics
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                else:
                    print('Not initializing')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
        self._freeze_backbone(cfg.CONFIG.MODEL.BACKBONE.FREEZE_AT)
        # self._freeze_bn()

    def _freeze_backbone(self, freeze_at):
        if freeze_at < 0:
            return
        for stage_index in range(freeze_at):
            if stage_index == 0:
                # stage 0 is the stem
                for p in self.conv1.parameters():
                    p.requires_grad = False
                for p in self.bn1.parameters():
                    p.requires_grad = False
            else:
                m = getattr(self, "layer" + str(stage_index))
                for p in m.parameters():
                    p.requires_grad = False

    def _freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def _make_layer(self, block, planes, blocks, stride=1, groups=1, norm_layer=None, filter_size=1):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            # downsample = nn.Sequential(
            #     conv1x1(self.inplanes, planes * block.expansion, stride, filter_size=filter_size),
            #     norm_layer(planes * block.expansion),
            # )

            downsample = [Downsample(filt_size=filter_size, stride=stride,
                                     channels=self.inplanes), ] if(stride != 1) else []
            downsample += [conv1x1(self.inplanes, planes * block.expansion, 1),
                           norm_layer(planes * block.expansion)]
            # print(downsample)
            downsample = nn.Sequential(*downsample)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            groups, norm_layer, filter_size=filter_size))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=groups,
                                norm_layer=norm_layer, filter_size=filter_size))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        outs = []
        outs.append(self.layer1(x))  # 1/4
        outs.append(self.layer2(outs[-1]))  # 1/8
        outs.append(self.layer3(outs[-1]))  # 1/16
        outs.append(self.layer4(outs[-1]))  # 1/32
        return {"res{}".format(idx + 2): outs[idx] for idx in self.return_idx}


class Downsample(nn.Module):
    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels

        # print('Filter size [%i]'%filt_size)
        if(self.filt_size == 1):
            a = np.array([1.,])
        elif(self.filt_size == 2):
            a = np.array([1., 1.])
        elif(self.filt_size == 3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size == 4):
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size == 5):
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size == 6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size == 7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:, None] * a[None, :])
        filt = filt / torch.sum(filt)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])

def get_pad_layer(pad_type):
    if(pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type == 'zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer


class Downsample1D(nn.Module):
    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0):
        super(Downsample1D, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        # print('Filter size [%i]' % filt_size)
        if(self.filt_size == 1):
            a = np.array([1., ])
        elif(self.filt_size == 2):
            a = np.array([1., 1.])
        elif(self.filt_size == 3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size == 4):
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size == 5):
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size == 6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size == 7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a)
        filt = filt / torch.sum(filt)
        self.register_buffer('filt', filt[None, None, :].repeat((self.channels, 1, 1)))

        self.pad = get_pad_layer_1d(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride]
        else:
            return F.conv1d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


def get_pad_layer_1d(pad_type):
    if(pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad1d
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad1d
    elif(pad_type == 'zero'):
        PadLayer = nn.ZeroPad1d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer


class FPN(Backbone):
    """
    This module implements :paper:`FPN`.
    It creates pyramid features built on top of some input feature maps.
    """

    def __init__(self, bottom_up, in_features, out_channels, norm="", top_block=None, fuse_type="sum"):
        """
        Args:
            bottom_up (Backbone): module representing the bottom up subnetwork.
                Must be a subclass of :class:`Backbone`. The multi-scale feature
                maps generated by the bottom up network, and listed in `in_features`,
                are used to generate FPN levels.
            in_features (list[str]): names of the input feature maps coming
                from the backbone to which FPN is attached. For example, if the
                backbone produces ["res2", "res3", "res4"], any *contiguous* sublist
                of these may be used; order must be from high to low resolution.
            out_channels (int): number of channels in the output feature maps.
            norm (str): the normalization to use.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra FPN levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            fuse_type (str): types for fusing the top down features and the lateral
                ones. It can be "sum" (default), which sums up element-wise; or "avg",
                which takes the element-wise mean of the two.
        """
        super(FPN, self).__init__()
        assert isinstance(bottom_up, Backbone)

        # Feature map strides and channels from the bottom up network (e.g. ResNet)
        input_shapes = bottom_up.output_shape()
        in_strides = [input_shapes[f].stride for f in in_features]
        in_channels = [input_shapes[f].channels for f in in_features]

        _assert_strides_are_log2_contiguous(in_strides)
        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(in_channels):
            lateral_norm = get_norm(norm, out_channels)
            output_norm = get_norm(norm, out_channels)

            lateral_conv = Conv2d(
                in_channels, out_channels, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            output_conv = Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
            )
            c2_xavier_fill(lateral_conv)
            c2_xavier_fill(output_conv)
            stage = int(math.log2(in_strides[idx]))
            self.add_module("fpn_lateral{}".format(stage), lateral_conv)
            self.add_module("fpn_output{}".format(stage), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
        self.top_block = top_block
        self.in_features = in_features
        self.bottom_up = bottom_up
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in in_strides}
        # top block output feature maps.
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = in_strides[-1]
        assert fuse_type in {"avg", "sum"}
        self._fuse_type = fuse_type

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, x):
        """
        Args:
            input (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.
        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        # Reverse feature maps into top-down order (from low to high resolution)
        # print(x.shape, x[:, :, 100:200, 100:200])
        # raise
        bottom_up_features = self.bottom_up(x)
        x = [bottom_up_features[f] for f in self.in_features[::-1]]
        results = []
        prev_features = self.lateral_convs[0](x[0])
        results.append(self.output_convs[0](prev_features))
        for features, lateral_conv, output_conv in zip(x[1:], self.lateral_convs[1:], self.output_convs[1:]):
            lateral_features = lateral_conv(features)
            top_down_features = F.interpolate(prev_features, size=lateral_features.shape[-2:], mode="nearest")
            prev_features = lateral_features + top_down_features
            if self._fuse_type == "avg":
                prev_features /= 2
            results.insert(0, output_conv(prev_features))

        if self.top_block is not None:
            top_block_in_feature = bottom_up_features.get(self.top_block.in_feature, None)
            if top_block_in_feature is None:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        return dict(zip(self._out_features, results))

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

def c2_xavier_fill(module: nn.Module) -> None:
    """
    Initialize `module.weight` using the "XavierFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.
    Args:
        module (torch.nn.Module): module to initialize.
    """
    # Caffe2 implementation of XavierFill in fact
    # corresponds to kaiming_uniform_ in PyTorch
    nn.init.kaiming_uniform_(module.weight, a=1)  # pyre-ignore
    if module.bias is not None:  # pyre-ignore
        nn.init.constant_(module.bias, 0)

def _assert_strides_are_log2_contiguous(strides):
    """
    Assert that each stride is 2x times its preceding stride, i.e. "contiguous in log2".
    """
    for i, stride in enumerate(strides[1:], 1):
        assert stride == 2 * strides[i - 1], "Strides {} {} are not log2 contiguous".format(
            stride, strides[i - 1]
        )

class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet and FCOS to generate extra layers, P6 and P7 from
    C5 or P5 feature.
    """

    def __init__(self, in_channels, out_channels, in_features="res5"):
        super().__init__()
        self.num_levels = 2
        self.in_feature = in_features
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for module in [self.p6, self.p7]:
            c2_xavier_fill(module)

    def forward(self, x):
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]


class LastLevelP6(nn.Module):
    """
    This module is used in FCOS to generate extra layers
    """

    def __init__(self, in_channels, out_channels, in_features="res5"):
        super().__init__()
        self.num_levels = 1
        self.in_feature = in_features
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        for module in [self.p6]:
            c2_xavier_fill(module)

    def forward(self, x):
        p6 = self.p6(x)
        return [p6]


class ResNetFPNDirectPose(nn.Module):
    def __init__(self, backbone, pose, pixel_mean=(0.485, 0.456, 0.406),
                 pixel_std=(0.229, 0.224, 0.225)):
        super(ResNetFPNDirectPose, self).__init__()
        self.backbone = backbone
        self.proposal_generator = pose
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs_or_tensor):
        if isinstance(batched_inputs_or_tensor, torch.Tensor):
            images = images_tensor = batched_inputs_or_tensor
            is_tensor = True
        else:
            # preprocess
            images = self.preprocess_image(batched_inputs_or_tensor)
            images_tensor = images.tensor
            is_tensor = False

        features = self.backbone(images_tensor)

        if not is_tensor and "instances" in batched_inputs_or_tensor[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs_or_tensor]
        elif not is_tensor and "targets" in batched_inputs_or_tensor[0]:
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs_or_tensor]
        else:
            gt_instances = None
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        if self.training:
            return proposal_losses

        if is_tensor:
            # raw tensor input, e.g., during tracing
            return proposals
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(proposals, batched_inputs_or_tensor, images.image_sizes):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results


    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x / 255 - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


def directpose_resnet_lpf_fpn(cfg):
    """
    Create a ResNetFPNDirectPose instance from config.
    Returns:
        ResNetFPNDirectPose: a :class:`ResNetFPNDirectPose` instance.
    """
    depth = cfg.CONFIG.MODEL.RESNETS.DEPTH
    out_features = cfg.CONFIG.MODEL.RESNETS.OUT_FEATURES

    num_blocks_per_stage = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}[depth]
    out_stage_idx = [{"res2": 0, "res3": 1, "res4": 2, "res5": 3}[f] for f in out_features]
    out_feature_channels = {"res2": 256, "res3": 512,
                            "res4": 1024, "res5": 2048}
    out_feature_strides = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}
    model = ResNetLPF(cfg, Bottleneck, num_blocks_per_stage, norm_layer=NaiveSyncBatchNorm,
                      filter_size=3, pool_only=True, return_idx=out_stage_idx)
    model._out_features = out_features
    model._out_feature_channels = out_feature_channels
    model._out_feature_strides = out_feature_strides

    if cfg.CONFIG.MODEL.BACKBONE.LOAD_URL:
        model.load_state_dict(
            model_zoo.load_url(cfg.CONFIG.MODEL.BACKBONE.LOAD_URL, map_location='cpu', check_hash=True)['state_dict'],
            strict=False)

    if cfg.CONFIG.MODEL.BACKBONE.ANTI_ALIAS:
        bottom_up = model
    else:
        raise NotImplementedError
    in_features = cfg.CONFIG.MODEL.FPN.IN_FEATURES
    out_channels = cfg.CONFIG.MODEL.FPN.OUT_CHANNELS
    top_levels = cfg.CONFIG.MODEL.DIRECTPOSE.TOP_LEVELS
    in_channels_top = out_channels
    if top_levels == 2:
        top_block = LastLevelP6P7(in_channels_top, out_channels, "p5")
    if top_levels == 1:
        top_block = LastLevelP6(in_channels_top, out_channels, "p5")
    elif top_levels == 0:
        top_block = None
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.CONFIG.MODEL.FPN.NORM,
        top_block=top_block,
        fuse_type=cfg.CONFIG.MODEL.FPN.FUSE_TYPE,
    )

    pose_net = DirectPose(cfg, backbone.output_shape())

    return ResNetFPNDirectPose(backbone, pose_net)

def directpose_resnet50_lpf_fpn_coco(cfg):
    base_cfg = get_cfg_defaults("directpose")
    base_cfg.merge_from_other_cfg(cfg)
    cfg = base_cfg
    depth = 50
    out_features = ["res3", "res4", "res5"]

    num_blocks_per_stage = {50: [3, 4, 6, 3], 101: [3, 4, 23, 3], 152: [3, 8, 36, 3]}[depth]
    out_stage_idx = [{"res2": 0, "res3": 1, "res4": 2, "res5": 3}[f] for f in out_features]
    out_feature_channels = {"res2": 256, "res3": 512,
                            "res4": 1024, "res5": 2048}
    out_feature_strides = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}
    model = ResNetLPF(cfg, Bottleneck, num_blocks_per_stage, norm_layer=NaiveSyncBatchNorm,
                      filter_size=3, pool_only=True, return_idx=out_stage_idx)
    model._out_features = out_features
    model._out_feature_channels = out_feature_channels
    model._out_feature_strides = out_feature_strides

    if cfg.CONFIG.MODEL.BACKBONE.LOAD_URL and not cfg.CONFIG.MODEL.PRETRAINED:
        model.load_state_dict(
            model_zoo.load_url(cfg.CONFIG.MODEL.BACKBONE.LOAD_URL, map_location='cpu', check_hash=True)['state_dict'],
            strict=False)

    bottom_up = model
    in_features = ["res3", "res4", "res5"]
    out_channels = 256
    in_channels_top = out_channels
    top_block = LastLevelP6P7(in_channels_top, out_channels, "p5")
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm="",
        top_block=top_block,
        fuse_type="sum",
    )

    cfg.CONFIG.MODEL.DIRECTPOSE.TOP_LEVELS = 2
    cfg.CONFIG.MODEL.DIRECTPOSE.SIZES_OF_INTEREST = [64, 128, 256, 512]
    cfg.CONFIG.MODEL.DIRECTPOSE.FPN_STRIDES = [8, 16, 32, 64, 128]
    cfg.CONFIG.MODEL.DIRECTPOSE.IN_FEATURES = ['p3', 'p4', 'p5', 'p6', 'p7']
    cfg.CONFIG.MODEL.DIRECTPOSE.HM_OFFSET = False
    cfg.CONFIG.MODEL.DIRECTPOSE.HM_TYPE = 'BinaryLabels'
    cfg.CONFIG.MODEL.DIRECTPOSE.REFINE_KPT = False
    cfg.CONFIG.MODEL.DIRECTPOSE.SAMPLE_FEATURE = 'lower'
    cfg.CONFIG.MODEL.DIRECTPOSE.KPALIGN_GROUPS = 9
    cfg.CONFIG.MODEL.DIRECTPOSE.SEPERATE_CONV_FEATURE = True
    cfg.CONFIG.MODEL.DIRECTPOSE.LOSS_ON_LOCATOR = False
    cfg.CONFIG.MODEL.DIRECTPOSE.KPT_VIS = True
    cfg.CONFIG.MODEL.DIRECTPOSE.CLOSEKPT_NMS = False
    cfg.CONFIG.MODEL.DIRECTPOSE.CENTER_BRANCH = 'kpt'
    cfg.CONFIG.MODEL.DIRECTPOSE.USE_SCALE = False

    pose_net = DirectPose(cfg, backbone.output_shape())

    model = ResNetFPNDirectPose(backbone, pose_net)

    if cfg.CONFIG.MODEL.PRETRAINED:
        from ..model_store import get_model_file
        state_dict = torch.load(get_model_file('directpose_resnet50_lpf_fpn_coco', tag=cfg.CONFIG.MODEL.PRETRAINED),
                                map_location=torch.device('cpu'))
        msg = model.load_state_dict(state_dict, strict=True)
        print("=> Initialized from a DirectPose pretrained on COCO dataset")
    return model
