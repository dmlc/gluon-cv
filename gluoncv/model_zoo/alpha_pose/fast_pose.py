"""Fast pose network for alpha pose"""
import os
import mxnet as mx
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet import initializer

from .utils import _try_load_parameters, _load_from_pytorch

__all__ = ['get_alphapose', 'alpha_pose_resnet101_v1b_coco']

class PixelShuffle(HybridBlock):
    """PixelShuffle layer for re-org channel to spatial dimention.

    Parameters
    ----------
    upscale_factor : int
        Upscaling factor for input->output spatially.

    """
    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def hybrid_forward(self, F, x):
        f1, f2 = self.upscale_factor, self.upscale_factor
        # (N, f1*f2*C, H, W)
        x = F.reshape(x, (0, -4, -1, f1 * f2, 0, 0))  # (N, C, f1*f2, H, W)
        x = F.reshape(x, (0, 0, -4, f1, f2, 0, 0))    # (N, C, f1, f2, H, W)
        x = F.transpose(x, (0, 1, 4, 2, 5, 3))        # (N, C, H, f1, W, f2)
        x = F.reshape(x, (0, 0, -3, -3))              # (N, C, H*f1, W*f2)
        return x


class DUC(HybridBlock):
    def __init__(self, planes, upscale_factor=2, norm_layer=nn.BatchNorm, **kwargs):
        super(DUC, self).__init__()
        with self.name_scope():
            self.conv = nn.Conv2D(planes, kernel_size=3, padding=1, use_bias=False)
            self.bn = norm_layer(**kwargs)
            self.relu = nn.Activation('relu')
            self.pixel_shuffle = PixelShuffle(upscale_factor)

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)

        return x


class SELayer(HybridBlock):
    def __init__(self, channel, reduction=1):
        super(SELayer, self).__init__()
        with self.name_scope():
            self.fc = nn.HybridSequential()
            self.fc.add(nn.Dense(channel // reduction))
            self.fc.add(nn.Activation('relu'))
            self.fc.add(nn.Dense(channel, activation='sigmoid'))

    def hybrid_forward(self, F, x):
        y = F.contrib.AdaptiveAvgPooling2D(x, output_size=1)
        y = self.fc(y)

        return y.expand_dims(-1).expand_dims(-1).broadcast_like(x) * x

class Bottleneck(HybridBlock):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=False, norm_layer=nn.BatchNorm, **kwargs):
        super(Bottleneck, self).__init__()

        with self.name_scope():
            self.conv1 = nn.Conv2D(planes, kernel_size=1, use_bias=False)
            self.bn1 = norm_layer(**kwargs)
            self.conv2 = nn.Conv2D(planes, kernel_size=3, strides=stride, padding=1, use_bias=False)
            self.bn2 = norm_layer(**kwargs)
            self.conv3 = nn.Conv2D(planes * 4, kernel_size=1, use_bias=False)
            self.bn3 = norm_layer(**kwargs)

        if reduction:
            self.se = SELayer(planes * 4)

        self.reduc = reduction
        self.downsample = downsample
        self.stride = stride

    def hybrid_forward(self, F, x):
        residual = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))

        out = self.bn3(self.conv3(out))
        if self.reduc:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = residual + out
        out = F.relu(out)

        return out


class FastSEResNet(HybridBlock):
    """ FastSEResNet """
    try_load_parameters = _try_load_parameters

    def __init__(self, architecture, norm_layer=nn.BatchNorm, **kwargs):
        super(FastSEResNet, self).__init__()
        architecture = architecture.split('_')[0]
        assert architecture in ["resnet50", "resnet101"]
        self.inplanes = 64
        self.norm_layer = norm_layer
        self.layers = [3, 4, {"resnet50": 6, "resnet101": 23}[architecture], 3]
        self.block = Bottleneck

        self.conv1 = nn.Conv2D(64, kernel_size=7, strides=2, padding=3, use_bias=False)
        self.bn1 = self.norm_layer(**kwargs)
        self.relu = nn.Activation('relu')
        self.maxpool = nn.MaxPool2D(pool_size=3, strides=2, padding=1)

        self.layer1 = self.make_layer(self.block, 64, self.layers[0], **kwargs)
        self.layer2 = self.make_layer(
            self.block, 128, self.layers[1], stride=2, **kwargs)
        self.layer3 = self.make_layer(
            self.block, 256, self.layers[2], stride=2, **kwargs)

        self.layer4 = self.make_layer(
            self.block, 512, self.layers[3], stride=2, **kwargs)

    def hybrid_forward(self, F, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))  # 64 * h/4 * w/4
        x = self.layer1(x)  # 256 * h/4 * w/4
        x = self.layer2(x)  # 512 * h/8 * w/8
        x = self.layer3(x)  # 1024 * h/16 * w/16
        x = self.layer4(x)  # 2048 * h/32 * w/32
        return x

    def stages(self):
        return [self.layer1, self.layer2, self.layer3, self.layer4]

    def make_layer(self, block, planes, blocks, stride=1, **kwargs):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.HybridSequential()
            downsample.add(nn.Conv2D(planes * block.expansion, kernel_size=1, strides=stride, use_bias=False))
            downsample.add(self.norm_layer(**kwargs))

        layers = nn.HybridSequential()
        if downsample is not None:
            layers.add(block(self.inplanes, planes, stride, downsample, reduction=True, norm_layer=self.norm_layer, **kwargs))
        else:
            layers.add(block(self.inplanes, planes, stride, downsample, norm_layer=self.norm_layer, **kwargs))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.add(block(self.inplanes, planes, norm_layer=self.norm_layer, **kwargs))

        return layers

class AlphaPose(HybridBlock):
    def __init__(self, preact, num_class, deconv_dim=256, norm_layer=nn.BatchNorm, norm_kwargs=None, **kwargs):
        super(AlphaPose, self).__init__(**kwargs)
        self.preact = preact
        self.num_class = num_class

        self.shuffle1 = PixelShuffle(2)
        if norm_kwargs is None:
            norm_kwargs = {}
        self.duc1 = DUC(1024, upscale_factor=2, norm_layer=norm_layer, **norm_kwargs)
        self.duc2 = DUC(512, upscale_factor=2, norm_layer=norm_layer, **norm_kwargs)

        self.conv_out = nn.Conv2D(
            channels=num_class,
            kernel_size=3,
            strides=1,
            padding=1,
            weight_initializer=initializer.Normal(0.001),
            bias_initializer=initializer.Zero()
        )

    def hybrid_forward(self, F, x):
        x = self.preact(x)
        x = self.shuffle1(x)
        x = self.duc1(x)
        x = self.duc2(x)
        x = self.conv_out(x)
        return x


def get_alphapose(name, dataset, num_class, pretrained=False, pretrained_base=True, ctx=mx.cpu(),
                  norm_layer=nn.BatchNorm, norm_kwargs=None,
                  root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    r"""Utility function to return AlphaPose networks.

    Parameters
    ----------
    name : str
        Model name.
    dataset : str
        The name of dataset.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : mxnet.Context
        Context such as mx.cpu(), mx.gpu(0).
    root : str
        Model weights storing path.

    Returns
    -------
    mxnet.gluon.HybridBlock
        The AlphaPose network.

    """
    if norm_kwargs is None:
        norm_kwargs = {}
    preact = FastSEResNet(name, norm_layer=norm_layer, **norm_kwargs)
    if not pretrained and pretrained_base:
        from ..model_zoo import get_model
        base_network = get_model(name, pretrained=True, root=root, ctx=ctx)
        _try_load_parameters(self=base_network, model=base_network, ctx=ctx)
    net = AlphaPose(preact, num_class, **kwargs)
    if pretrained:
        from ..model_store import get_model_file
        full_name = '_'.join(('alpha_pose', name, dataset))
        net.load_parameters(get_model_file(full_name, tag=pretrained, root=root), ctx=ctx)
    net.collect_params().reset_ctx(ctx)
    return net

def alpha_pose_resnet101_v1b_coco(**kwargs):
    from ...data import COCOKeyPoints
    keypoints = COCOKeyPoints.KEYPOINTS
    return get_alphapose(name='resnet101_v1b', dataset='coco', num_class=len(keypoints), **kwargs)
