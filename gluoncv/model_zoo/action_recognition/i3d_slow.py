# pylint: disable=missing-function-docstring, line-too-long, unused-argument
"""
SlowFast Networks for Video Recognition
ICCV 2019, https://arxiv.org/abs/1812.03982
Code adapted from https://github.com/open-mmlab/mmaction and
https://github.com/decisionforce/TPN
"""
from mxnet import init
from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm

from .non_local import build_nonlocal_block


__all__ = ['ResNet_SlowFast', 'i3d_slow_resnet101_f16s4_kinetics700']


def conv3x3x3(in_planes, out_planes, spatial_stride=1, temporal_stride=1, dilation=1):
    "3x3x3 convolution with padding"
    return nn.Conv3D(in_channels=in_planes,
                     channels=out_planes,
                     kernel_size=3,
                     strides=(temporal_stride, spatial_stride, spatial_stride),
                     padding=dilation,
                     dilation=dilation,
                     use_bias=False)


def conv1x3x3(in_planes, out_planes, spatial_stride=1, temporal_stride=1, dilation=1):
    "1x3x3 convolution with padding"
    return nn.Conv3D(in_channels=in_planes,
                     channels=out_planes,
                     kernel_size=(1, 3, 3),
                     strides=(temporal_stride, spatial_stride, spatial_stride),
                     padding=(0, dilation, dilation),
                     dilation=dilation,
                     use_bias=False)


class Bottleneck(HybridBlock):
    """Bottleneck building block for ResNet50, ResNet101 and ResNet152.
    """
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
                 norm_layer=BatchNorm,
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
        self.layer_name = layer_name

        if if_inflate:
            if inflate_style == '3x1x1':
                self.conv1 = nn.Conv3D(in_channels=inplanes,
                                       channels=planes,
                                       kernel_size=(3, 1, 1),
                                       strides=(self.conv1_stride_t, self.conv1_stride, self.conv1_stride),
                                       padding=(1, 0, 0),
                                       use_bias=False)
                self.conv2 = nn.Conv3D(in_channels=planes,
                                       channels=planes,
                                       kernel_size=(1, 3, 3),
                                       strides=(self.conv2_stride_t, self.conv2_stride, self.conv2_stride),
                                       padding=(0, dilation, dilation),
                                       dilation=(1, dilation, dilation),
                                       use_bias=False)
            else:
                self.conv1 = nn.Conv3D(in_channels=inplanes,
                                       channels=planes,
                                       kernel_size=1,
                                       strides=(self.conv1_stride_t, self.conv1_stride, self.conv1_stride),
                                       use_bias=False)
                self.conv2 = nn.Conv3D(in_channels=planes,
                                       channels=planes,
                                       kernel_size=3,
                                       strides=(self.conv2_stride_t, self.conv2_stride, self.conv2_stride),
                                       padding=(1, dilation, dilation),
                                       dilation=(1, dilation, dilation),
                                       use_bias=False)
        else:
            self.conv1 = nn.Conv3D(in_channels=inplanes,
                                   channels=planes,
                                   kernel_size=1,
                                   strides=(1, self.conv1_stride, self.conv1_stride),
                                   use_bias=False)
            self.conv2 = nn.Conv3D(in_channels=planes,
                                   channels=planes,
                                   kernel_size=(1, 3, 3),
                                   strides=(1, self.conv2_stride, self.conv2_stride),
                                   padding=(0, dilation, dilation),
                                   dilation=(1, dilation, dilation),
                                   use_bias=False)

        self.bn1 = norm_layer(in_channels=planes, **({} if norm_kwargs is None else norm_kwargs))
        self.bn2 = norm_layer(in_channels=planes, **({} if norm_kwargs is None else norm_kwargs))
        self.conv3 = nn.Conv3D(in_channels=planes,
                               channels=planes * self.expansion,
                               kernel_size=1,
                               use_bias=False)
        self.bn3 = norm_layer(in_channels=planes * self.expansion, **({} if norm_kwargs is None else norm_kwargs))
        self.relu = nn.Activation('relu')

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

    def hybrid_forward(self, F, x):
        """Hybrid forward of a ResNet bottleneck block"""
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
        out = F.Activation(out + identity, act_type='relu')

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
                   norm_layer=BatchNorm,
                   norm_kwargs=None,
                   layer_name=''):
    inflate_freq = inflate_freq if not isinstance(inflate_freq, int) else (inflate_freq,) * blocks
    nonlocal_freq = nonlocal_freq if not isinstance(nonlocal_freq, int) else (nonlocal_freq,) * blocks
    assert len(inflate_freq) == blocks
    assert len(nonlocal_freq) == blocks

    downsample = None
    if spatial_stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.HybridSequential(prefix=layer_name+'downsample_')
        with downsample.name_scope():
            downsample.add(nn.Conv3D(in_channels=inplanes,
                                     channels=planes * block.expansion,
                                     kernel_size=1,
                                     strides=(temporal_stride, spatial_stride, spatial_stride),
                                     use_bias=False))
            downsample.add(norm_layer(in_channels=planes * block.expansion, **({} if norm_kwargs is None else norm_kwargs)))


    layers = nn.HybridSequential(prefix=layer_name)
    cnt = 0
    with layers.name_scope():
        layers.add(block(inplanes=inplanes,
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
            layers.add(block(inplanes=inplanes,
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
    return layers


class ResNet_SlowFast(HybridBlock):
    """ResNe(x)t_SlowFast backbone.
    Args:
        depth (int): Depth of resnet, from {50, 101, 152}.
        num_stages (int): Resnet stages, normally 4.
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
                 num_segments=1,
                 num_crop=1,
                 num_stages=4,
                 spatial_strides=(1, 2, 2, 2),
                 temporal_strides=(1, 1, 1, 1),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 conv1_kernel_t=1,
                 conv1_stride_t=1,
                 pool1_kernel_t=1,
                 pool1_stride_t=1,
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
                 dropout_ratio=0.5,
                 init_std=0.01,
                 norm_layer=BatchNorm,
                 norm_kwargs=None,
                 ctx=None,
                 **kwargs):
        super(ResNet_SlowFast, self).__init__()

        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))

        self.num_classes = num_classes
        self.depth = depth
        self.pretrained = pretrained
        self.pretrained_base = pretrained_base
        self.feat_ext = feat_ext
        self.num_segments = num_segments
        self.num_crop = num_crop
        self.num_stages = num_stages
        assert 1 <= num_stages <= 4
        self.spatial_strides = spatial_strides
        self.temporal_strides = temporal_strides
        self.dilations = dilations
        assert len(spatial_strides) == len(temporal_strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.frozen_stages = frozen_stages
        self.inflate_freqs = inflate_freq if not isinstance(inflate_freq, int) else (inflate_freq,) * num_stages
        self.inflate_style = inflate_style
        self.nonlocal_stages = nonlocal_stages
        self.nonlocal_freqs = nonlocal_freq if not isinstance(nonlocal_freq, int) else (nonlocal_freq,) * num_stages
        self.nonlocal_cfg = nonlocal_cfg
        self.bn_eval = bn_eval
        self.bn_frozen = bn_frozen
        self.partial_bn = partial_bn
        self.feat_ext = feat_ext

        self.dropout_ratio = dropout_ratio
        self.init_std = init_std

        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = 64

        self.first_stage = nn.HybridSequential(prefix='')
        self.first_stage.add(nn.Conv3D(in_channels=3, channels=64, kernel_size=(conv1_kernel_t, 7, 7),
                                       strides=(conv1_stride_t, 2, 2), padding=((conv1_kernel_t - 1)//2, 3, 3), use_bias=False))
        self.first_stage.add(norm_layer(in_channels=64, **({} if norm_kwargs is None else norm_kwargs)))
        self.first_stage.add(nn.Activation('relu'))
        self.first_stage.add(nn.MaxPool3D(pool_size=(pool1_kernel_t, 3, 3), strides=(pool1_stride_t, 2, 2), padding=(pool1_kernel_t//2, 1, 1)))

        self.res_layers = nn.HybridSequential(prefix='')
        for i, num_blocks in enumerate(self.stage_blocks):
            spatial_stride = spatial_strides[i]
            temporal_stride = temporal_strides[i]
            dilation = dilations[i]
            planes = 64 * 2**i
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
            self.res_layers.add(res_layer)

        self.feat_dim = self.block.expansion * 64 * 2 ** (len(self.stage_blocks) - 1)

        self.st_avg = nn.GlobalAvgPool3D()

        self.head = nn.HybridSequential(prefix='')
        self.head.add(nn.Dropout(rate=self.dropout_ratio))
        self.fc = nn.Dense(in_units=self.feat_dim, units=num_classes, weight_initializer=init.Normal(sigma=self.init_std))
        self.head.add(self.fc)

        self.init_weights(ctx)

    def init_weights(self, ctx):
        """Initial I3D_slow network."""
        self.first_stage.initialize(ctx=ctx)
        self.res_layers.initialize(ctx=ctx)
        self.head.initialize(ctx=ctx)

    def hybrid_forward(self, F, x):
        """Hybrid forward of I3D_slow network"""
        x = self.first_stage(x)
        outs = []
        for i, res_layer in enumerate(self.res_layers):
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        feat = outs[0]

        # spatial temporal average
        pooled_feat = self.st_avg(feat)
        x = F.squeeze(pooled_feat, axis=(2, 3, 4))

        # segmental consensus
        x = F.reshape(x, shape=(-1, self.num_segments * self.num_crop, self.feat_dim))
        x = F.mean(x, axis=1)

        if self.feat_ext:
            return x

        out = self.head(x)
        return out


def i3d_slow_resnet101_f16s4_kinetics700(nclass=700, pretrained=False, pretrained_base=True, ctx=cpu(),
                                         root='~/.mxnet/models', num_segments=1, num_crop=1,
                                         partial_bn=False, bn_frozen=False, feat_ext=False, **kwargs):
    model = ResNet_SlowFast(num_classes=nclass,
                            depth=101,
                            pretrained=pretrained,
                            pretrained_base=pretrained_base,
                            feat_ext=feat_ext,
                            num_segments=num_segments,
                            num_crop=num_crop,
                            out_indices=[3],
                            bn_eval=False,
                            partial_bn=partial_bn,
                            bn_frozen=bn_frozen,
                            ctx=ctx,
                            **kwargs)

    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('i3d_slow_resnet101_f16s4_kinetics700',
                                             tag=pretrained, root=root), ctx=ctx)
        from ...data import Kinetics700Attr
        attrib = Kinetics700Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)

    return model
