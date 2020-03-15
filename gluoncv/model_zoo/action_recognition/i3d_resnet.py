# pylint: disable=line-too-long,too-many-lines,missing-docstring,arguments-differ,unused-argument
# Code adapted from https://github.com/open-mmlab/mmaction.

__all__ = ['I3D_ResNetV1', 'i3d_resnet50_v1_kinetics400', 'i3d_resnet101_v1_kinetics400',
           'i3d_nl5_resnet50_v1_kinetics400', 'i3d_nl10_resnet50_v1_kinetics400',
           'i3d_nl5_resnet101_v1_kinetics400', 'i3d_nl10_resnet101_v1_kinetics400',
           'i3d_resnet50_v1_sthsthv2', 'i3d_resnet50_v1_hmdb51', 'i3d_resnet50_v1_ucf101',
           'i3d_resnet50_v1_custom']

from mxnet import nd
from mxnet import init
from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm
from ..resnetv1b import resnet50_v1b, resnet101_v1b
from .non_local import build_nonlocal_block

def conv3x3x3(in_planes, out_planes, spatial_stride=1, temporal_stride=1, dilation=1):
    """3x3x3 convolution with padding"""
    return nn.Conv3D(in_channels=in_planes,
                     channels=out_planes,
                     kernel_size=3,
                     strides=(temporal_stride, spatial_stride, spatial_stride),
                     dilation=dilation,
                     use_bias=False)


def conv1x3x3(in_planes, out_planes, spatial_stride=1, temporal_stride=1, dilation=1):
    """1x3x3 convolution with padding"""
    return nn.Conv3D(in_channels=in_planes,
                     channels=out_planes,
                     kernel_size=(1, 3, 3),
                     strides=(temporal_stride, spatial_stride, spatial_stride),
                     padding=(0, dilation, dilation),
                     dilation=dilation,
                     use_bias=False)

class BasicBlock(HybridBlock):
    """
    Basic building block for ResNet18 and ResNet34.
    Not supported for I3D at this moment.
    """
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
                 norm_layer=BatchNorm,
                 norm_kwargs=None,
                 layer_name='',
                 **kwargs):
        super(BasicBlock, self).__init__()

        self.basicblock = nn.HybridSequential(prefix=layer_name)
        with self.basicblock.name_scope():
            if if_inflate:
                self.conv1 = conv3x3x3(inplanes, planes, spatial_stride, temporal_stride, dilation)
            else:
                self.conv1 = conv1x3x3(inplanes, planes, spatial_stride, temporal_stride, dilation)
            self.bn1 = norm_layer(**({} if norm_kwargs is None else norm_kwargs))
            self.relu = nn.Activation('relu')
            if if_inflate:
                self.conv2 = conv3x3x3(planes, planes)
            else:
                self.conv2 = conv1x3x3(planes, planes)
            self.bn2 = norm_layer(**({} if norm_kwargs is None else norm_kwargs))

            self.basicblock.add(self.conv1)
            self.basicblock.add(self.bn1)
            self.basicblock.add(self.relu)
            self.basicblock.add(self.conv2)
            self.basicblock.add(self.bn2)

            self.downsample = downsample
            self.spatial_stride = spatial_stride
            self.temporal_stride = temporal_stride
            self.dilation = dilation

    def hybrid_forward(self, F, x):
        """Hybrid forward of a ResNet basic block"""
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = F.Activation(out + identity, act_type='relu')
        return out

class Bottleneck(HybridBlock):
    r"""
    Bottleneck building block for ResNet50, ResNet101 and ResNet152.

    Parameters
    ----------
    inplanes : int.
        Input channels of each block.
    planes : int.
        Output channels of each block.
    spatial_stride : int, default is 1.
        Stride in spatial dimension of convolutional layers in a block.
    temporal_stride : int, default is 1.
        Stride in temporal dimension of convolutional layers in a block.
    dilation : int, default is 1.
        Dilation of convolutional layers in a block.
    downsample : bool.
        Whether to contain a downsampling layer in the block.
    if_inflate : bool.
        Whether enable inflation of 3D convolutional layers in this block.
    inflate_style : str, default is '3x1x1'.
        How to inflate a 2D kernel, either '3x1x1' or '1x3x3'.
    if_nonlocal : bool.
        Whether to insert a non-local block after this ResNet block.
    nonlocal_cfg : dict.
        Additional `non-local` arguments, for example `nonlocal_type='gaussian'`.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    layer_name : str, default is ''.
        Give a name to current block.
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

        self.bottleneck = nn.HybridSequential(prefix=layer_name)
        with self.bottleneck.name_scope():
            self.conv1_stride = 1
            self.conv2_stride = spatial_stride
            self.conv1_stride_t = 1
            self.conv2_stride_t = temporal_stride

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

            self.bottleneck.add(self.conv1)
            self.bottleneck.add(self.bn1)
            self.bottleneck.add(self.relu)
            self.bottleneck.add(self.conv2)
            self.bottleneck.add(self.bn2)
            self.bottleneck.add(self.relu)
            self.bottleneck.add(self.conv3)
            self.bottleneck.add(self.bn3)

            self.spatial_tride = spatial_stride
            self.temporal_tride = temporal_stride
            self.dilation = dilation

            if if_nonlocal and nonlocal_cfg is not None:
                nonlocal_cfg_ = nonlocal_cfg.copy()
                nonlocal_cfg_['in_channels'] = planes * self.expansion
                self.nonlocal_block = build_nonlocal_block(nonlocal_cfg_)
                self.bottleneck.add(self.nonlocal_block)
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
    """Build each stage of a ResNet"""
    inflate_freq = inflate_freq if not isinstance(inflate_freq, int) else (inflate_freq, ) * blocks
    nonlocal_freq = nonlocal_freq if not isinstance(nonlocal_freq, int) else (nonlocal_freq, ) * blocks
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


class I3D_ResNetV1(HybridBlock):
    r"""ResNet_I3D backbone.
    Inflated 3D model (I3D) from
    `"Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset"
    <https://arxiv.org/abs/1705.07750>`_ paper.

    Parameters
    ----------
    nclass : int.
        Number of categories in the dataset.
    depth : int, default is 50.
        Depth of ResNet, from {18, 34, 50, 101, 152}.
    num_stages : int, default is 4.
        Number of stages in a ResNet.
    pretrained : bool or str.
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True.
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    feat_ext : bool.
        Whether to extract features before dense classification layer or
        do a complete forward pass.
    num_segments : int, default is 1.
        Number of segments used to evenly divide a video.
    num_crop : int, default is 1.
        Number of crops used during evaluation, choices are 1, 3 or 10.
    spatial_strides : tuple of int.
        Strides in the spatial dimension of the first block of each stage.
    temporal_strides : tuple of int.
        Strides in the temporal dimension of the first block of each stage.
    dilations : tuple of int.
        Dilation ratio of each stage.
    out_indices : tuple of int.
        Collect features from the selected stages of ResNet,
        usually used for feature extraction or auxililary loss.
    conv1_kernel_t : int, default is 5.
        The kernel size of first convolutional layer in a ResNet.
    conv1_stride_t : int, default is 2.
        The stride of first convolutional layer in a ResNet.
    pool1_kernel_t : int, default is 1.
        The kernel size of first pooling layer in a ResNet.
    pool1_stride_t : int, default is 2.
        The stride of first pooling layer in a ResNet.
    inflate_freq : tuple of int.
        Select which 2D convolutional layers to be inflated to 3D convolutional layers in each stage.
    inflate_stride : tuple of int.
        The stride for inflated layers in each stage.
    inflate_style : str, default is '3x1x1'.
        How to inflate a 2D kernel, either '3x1x1' or '1x3x3'.
    nonlocal_stages : tuple of int.
        Select which stage we need non-local blocks.
    nonlocal_freq : tuple of int.
        Select where to insert non-local blocks in each stage.
    nonlocal_cfg : dict.
        Additional `non-local` arguments, for example `nonlocal_type='gaussian'`.
    bn_eval : bool.
        Whether to set BN layers to eval mode, namely, freeze
        running stats (mean and var).
    bn_frozen : bool.
        Whether to freeze weight and bias of BN layers.
    partial_bn : bool, default False.
        Freeze all batch normalization layers during training except the first layer.
    frozen_stages : int.
        Stages to be frozen (all param fixed). -1 means not freezing any parameters.
    dropout_ratio : float, default is 0.5.
        The dropout rate of a dropout layer.
        The larger the value, the more strength to prevent overfitting.
    init_std : float, default is 0.001.
        Standard deviation value when initialize the dense layers.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    ctx : Context, default CPU.
        The context in which to load the pretrained weights.
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 nclass,
                 depth,
                 num_stages=4,
                 pretrained=False,
                 pretrained_base=True,
                 feat_ext=False,
                 num_segments=1,
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
                 nonlocal_stages=(-1, ),
                 nonlocal_freq=(0, 1, 1, 0),
                 nonlocal_cfg=None,
                 bn_eval=True,
                 bn_frozen=False,
                 partial_bn=False,
                 frozen_stages=-1,
                 dropout_ratio=0.5,
                 init_std=0.01,
                 norm_layer=BatchNorm,
                 norm_kwargs=None,
                 ctx=None,
                 **kwargs):
        super(I3D_ResNetV1, self).__init__()

        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))

        self.nclass = nclass
        self.depth = depth
        self.num_stages = num_stages
        self.pretrained = pretrained
        self.pretrained_base = pretrained_base
        self.feat_ext = feat_ext
        self.num_segments = num_segments
        self.num_crop = num_crop
        self.spatial_strides = spatial_strides
        self.temporal_strides = temporal_strides
        self.dilations = dilations
        assert len(spatial_strides) == len(temporal_strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.inflate_freqs = inflate_freq if not isinstance(inflate_freq, int) else (inflate_freq, ) * num_stages
        self.inflate_style = inflate_style
        self.nonlocal_stages = nonlocal_stages
        self.nonlocal_freqs = nonlocal_freq if not isinstance(nonlocal_freq, int) else (nonlocal_freq, ) * num_stages
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

        self.first_stage = nn.HybridSequential(prefix='')
        self.first_stage.add(nn.Conv3D(in_channels=3, channels=64, kernel_size=(conv1_kernel_t, 7, 7),
                                       strides=(conv1_stride_t, 2, 2), padding=((conv1_kernel_t - 1)//2, 3, 3), use_bias=False))
        self.first_stage.add(norm_layer(in_channels=64, **({} if norm_kwargs is None else norm_kwargs)))
        self.first_stage.add(nn.Activation('relu'))
        self.first_stage.add(nn.MaxPool3D(pool_size=(pool1_kernel_t, 3, 3), strides=(pool1_stride_t, 2, 2), padding=(pool1_kernel_t//2, 1, 1)))

        self.pool2 = nn.MaxPool3D(pool_size=(2, 1, 1), strides=(2, 1, 1), padding=(0, 0, 0))

        if self.partial_bn:
            if norm_kwargs is not None:
                norm_kwargs['use_global_stats'] = True
            else:
                norm_kwargs = {}
                norm_kwargs['use_global_stats'] = True

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

        self.feat_dim = self.block.expansion * 64 * 2**(len(self.stage_blocks) - 1)

        # We use ``GlobalAvgPool3D`` here for simplicity. Otherwise the input size must be fixed.
        # You can also use ``AvgPool3D`` and specify the arguments on your own, e.g.
        # self.st_avg = nn.AvgPool3D(pool_size=(4, 7, 7), strides=1, padding=0)
        # ``AvgPool3D`` is 10% faster, but ``GlobalAvgPool3D`` makes the code cleaner.
        self.st_avg = nn.GlobalAvgPool3D()

        self.head = nn.HybridSequential(prefix='')
        self.head.add(nn.Dropout(rate=self.dropout_ratio))
        self.fc = nn.Dense(in_units=self.feat_dim, units=nclass, weight_initializer=init.Normal(sigma=self.init_std))
        self.head.add(self.fc)

        self.init_weights(ctx)

    def init_weights(self, ctx):
        """Initial I3D network with its 2D pretrained weights."""

        self.first_stage.initialize(ctx=ctx)
        self.res_layers.initialize(ctx=ctx)
        self.head.initialize(ctx=ctx)

        if self.pretrained_base and not self.pretrained:
            if self.depth == 50:
                resnet2d = resnet50_v1b(pretrained=True)
            elif self.depth == 101:
                resnet2d = resnet101_v1b(pretrained=True)
            else:
                print('No such 2D pre-trained network of depth %d.' % (self.depth))

            weights2d = resnet2d.collect_params()
            if self.nonlocal_cfg is None:
                weights3d = self.collect_params()
            else:
                train_params_list = []
                raw_params = self.collect_params()
                for raw_name in raw_params.keys():
                    if 'nonlocal' in raw_name:
                        continue
                    train_params_list.append(raw_name)
                init_patterns = '|'.join(train_params_list)
                weights3d = self.collect_params(init_patterns)
            assert len(weights2d.keys()) == len(weights3d.keys()), 'Number of parameters should be same.'

            dict2d = {}
            for key_id, key_name in enumerate(weights2d.keys()):
                dict2d[key_id] = key_name

            dict3d = {}
            for key_id, key_name in enumerate(weights3d.keys()):
                dict3d[key_id] = key_name

            dict_transform = {}
            for key_id, key_name in dict3d.items():
                dict_transform[dict2d[key_id]] = key_name

            cnt = 0
            for key2d, key3d in dict_transform.items():
                if 'conv' in key3d:
                    temporal_dim = weights3d[key3d].shape[2]
                    temporal_2d = nd.expand_dims(weights2d[key2d].data(), axis=2)
                    inflated_2d = nd.broadcast_to(temporal_2d, shape=[0, 0, temporal_dim, 0, 0]) / temporal_dim
                    assert inflated_2d.shape == weights3d[key3d].shape, 'the shape of %s and %s does not match. ' % (key2d, key3d)
                    weights3d[key3d].set_data(inflated_2d)
                    cnt += 1
                    print('%s is done with shape: ' % (key3d), weights3d[key3d].shape)
                if 'batchnorm' in key3d:
                    assert weights2d[key2d].shape == weights3d[key3d].shape, 'the shape of %s and %s does not match. ' % (key2d, key3d)
                    weights3d[key3d].set_data(weights2d[key2d].data())
                    cnt += 1
                    print('%s is done with shape: ' % (key3d), weights3d[key3d].shape)
                if 'dense' in key3d:
                    cnt += 1
                    print('%s is skipped with shape: ' % (key3d), weights3d[key3d].shape)

            assert cnt == len(weights2d.keys()), 'Not all parameters have been ported, check the initialization.'

    def hybrid_forward(self, F, x):
        """Hybrid forward of I3D network"""
        x = self.first_stage(x)
        outs = []
        for i, res_layer in enumerate(self.res_layers):
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
            if i == 0:
                x = self.pool2(x)

        feat = outs[0]

        # spatial temporal average
        pooled_feat = self.st_avg(feat)
        x = F.squeeze(pooled_feat, axis=(2, 3, 4))

        # segmental consensus
        x = F.reshape(x, shape=(-1, self.num_segments * self.num_crop, self.feat_dim))
        x = F.mean(x, axis=1)

        if self.feat_ext:
            return x

        x = self.head(x)
        return x

def i3d_resnet50_v1_kinetics400(nclass=400, pretrained=False, pretrained_base=True, ctx=cpu(),
                                root='~/.mxnet/models', use_tsn=False, num_segments=1, num_crop=1,
                                partial_bn=False, bn_frozen=False, feat_ext=False, **kwargs):
    r"""Inflated 3D model (I3D) with ResNet50 backbone trained on Kinetics400 dataset.

    Parameters
    ----------
    nclass : int.
        Number of categories in the dataset.
    pretrained : bool or str.
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True.
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    ctx : Context, default CPU.
        The context in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    num_segments : int, default is 1.
        Number of segments used to evenly divide a video.
    num_crop : int, default is 1.
        Number of crops used during evaluation, choices are 1, 3 or 10.
    partial_bn : bool, default False.
        Freeze all batch normalization layers during training except the first layer.
    bn_frozen : bool.
        Whether to freeze weight and bias of BN layers.
    feat_ext : bool.
        Whether to extract features before dense classification layer or
        do a complete forward pass.
    """

    model = I3D_ResNetV1(nclass=nclass,
                         depth=50,
                         pretrained=pretrained,
                         pretrained_base=pretrained_base,
                         feat_ext=feat_ext,
                         num_segments=num_segments,
                         num_crop=num_crop,
                         out_indices=[3],
                         inflate_freq=((1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 1, 0)),
                         bn_eval=False,
                         partial_bn=partial_bn,
                         bn_frozen=bn_frozen,
                         ctx=ctx,
                         **kwargs)

    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('i3d_resnet50_v1_kinetics400',
                                             tag=pretrained, root=root), ctx=ctx)
        from ...data import Kinetics400Attr
        attrib = Kinetics400Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)

    return model

def i3d_resnet101_v1_kinetics400(nclass=400, pretrained=False, pretrained_base=True, ctx=cpu(),
                                 root='~/.mxnet/models', use_tsn=False, num_segments=1, num_crop=1,
                                 partial_bn=False, feat_ext=False, **kwargs):
    r"""Inflated 3D model (I3D) with ResNet101 backbone trained on Kinetics400 dataset.

    Parameters
    ----------
    nclass : int.
        Number of categories in the dataset.
    pretrained : bool or str.
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True.
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    ctx : Context, default CPU.
        The context in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    num_segments : int, default is 1.
        Number of segments used to evenly divide a video.
    num_crop : int, default is 1.
        Number of crops used during evaluation, choices are 1, 3 or 10.
    partial_bn : bool, default False.
        Freeze all batch normalization layers during training except the first layer.
    bn_frozen : bool.
        Whether to freeze weight and bias of BN layers.
    feat_ext : bool.
        Whether to extract features before dense classification layer or
        do a complete forward pass.
    """

    model = I3D_ResNetV1(nclass=nclass,
                         depth=101,
                         pretrained=pretrained,
                         pretrained_base=pretrained_base,
                         feat_ext=feat_ext,
                         num_segments=num_segments,
                         num_crop=num_crop,
                         out_indices=[3],
                         inflate_freq=((1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1), (0, 1, 0)),
                         bn_eval=False,
                         partial_bn=partial_bn,
                         ctx=ctx,
                         **kwargs)

    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('i3d_resnet101_v1_kinetics400',
                                             tag=pretrained, root=root), ctx=ctx)
        from ...data import Kinetics400Attr
        attrib = Kinetics400Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)

    return model

def i3d_nl5_resnet50_v1_kinetics400(nclass=400, pretrained=False, pretrained_base=True, ctx=cpu(),
                                    root='~/.mxnet/models', use_tsn=False, num_segments=1, num_crop=1,
                                    partial_bn=False, feat_ext=False, **kwargs):
    r"""Inflated 3D model (I3D) with ResNet50 backbone and 5 non-local blocks
    trained on Kinetics400 dataset.

    Parameters
    ----------
    nclass : int.
        Number of categories in the dataset.
    pretrained : bool or str.
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True.
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    ctx : Context, default CPU.
        The context in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    num_segments : int, default is 1.
        Number of segments used to evenly divide a video.
    num_crop : int, default is 1.
        Number of crops used during evaluation, choices are 1, 3 or 10.
    partial_bn : bool, default False.
        Freeze all batch normalization layers during training except the first layer.
    bn_frozen : bool.
        Whether to freeze weight and bias of BN layers.
    feat_ext : bool.
        Whether to extract features before dense classification layer or
        do a complete forward pass.
    """

    model = I3D_ResNetV1(nclass=nclass,
                         depth=50,
                         pretrained=pretrained,
                         pretrained_base=pretrained_base,
                         feat_ext=feat_ext,
                         num_segments=num_segments,
                         num_crop=num_crop,
                         out_indices=[3],
                         inflate_freq=((1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 1, 0)),
                         nonlocal_stages=(1, 2),
                         nonlocal_cfg=dict(nonlocal_type="gaussian"),
                         nonlocal_freq=((0, 0, 0), (0, 1, 0, 1), (0, 1, 0, 1, 0, 1), (0, 0, 0)),
                         bn_eval=False,
                         partial_bn=partial_bn,
                         ctx=ctx,
                         **kwargs)

    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('i3d_nl5_resnet50_v1_kinetics400',
                                             tag=pretrained, root=root), ctx=ctx)
        from ...data import Kinetics400Attr
        attrib = Kinetics400Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)

    return model

def i3d_nl10_resnet50_v1_kinetics400(nclass=400, pretrained=False, pretrained_base=True, ctx=cpu(),
                                     root='~/.mxnet/models', use_tsn=False, num_segments=1, num_crop=1,
                                     partial_bn=False, feat_ext=False, **kwargs):
    r"""Inflated 3D model (I3D) with ResNet50 backbone and 10 non-local blocks
    trained on Kinetics400 dataset.

    Parameters
    ----------
    nclass : int.
        Number of categories in the dataset.
    pretrained : bool or str.
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True.
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    ctx : Context, default CPU.
        The context in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    num_segments : int, default is 1.
        Number of segments used to evenly divide a video.
    num_crop : int, default is 1.
        Number of crops used during evaluation, choices are 1, 3 or 10.
    partial_bn : bool, default False.
        Freeze all batch normalization layers during training except the first layer.
    bn_frozen : bool.
        Whether to freeze weight and bias of BN layers.
    feat_ext : bool.
        Whether to extract features before dense classification layer or
        do a complete forward pass.
    """

    model = I3D_ResNetV1(nclass=nclass,
                         depth=50,
                         pretrained=pretrained,
                         pretrained_base=pretrained_base,
                         feat_ext=feat_ext,
                         num_segments=num_segments,
                         num_crop=num_crop,
                         out_indices=[3],
                         inflate_freq=((1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 1, 0)),
                         nonlocal_stages=(1, 2),
                         nonlocal_cfg=dict(nonlocal_type="gaussian"),
                         nonlocal_freq=((0, 0, 0), (1, 1, 1, 1), (1, 1, 1, 1, 1, 1), (0, 0, 0)),
                         bn_eval=False,
                         partial_bn=partial_bn,
                         ctx=ctx,
                         **kwargs)

    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('i3d_nl10_resnet50_v1_kinetics400',
                                             tag=pretrained, root=root), ctx=ctx)
        from ...data import Kinetics400Attr
        attrib = Kinetics400Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)

    return model

def i3d_nl5_resnet101_v1_kinetics400(nclass=400, pretrained=False, pretrained_base=True, ctx=cpu(),
                                     root='~/.mxnet/models', use_tsn=False, num_segments=1, num_crop=1,
                                     partial_bn=False, feat_ext=False, **kwargs):
    r"""Inflated 3D model (I3D) with ResNet101 backbone and 5 non-local blocks
    trained on Kinetics400 dataset.

    Parameters
    ----------
    nclass : int.
        Number of categories in the dataset.
    pretrained : bool or str.
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True.
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    ctx : Context, default CPU.
        The context in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    num_segments : int, default is 1.
        Number of segments used to evenly divide a video.
    num_crop : int, default is 1.
        Number of crops used during evaluation, choices are 1, 3 or 10.
    partial_bn : bool, default False.
        Freeze all batch normalization layers during training except the first layer.
    bn_frozen : bool.
        Whether to freeze weight and bias of BN layers.
    feat_ext : bool.
        Whether to extract features before dense classification layer or
        do a complete forward pass.
    """

    model = I3D_ResNetV1(nclass=nclass,
                         depth=101,
                         pretrained=pretrained,
                         pretrained_base=pretrained_base,
                         feat_ext=feat_ext,
                         num_segments=num_segments,
                         num_crop=num_crop,
                         out_indices=[3],
                         inflate_freq=((1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1), (0, 1, 0)),
                         nonlocal_stages=(1, 2),
                         nonlocal_cfg=dict(nonlocal_type="gaussian"),
                         nonlocal_freq=((0, 0, 0), (0, 1, 0, 1), (0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0), (0, 0, 0)),
                         bn_eval=False,
                         partial_bn=partial_bn,
                         ctx=ctx,
                         **kwargs)

    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('i3d_nl5_resnet101_v1_kinetics400',
                                             tag=pretrained, root=root), ctx=ctx)
        from ...data import Kinetics400Attr
        attrib = Kinetics400Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)

    return model

def i3d_nl10_resnet101_v1_kinetics400(nclass=400, pretrained=False, pretrained_base=True, ctx=cpu(),
                                      root='~/.mxnet/models', use_tsn=False, num_segments=1, num_crop=1,
                                      partial_bn=False, feat_ext=False, **kwargs):
    r"""Inflated 3D model (I3D) with ResNet101 backbone and 10 non-local blocks
    trained on Kinetics400 dataset.

    Parameters
    ----------
    nclass : int.
        Number of categories in the dataset.
    pretrained : bool or str.
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True.
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    ctx : Context, default CPU.
        The context in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    num_segments : int, default is 1.
        Number of segments used to evenly divide a video.
    num_crop : int, default is 1.
        Number of crops used during evaluation, choices are 1, 3 or 10.
    partial_bn : bool, default False.
        Freeze all batch normalization layers during training except the first layer.
    bn_frozen : bool.
        Whether to freeze weight and bias of BN layers.
    feat_ext : bool.
        Whether to extract features before dense classification layer or
        do a complete forward pass.
    """

    model = I3D_ResNetV1(nclass=nclass,
                         depth=101,
                         pretrained=pretrained,
                         pretrained_base=pretrained_base,
                         feat_ext=feat_ext,
                         num_segments=num_segments,
                         num_crop=num_crop,
                         out_indices=[3],
                         inflate_freq=((1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1), (0, 1, 0)),
                         nonlocal_stages=(1, 2),
                         nonlocal_cfg=dict(nonlocal_type="gaussian"),
                         nonlocal_freq=((0, 0, 0), (1, 1, 1, 1), (0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1), (0, 0, 0)),
                         bn_eval=False,
                         partial_bn=partial_bn,
                         ctx=ctx,
                         **kwargs)

    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('i3d_nl10_resnet101_v1_kinetics400',
                                             tag=pretrained, root=root), ctx=ctx)
        from ...data import Kinetics400Attr
        attrib = Kinetics400Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)

    return model

def i3d_resnet50_v1_sthsthv2(nclass=174, pretrained=False, pretrained_base=True, ctx=cpu(),
                             root='~/.mxnet/models', use_tsn=False, num_segments=1, num_crop=1,
                             partial_bn=False, feat_ext=False, **kwargs):
    r"""Inflated 3D model (I3D) with ResNet50 backbone trained on Something-Something-V2 dataset.

    Parameters
    ----------
    nclass : int.
        Number of categories in the dataset.
    pretrained : bool or str.
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True.
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    ctx : Context, default CPU.
        The context in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    num_segments : int, default is 1.
        Number of segments used to evenly divide a video.
    num_crop : int, default is 1.
        Number of crops used during evaluation, choices are 1, 3 or 10.
    partial_bn : bool, default False.
        Freeze all batch normalization layers during training except the first layer.
    bn_frozen : bool.
        Whether to freeze weight and bias of BN layers.
    feat_ext : bool.
        Whether to extract features before dense classification layer or
        do a complete forward pass.
    """

    model = I3D_ResNetV1(nclass=nclass,
                         depth=50,
                         pretrained=pretrained,
                         pretrained_base=pretrained_base,
                         feat_ext=feat_ext,
                         num_segments=num_segments,
                         num_crop=num_crop,
                         out_indices=[3],
                         inflate_freq=((1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 1, 0)),
                         bn_eval=False,
                         partial_bn=partial_bn,
                         ctx=ctx,
                         **kwargs)

    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('i3d_resnet50_v1_sthsthv2',
                                             tag=pretrained, root=root), ctx=ctx)
        from ...data import SomethingSomethingV2Attr
        attrib = SomethingSomethingV2Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)

    return model

def i3d_resnet50_v1_hmdb51(nclass=51, pretrained=False, pretrained_base=True, ctx=cpu(),
                           root='~/.mxnet/models', use_tsn=False, num_segments=1, num_crop=1,
                           partial_bn=False, use_kinetics_pretrain=True, feat_ext=False, **kwargs):
    r"""Inflated 3D model (I3D) with ResNet50 backbone trained on HMDB51 dataset.

    Parameters
    ----------
    nclass : int.
        Number of categories in the dataset.
    pretrained : bool or str.
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True.
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    ctx : Context, default CPU.
        The context in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    num_segments : int, default is 1.
        Number of segments used to evenly divide a video.
    num_crop : int, default is 1.
        Number of crops used during evaluation, choices are 1, 3 or 10.
    partial_bn : bool, default False.
        Freeze all batch normalization layers during training except the first layer.
    bn_frozen : bool.
        Whether to freeze weight and bias of BN layers.
    feat_ext : bool.
        Whether to extract features before dense classification layer or
        do a complete forward pass.
    """

    model = I3D_ResNetV1(nclass=nclass,
                         depth=50,
                         pretrained=pretrained,
                         pretrained_base=pretrained_base,
                         feat_ext=feat_ext,
                         num_segments=num_segments,
                         num_crop=num_crop,
                         out_indices=[3],
                         inflate_freq=((1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 1, 0)),
                         bn_eval=False,
                         partial_bn=False,
                         ctx=ctx,
                         dropout_ratio=0.8,
                         init_std=0.001,
                         **kwargs)

    if use_kinetics_pretrain and not pretrained:
        from gluoncv.model_zoo import get_model
        kinetics_model = get_model('i3d_resnet50_v1_kinetics400', nclass=400, pretrained=True)
        source_params = kinetics_model.collect_params()
        target_params = model.collect_params()
        assert len(source_params.keys()) == len(target_params.keys())

        pretrained_weights = []
        for layer_name in source_params.keys():
            pretrained_weights.append(source_params[layer_name].data())

        for i, layer_name in enumerate(target_params.keys()):
            if i + 2 == len(source_params.keys()):
                # skip the last dense layer
                break
            target_params[layer_name].set_data(pretrained_weights[i])

    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('i3d_resnet50_v1_hmdb51',
                                             tag=pretrained, root=root), ctx=ctx)
        from ...data import HMDB51Attr
        attrib = HMDB51Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)

    return model

def i3d_resnet50_v1_ucf101(nclass=101, pretrained=False, pretrained_base=True, ctx=cpu(),
                           root='~/.mxnet/models', use_tsn=False, num_segments=1, num_crop=1,
                           partial_bn=False, use_kinetics_pretrain=True, feat_ext=False, **kwargs):
    r"""Inflated 3D model (I3D) with ResNet50 backbone trained on UCF101 dataset.

    Parameters
    ----------
    nclass : int.
        Number of categories in the dataset.
    pretrained : bool or str.
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True.
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    ctx : Context, default CPU.
        The context in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    num_segments : int, default is 1.
        Number of segments used to evenly divide a video.
    num_crop : int, default is 1.
        Number of crops used during evaluation, choices are 1, 3 or 10.
    partial_bn : bool, default False.
        Freeze all batch normalization layers during training except the first layer.
    bn_frozen : bool.
        Whether to freeze weight and bias of BN layers.
    feat_ext : bool.
        Whether to extract features before dense classification layer or
        do a complete forward pass.
    """

    model = I3D_ResNetV1(nclass=nclass,
                         depth=50,
                         pretrained=pretrained,
                         pretrained_base=pretrained_base,
                         feat_ext=feat_ext,
                         num_segments=num_segments,
                         num_crop=num_crop,
                         out_indices=[3],
                         inflate_freq=((1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 1, 0)),
                         bn_eval=False,
                         partial_bn=True,
                         ctx=ctx,
                         dropout_ratio=0.8,
                         init_std=0.001,
                         **kwargs)

    if use_kinetics_pretrain and not pretrained:
        from gluoncv.model_zoo import get_model
        kinetics_model = get_model('i3d_resnet50_v1_kinetics400', nclass=400, pretrained=True)
        source_params = kinetics_model.collect_params()
        target_params = model.collect_params()
        assert len(source_params.keys()) == len(target_params.keys())

        pretrained_weights = []
        for layer_name in source_params.keys():
            pretrained_weights.append(source_params[layer_name].data())

        for i, layer_name in enumerate(target_params.keys()):
            if i + 2 == len(source_params.keys()):
                # skip the last dense layer
                break
            target_params[layer_name].set_data(pretrained_weights[i])

    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('i3d_resnet50_v1_ucf101',
                                             tag=pretrained, root=root), ctx=ctx)
        from ...data import UCF101Attr
        attrib = UCF101Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)

    return model

def i3d_resnet50_v1_custom(nclass=400, pretrained=False, pretrained_base=True, ctx=cpu(),
                           root='~/.mxnet/models', use_tsn=False, num_segments=1, num_crop=1,
                           partial_bn=False, use_kinetics_pretrain=True, feat_ext=False, **kwargs):
    r"""Inflated 3D model (I3D) with ResNet50 backbone. Customized for users's own dataset.

    Parameters
    ----------
    nclass : int.
        Number of categories in the dataset.
    pretrained : bool or str.
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True.
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    ctx : Context, default CPU.
        The context in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    num_segments : int, default is 1.
        Number of segments used to evenly divide a video.
    num_crop : int, default is 1.
        Number of crops used during evaluation, choices are 1, 3 or 10.
    partial_bn : bool, default False.
        Freeze all batch normalization layers during training except the first layer.
    bn_frozen : bool.
        Whether to freeze weight and bias of BN layers.
    feat_ext : bool.
        Whether to extract features before dense classification layer or
        do a complete forward pass.
    use_kinetics_pretrain : bool.
        Whether to load Kinetics-400 pre-trained model weights.
    """

    model = I3D_ResNetV1(nclass=nclass,
                         depth=50,
                         pretrained=pretrained,
                         pretrained_base=pretrained_base,
                         feat_ext=feat_ext,
                         num_segments=num_segments,
                         num_crop=num_crop,
                         out_indices=[3],
                         inflate_freq=((1, 1, 1), (1, 0, 1, 0), (1, 0, 1, 0, 1, 0), (0, 1, 0)),
                         bn_eval=False,
                         partial_bn=True,
                         ctx=ctx,
                         dropout_ratio=0.8,
                         init_std=0.001,
                         **kwargs)

    if use_kinetics_pretrain and not pretrained:
        from gluoncv.model_zoo import get_model
        kinetics_model = get_model('i3d_resnet50_v1_kinetics400', nclass=400, pretrained=True)
        source_params = kinetics_model.collect_params()
        target_params = model.collect_params()
        assert len(source_params.keys()) == len(target_params.keys())

        pretrained_weights = []
        for layer_name in source_params.keys():
            pretrained_weights.append(source_params[layer_name].data())

        for i, layer_name in enumerate(target_params.keys()):
            if i + 2 == len(source_params.keys()):
                # skip the last dense layer
                break
            target_params[layer_name].set_data(pretrained_weights[i])

    model.collect_params().reset_ctx(ctx)

    return model
