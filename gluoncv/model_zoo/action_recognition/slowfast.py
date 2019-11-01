# pylint: disable=line-too-long,too-many-lines,missing-docstring,arguments-differ,unused-argument
# Code partially borrowed from https://github.com/r1ch88/SlowFastNetworks.

__all__ = ['SlowFast', 'slowfast_4x16_resnet50_kinetics400']

from mxnet import init
from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm

class Bottleneck(HybridBlock):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 strides=1,
                 downsample=None,
                 head_conv=1,
                 norm_layer=BatchNorm,
                 norm_kwargs=None,
                 layer_name=''):
        super(Bottleneck, self).__init__()

        bottleneck = nn.HybridSequential(prefix=layer_name)
        with bottleneck.name_scope():
            if head_conv == 1:
                self.conv1 = nn.Conv3D(in_channels=inplanes, channels=planes, kernel_size=1, use_bias=False)
                self.bn1 = norm_layer(in_channels=planes, **({} if norm_kwargs is None else norm_kwargs))
            elif head_conv == 3:
                self.conv1 = nn.Conv3D(in_channels=inplanes, channels=planes, kernel_size=(3, 1, 1), padding=(1, 0, 0), use_bias=False)
                self.bn1 = norm_layer(in_channels=planes, **({} if norm_kwargs is None else norm_kwargs))
            else:
                raise ValueError("Unsupported head_conv!")
            self.conv2 = nn.Conv3D(in_channels=planes, channels=planes, kernel_size=(1, 3, 3), strides=(1, strides, strides), padding=(0, 1, 1), use_bias=False)
            self.bn2 = norm_layer(in_channels=planes, **({} if norm_kwargs is None else norm_kwargs))
            self.conv3 = nn.Conv3D(in_channels=planes, channels=planes * 4, kernel_size=1, use_bias=False)
            self.bn3 = norm_layer(in_channels=planes * self.expansion, gamma_initializer='zeros', **({} if norm_kwargs is None else norm_kwargs))
            self.relu = nn.Activation('relu')
            self.downsample = downsample

    def hybrid_forward(self, F, x):
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
        return out


class SlowFast(HybridBlock):
    """SlowFast networks (SlowFast) from
    `"SlowFast Networks for Video Recognition"
    <https://arxiv.org/abs/1812.03982>`_ paper.
    """
    def __init__(self,
                 nclass,
                 block=Bottleneck,
                 layers=None,
                 pretrained=False,
                 pretrained_base=True,
                 num_segments=1,
                 num_crop=1,
                 bn_eval=True,
                 bn_frozen=False,
                 partial_bn=False,
                 frozen_stages=-1,
                 dropout_ratio=0.5,
                 init_std=0.01,
                 slow_temporal_stride=16,
                 fast_temporal_stride=2,
                 slow_frames=4,
                 fast_frames=32,
                 norm_layer=BatchNorm,
                 norm_kwargs=None,
                 ctx=None,
                 **kwargs):
        super(SlowFast, self).__init__()
        self.slow_temporal_stride = slow_temporal_stride
        self.fast_temporal_stride = fast_temporal_stride
        self.slow_frames = slow_frames
        self.fast_frames = fast_frames
        self.num_segments = num_segments
        self.num_crop = num_crop
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std

        with self.name_scope():
            # build fast pathway
            self.fast_inplanes = 8
            fast = nn.HybridSequential(prefix='fast_')
            with fast.name_scope():
                self.fast_conv1 = nn.Conv3D(in_channels=3, channels=8, kernel_size=(5, 7, 7), strides=(1, 2, 2), padding=(2, 3, 3), use_bias=False)
                self.fast_bn1 = norm_layer(in_channels=8, **({} if norm_kwargs is None else norm_kwargs))
                self.fast_relu = nn.Activation('relu')
                self.fast_maxpool = nn.MaxPool3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding=(0, 1, 1))
                self.fast_maxpool2 = nn.MaxPool3D(pool_size=(1, 1, 1), strides=(1, 1, 1), padding=(0, 0, 0))
            self.fast_res2 = self._make_layer_fast(block, planes=8, blocks=layers[0], head_conv=3, norm_layer=norm_layer, norm_kwargs=norm_kwargs, layer_name='fast_res2_')
            self.fast_res3 = self._make_layer_fast(block, planes=16, blocks=layers[1], strides=2, head_conv=3, norm_layer=norm_layer, norm_kwargs=norm_kwargs, layer_name='fast_res3_')
            self.fast_res4 = self._make_layer_fast(block, planes=32, blocks=layers[2], strides=2, head_conv=3, norm_layer=norm_layer, norm_kwargs=norm_kwargs, layer_name='fast_res4_')
            self.fast_res5 = self._make_layer_fast(block, planes=64, blocks=layers[3], strides=2, head_conv=3, norm_layer=norm_layer, norm_kwargs=norm_kwargs, layer_name='fast_res5_')

            # build lateral connections
            self.lateral_p1 = nn.HybridSequential(prefix='lateral_p1_')
            with self.lateral_p1.name_scope():
                self.lateral_p1.add(nn.Conv3D(in_channels=8, channels=8*2, kernel_size=(5, 1, 1), strides=(8, 1, 1), padding=(2, 0, 0), use_bias=False))
                self.lateral_p1.add(norm_layer(in_channels=8*2, **({} if norm_kwargs is None else norm_kwargs)))
                self.lateral_p1.add(nn.Activation('relu'))
            self.lateral_res2 = nn.HybridSequential(prefix='lateral_res2_')
            with self.lateral_res2.name_scope():
                self.lateral_res2.add(nn.Conv3D(in_channels=32, channels=32*2, kernel_size=(5, 1, 1), strides=(8, 1, 1), padding=(2, 0, 0), use_bias=False))
                self.lateral_res2.add(norm_layer(in_channels=32*2, **({} if norm_kwargs is None else norm_kwargs)))
                self.lateral_res2.add(nn.Activation('relu'))
            self.lateral_res3 = nn.HybridSequential(prefix='lateral_res3_')
            with self.lateral_res3.name_scope():
                self.lateral_res3.add(nn.Conv3D(in_channels=64, channels=64*2, kernel_size=(5, 1, 1), strides=(8, 1, 1), padding=(2, 0, 0), use_bias=False))
                self.lateral_res3.add(norm_layer(in_channels=64*2, **({} if norm_kwargs is None else norm_kwargs)))
                self.lateral_res3.add(nn.Activation('relu'))
            self.lateral_res4 = nn.HybridSequential(prefix='lateral_res4_')
            with self.lateral_res4.name_scope():
                self.lateral_res4.add(nn.Conv3D(in_channels=128, channels=128*2, kernel_size=(5, 1, 1), strides=(8, 1, 1), padding=(2, 0, 0), use_bias=False))
                self.lateral_res4.add(norm_layer(in_channels=128*2, **({} if norm_kwargs is None else norm_kwargs)))
                self.lateral_res4.add(nn.Activation('relu'))

            # build slow pathway
            self.slow_inplanes = 64 + 64 // 8 * 2
            slow = nn.HybridSequential(prefix='slow_')
            with slow.name_scope():
                self.slow_conv1 = nn.Conv3D(in_channels=3, channels=64, kernel_size=(1, 7, 7), strides=(1, 2, 2), padding=(0, 3, 3), use_bias=False)
                self.slow_bn1 = norm_layer(in_channels=64, **({} if norm_kwargs is None else norm_kwargs))
                self.slow_relu = nn.Activation('relu')
                self.slow_maxpool = nn.MaxPool3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding=(0, 1, 1))
                self.slow_maxpool2 = nn.MaxPool3D(pool_size=(1, 1, 1), strides=(1, 1, 1), padding=(0, 0, 0))
            self.slow_res2 = self._make_layer_slow(block, planes=64, blocks=layers[0], head_conv=1, norm_layer=norm_layer, norm_kwargs=norm_kwargs, layer_name='slow_res2_')
            self.slow_res3 = self._make_layer_slow(block, planes=128, blocks=layers[1], strides=2, head_conv=1, norm_layer=norm_layer, norm_kwargs=norm_kwargs, layer_name='slow_res3_')
            self.slow_res4 = self._make_layer_slow(block, planes=256, blocks=layers[2], strides=2, head_conv=3, norm_layer=norm_layer, norm_kwargs=norm_kwargs, layer_name='slow_res4_')
            self.slow_res5 = self._make_layer_slow(block, planes=512, blocks=layers[3], strides=2, head_conv=3, norm_layer=norm_layer, norm_kwargs=norm_kwargs, layer_name='slow_res5_')

            # build classifier
            self.avg = nn.GlobalAvgPool3D()
            self.dp = nn.Dropout(rate=self.dropout_ratio)
            self.feat_dim = self.fast_inplanes + 2048
            self.fc = nn.Dense(in_units=self.feat_dim, units=nclass, weight_initializer=init.Normal(sigma=self.init_std), use_bias=True)

            self.initialize(init.MSRAPrelu(), ctx=ctx)

    def hybrid_forward(self, F, x):
        fast_input = F.slice(x, begin=(None, None, 0, None, None), end=(None, None, self.fast_frames, None, None))
        slow_input = F.slice(x, begin=(None, None, self.fast_frames, None, None), end=(None, None, self.fast_frames + self.slow_frames, None, None))

        fast, lateral = self.FastPath(F, fast_input)
        slow = self.SlowPath(F, slow_input, lateral)
        x = F.concat(slow, fast, dim=1)                 # bx2304

        # segmental consensus
        x = F.reshape(x, shape=(-1, self.num_segments * self.num_crop, self.feat_dim))
        x = F.mean(x, axis=1)

        x = self.dp(x)
        x = self.fc(x)                                  # bxnclass
        return x

    def SlowPath(self, F, x, lateral):
        x = self.slow_conv1(x)                          # bx64x4x112x112, input is bx3x4x224x224
        x = self.slow_bn1(x)
        x = self.slow_relu(x)
        pool1 = self.slow_maxpool(x)                    # bx64x4x56x56
        pool1_lat = F.concat(pool1, lateral[0], dim=1)  # bx80x4x56x56

        res2 = self.slow_res2(pool1_lat)                # bx256x4x56x56
        res2_lat = F.concat(res2, lateral[1], dim=1)    # bx320x4x56x56

        res2_lat = self.slow_maxpool2(res2_lat)         # bx320x4x56x56

        res3 = self.slow_res3(res2_lat)                 # bx512x4x28x28
        res3_lat = F.concat(res3, lateral[2], dim=1)    # bx640x4x28x28

        res4 = self.slow_res4(res3_lat)                 # bx1024x4x14x14
        res4_lat = F.concat(res4, lateral[3], dim=1)    # bx1280x4x14x14

        res5 = self.slow_res5(res4_lat)                 # bx2048x4x7x7
        out = self.avg(res5)                            # bx2048x1x1x1
        out = F.squeeze(out, axis=(2, 3, 4))            # bx2048
        return out

    def FastPath(self, F, x):
        lateral = []
        x = self.fast_conv1(x)                          # bx8x32x112x112, input is bx3x32x224x224
        x = self.fast_bn1(x)
        x = self.fast_relu(x)
        pool1 = self.fast_maxpool(x)                    # bx8x32x56x56
        lateral_p = self.lateral_p1(pool1)              # bx16x4x56x56
        lateral.append(lateral_p)

        res2 = self.fast_res2(pool1)                    # bx32x32x56x56
        lateral_res2 = self.lateral_res2(res2)          # bx64x4x56x56
        lateral.append(lateral_res2)

        res2 = self.fast_maxpool2(res2)                 # bx32x32x56x56

        res3 = self.fast_res3(res2)                     # bx64x32x28x28
        lateral_res3 = self.lateral_res3(res3)          # bx128x4x28x28
        lateral.append(lateral_res3)

        res4 = self.fast_res4(res3)                     # bx128x32x14x14
        lateral_res4 = self.lateral_res4(res4)          # bx256x4x14x14
        lateral.append(lateral_res4)

        res5 = self.fast_res5(res4)                     # bx256x32x7x7
        out = self.avg(res5)                            # bx256x1x1x1
        out = F.squeeze(out, axis=(2, 3, 4))            # bx256
        return out, lateral

    def _make_layer_fast(self,
                         block,
                         planes,
                         blocks,
                         strides=1,
                         head_conv=1,
                         norm_layer=BatchNorm,
                         norm_kwargs=None,
                         layer_name=''):
        downsample = None
        if strides != 1 or self.fast_inplanes != planes * block.expansion:
            downsample = nn.HybridSequential(prefix=layer_name+'downsample_')
            with downsample.name_scope():
                downsample.add(nn.Conv3D(in_channels=self.fast_inplanes,
                                         channels=planes * block.expansion,
                                         kernel_size=1,
                                         strides=(1, strides, strides),
                                         use_bias=False))
                downsample.add(norm_layer(in_channels=planes * block.expansion, **({} if norm_kwargs is None else norm_kwargs)))

        layers = nn.HybridSequential(prefix=layer_name)
        cnt = 0
        with layers.name_scope():
            layers.add(block(inplanes=self.fast_inplanes,
                             planes=planes,
                             strides=strides,
                             downsample=downsample,
                             head_conv=head_conv,
                             layer_name='block%d_' % cnt))
            self.fast_inplanes = planes * block.expansion
            cnt += 1
            for _ in range(1, blocks):
                layers.add(block(inplanes=self.fast_inplanes,
                                 planes=planes,
                                 head_conv=head_conv,
                                 layer_name='block%d_' % cnt))
                cnt += 1
        return layers

    def _make_layer_slow(self,
                         block,
                         planes,
                         blocks,
                         strides=1,
                         head_conv=1,
                         norm_layer=BatchNorm,
                         norm_kwargs=None,
                         layer_name=''):
        downsample = None
        if strides != 1 or self.slow_inplanes != planes * block.expansion:
            downsample = nn.HybridSequential(prefix=layer_name+'downsample_')
            with downsample.name_scope():
                downsample.add(nn.Conv3D(in_channels=self.slow_inplanes,
                                         channels=planes * block.expansion,
                                         kernel_size=1,
                                         strides=(1, strides, strides),
                                         use_bias=False))
                downsample.add(norm_layer(in_channels=planes * block.expansion, **({} if norm_kwargs is None else norm_kwargs)))

        layers = nn.HybridSequential(prefix=layer_name)
        cnt = 0
        with layers.name_scope():
            layers.add(block(inplanes=self.slow_inplanes,
                             planes=planes,
                             strides=strides,
                             downsample=downsample,
                             head_conv=head_conv,
                             layer_name='block%d_' % cnt))
            self.slow_inplanes = planes * block.expansion
            cnt += 1
            for _ in range(1, blocks):
                layers.add(block(inplanes=self.slow_inplanes,
                                 planes=planes,
                                 head_conv=head_conv,
                                 layer_name='block%d_' % cnt))
                cnt += 1
        self.slow_inplanes = planes * block.expansion + planes * block.expansion // 8 * 2
        return layers

def slowfast_4x16_resnet50_kinetics400(nclass=400, pretrained=False, pretrained_base=True,
                                       use_tsn=False, num_segments=1, num_crop=1,
                                       partial_bn=False,
                                       root='~/.mxnet/models', ctx=cpu(), **kwargs):
    r"""SlowFast networks (SlowFast) from
    `"SlowFast Networks for Video Recognition"
    <https://arxiv.org/abs/1812.03982>`_ paper.

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    partial_bn : bool, default False
        Freeze all batch normalization layers during training except the first layer.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """

    model = SlowFast(nclass=nclass,
                     layers=[3, 4, 6, 3],
                     pretrained=pretrained,
                     pretrained_base=pretrained_base,
                     num_segments=num_segments,
                     num_crop=num_crop,
                     partial_bn=partial_bn,
                     ctx=ctx,
                     **kwargs)

    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('slowfast_4x16_resnet50_kinetics400',
                                             tag=pretrained, root=root), ctx=ctx)
        from ...data import Kinetics400Attr
        attrib = Kinetics400Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)

    return model
