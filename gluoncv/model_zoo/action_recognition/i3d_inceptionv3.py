# pylint: disable=line-too-long,too-many-lines,missing-docstring,arguments-differ,unused-argument

__all__ = ['I3D_InceptionV3', 'i3d_inceptionv3_kinetics400']

from mxnet import nd
from mxnet import init
from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm
from mxnet.gluon.contrib.nn import HybridConcurrent
from gluoncv.model_zoo.inception import inception_v3

def _make_basic_conv(in_channels, channels, norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
    out = nn.HybridSequential(prefix='')
    out.add(nn.Conv3D(in_channels=in_channels, channels=channels, use_bias=False, **kwargs))
    out.add(norm_layer(in_channels=channels, epsilon=0.001, **({} if norm_kwargs is None else norm_kwargs)))
    out.add(nn.Activation('relu'))
    return out

def _make_branch(use_pool, norm_layer, norm_kwargs, *conv_settings):
    out = nn.HybridSequential(prefix='')
    if use_pool == 'avg':
        out.add(nn.AvgPool3D(pool_size=3, strides=1, padding=1))
    elif use_pool == 'max':
        out.add(nn.MaxPool3D(pool_size=3, strides=2, padding=(1, 0, 0)))
    setting_names = ['in_channels', 'channels', 'kernel_size', 'strides', 'padding']
    for setting in conv_settings:
        kwargs = {}
        for i, value in enumerate(setting):
            if value is not None:
                if setting_names[i] == 'in_channels':
                    in_channels = value
                elif setting_names[i] == 'channels':
                    channels = value
                else:
                    kwargs[setting_names[i]] = value
        out.add(_make_basic_conv(in_channels, channels, norm_layer, norm_kwargs, **kwargs))
    return out

def _make_A(in_channels, pool_features, prefix, norm_layer, norm_kwargs):
    out = HybridConcurrent(axis=1, prefix=prefix)
    with out.name_scope():
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 64, 1, None, None)))
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 48, 1, None, None),
                             (48, 64, 5, None, 2)))
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 64, 1, None, None),
                             (64, 96, 3, None, 1),
                             (96, 96, 3, None, 1)))
        out.add(_make_branch('avg', norm_layer, norm_kwargs,
                             (in_channels, pool_features, 1, None, None)))
    return out

def _make_B(prefix, norm_layer, norm_kwargs):
    out = HybridConcurrent(axis=1, prefix=prefix)
    with out.name_scope():
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (288, 384, 3, 2, (1, 0, 0))))
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (288, 64, 1, None, None),
                             (64, 96, 3, None, 1),
                             (96, 96, 3, 2, (1, 0, 0))))
        out.add(_make_branch('max', norm_layer, norm_kwargs))
    return out

def _make_C(in_channels, channels_7x7, prefix, norm_layer, norm_kwargs):
    out = HybridConcurrent(axis=1, prefix=prefix)
    with out.name_scope():
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 192, 1, None, None)))
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, channels_7x7, 1, None, None),
                             (channels_7x7, channels_7x7, (7, 1, 7), None, (3, 0, 3)),
                             (channels_7x7, 192, (7, 7, 1), None, (3, 3, 0))))
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, channels_7x7, 1, None, None),
                             (channels_7x7, channels_7x7, (7, 7, 1), None, (3, 3, 0)),
                             (channels_7x7, channels_7x7, (7, 1, 7), None, (3, 0, 3)),
                             (channels_7x7, channels_7x7, (7, 7, 1), None, (3, 3, 0)),
                             (channels_7x7, 192, (7, 1, 7), None, (3, 0, 3))))
        out.add(_make_branch('avg', norm_layer, norm_kwargs,
                             (in_channels, 192, 1, None, None)))
    return out

def _make_D(prefix, norm_layer, norm_kwargs):
    out = HybridConcurrent(axis=1, prefix=prefix)
    with out.name_scope():
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (768, 192, 1, None, None),
                             (192, 320, 3, 2, (1, 0, 0))))
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (768, 192, 1, None, None),
                             (192, 192, (7, 1, 7), None, (3, 0, 3)),
                             (192, 192, (7, 7, 1), None, (3, 3, 0)),
                             (192, 192, 3, 2, (1, 0, 0))))
        out.add(_make_branch('max', norm_layer, norm_kwargs))
    return out

def _make_E(in_channels, prefix, norm_layer, norm_kwargs):
    out = HybridConcurrent(axis=1, prefix=prefix)
    with out.name_scope():
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 320, 1, None, None)))

        branch_3x3 = nn.HybridSequential(prefix='')
        out.add(branch_3x3)
        branch_3x3.add(_make_branch(None, norm_layer, norm_kwargs,
                                    (in_channels, 384, 1, None, None)))
        branch_3x3_split = HybridConcurrent(axis=1, prefix='')
        branch_3x3_split.add(_make_branch(None, norm_layer, norm_kwargs,
                                          (384, 384, (3, 1, 3), None, (1, 0, 1))))
        branch_3x3_split.add(_make_branch(None, norm_layer, norm_kwargs,
                                          (384, 384, (3, 3, 1), None, (1, 1, 0))))
        branch_3x3.add(branch_3x3_split)

        branch_3x3dbl = nn.HybridSequential(prefix='')
        out.add(branch_3x3dbl)
        branch_3x3dbl.add(_make_branch(None, norm_layer, norm_kwargs,
                                       (in_channels, 448, 1, None, None),
                                       (448, 384, 3, None, 1)))
        branch_3x3dbl_split = HybridConcurrent(axis=1, prefix='')
        branch_3x3dbl.add(branch_3x3dbl_split)
        branch_3x3dbl_split.add(_make_branch(None, norm_layer, norm_kwargs,
                                             (384, 384, (3, 1, 3), None, (1, 0, 1))))
        branch_3x3dbl_split.add(_make_branch(None, norm_layer, norm_kwargs,
                                             (384, 384, (3, 3, 1), None, (1, 1, 0))))

        out.add(_make_branch('avg', norm_layer, norm_kwargs,
                             (in_channels, 192, 1, None, None)))
    return out

class I3D_InceptionV3(HybridBlock):
    r"""Inception v3 model from
    `"Rethinking the Inception Architecture for Computer Vision"
    <http://arxiv.org/abs/1512.00567>`_ paper.

    Inflated 3D model (I3D) from
    `"Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset"
    <https://arxiv.org/abs/1705.07750>`_ paper.

    This model definition file is written by Brais and modified by Yi.

    Parameters
    ----------
    nclass : int
        Number of classes in the training dataset.
    pretrained : bool or str.
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True.
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    dropout_ratio : float, default is 0.5.
        The dropout rate of a dropout layer.
        The larger the value, the more strength to prevent overfitting.
    num_segments : int, default is 1.
        Number of segments used to evenly divide a video.
    num_crop : int, default is 1.
        Number of crops used during evaluation, choices are 1, 3 or 10.
    feat_ext : bool.
        Whether to extract features before dense classification layer or
        do a complete forward pass.
    init_std : float, default is 0.001.
        Standard deviation value when initialize the dense layers.
    ctx : Context, default CPU.
        The context in which to load the pretrained weights.
    partial_bn : bool, default False.
        Freeze all batch normalization layers during training except the first layer.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    def __init__(self, nclass=1000, pretrained=False, pretrained_base=True,
                 num_segments=1, num_crop=1, feat_ext=False,
                 dropout_ratio=0.5, init_std=0.01, partial_bn=False,
                 ctx=None, norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(I3D_InceptionV3, self).__init__(**kwargs)

        self.num_segments = num_segments
        self.num_crop = num_crop
        self.feat_dim = 2048
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.feat_ext = feat_ext

        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')
            self.features.add(_make_basic_conv(in_channels=3, channels=32, kernel_size=3, strides=2, padding=(1, 0, 0),
                                               norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            if partial_bn:
                if norm_kwargs is not None:
                    norm_kwargs['use_global_stats'] = True
                else:
                    norm_kwargs = {}
                    norm_kwargs['use_global_stats'] = True

            self.features.add(_make_basic_conv(in_channels=32, channels=32, kernel_size=3, padding=(1, 0, 0),
                                               norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            self.features.add(_make_basic_conv(in_channels=32, channels=64, kernel_size=3, padding=1,
                                               norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            self.features.add(nn.MaxPool3D(pool_size=3, strides=(1, 2, 2), padding=(1, 0, 0)))
            self.features.add(_make_basic_conv(in_channels=64, channels=80, kernel_size=1,
                                               norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            self.features.add(_make_basic_conv(in_channels=80, channels=192, kernel_size=3, padding=(1, 0, 0),
                                               norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            self.features.add(nn.MaxPool3D(pool_size=3, strides=(1, 2, 2), padding=(1, 0, 0)))
            self.features.add(_make_A(192, 32, 'A1_', norm_layer, norm_kwargs))
            self.features.add(_make_A(256, 64, 'A2_', norm_layer, norm_kwargs))
            self.features.add(_make_A(288, 64, 'A3_', norm_layer, norm_kwargs))
            self.features.add(_make_B('B_', norm_layer, norm_kwargs))
            self.features.add(_make_C(768, 128, 'C1_', norm_layer, norm_kwargs))
            self.features.add(_make_C(768, 160, 'C2_', norm_layer, norm_kwargs))
            self.features.add(_make_C(768, 160, 'C3_', norm_layer, norm_kwargs))
            self.features.add(_make_C(768, 192, 'C4_', norm_layer, norm_kwargs))
            self.features.add(_make_D('D_', norm_layer, norm_kwargs))
            self.features.add(_make_E(1280, 'E1_', norm_layer, norm_kwargs))
            self.features.add(_make_E(2048, 'E2_', norm_layer, norm_kwargs))
            self.features.add(nn.GlobalAvgPool3D())

            self.head = nn.HybridSequential(prefix='')
            self.head.add(nn.Dropout(rate=self.dropout_ratio))
            self.output = nn.Dense(units=nclass, in_units=self.feat_dim, weight_initializer=init.Normal(sigma=self.init_std))
            self.head.add(self.output)

            self.features.initialize(ctx=ctx)
            self.head.initialize(ctx=ctx)

            if pretrained_base and not pretrained:
                inceptionv3_2d = inception_v3(pretrained=True)
                weights2d = inceptionv3_2d.collect_params()
                weights3d = self.collect_params()
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
        x = self.features(x)
        x = F.squeeze(x, axis=(2, 3, 4))

        # segmental consensus
        x = F.reshape(x, shape=(-1, self.num_segments * self.num_crop, self.feat_dim))
        x = F.mean(x, axis=1)

        if self.feat_ext:
            return x

        x = self.head(x)
        return x

def i3d_inceptionv3_kinetics400(nclass=400, pretrained=False, pretrained_base=True,
                                ctx=cpu(), root='~/.mxnet/models', use_tsn=False,
                                num_segments=1, num_crop=1, partial_bn=False,
                                feat_ext=False, **kwargs):
    r"""Inception v3 model trained on Kinetics400 dataset from
    `"Rethinking the Inception Architecture for Computer Vision"
    <http://arxiv.org/abs/1512.00567>`_ paper.

    Inflated 3D model (I3D) from
    `"Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset"
    <https://arxiv.org/abs/1705.07750>`_ paper.

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
    feat_ext : bool.
        Whether to extract features before dense classification layer or
        do a complete forward pass.
    """

    model = I3D_InceptionV3(nclass=nclass,
                            partial_bn=partial_bn,
                            pretrained=pretrained,
                            pretrained_base=pretrained_base,
                            feat_ext=feat_ext,
                            num_segments=num_segments,
                            num_crop=num_crop,
                            dropout_ratio=0.5,
                            init_std=0.01,
                            ctx=ctx,
                            **kwargs)

    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('i3d_inceptionv3_kinetics400',
                                             tag=pretrained, root=root), ctx=ctx)
        from ...data import Kinetics400Attr
        attrib = Kinetics400Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)

    return model
