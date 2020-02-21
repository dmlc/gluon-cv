# pylint: disable=line-too-long,too-many-lines,missing-docstring,arguments-differ,unused-argument

__all__ = ['I3D_InceptionV1', 'i3d_inceptionv1_kinetics400']

from mxnet import nd
from mxnet import init
from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm
from mxnet.gluon.contrib.nn import HybridConcurrent
from gluoncv.model_zoo.googlenet import googlenet

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
        out.add(nn.MaxPool3D(pool_size=3, strides=1, padding=1))
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

def _make_Mixed_3a(in_channels, pool_features, prefix, norm_layer, norm_kwargs):
    out = HybridConcurrent(axis=1, prefix=prefix)
    with out.name_scope():
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 64, 1, None, None)))
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 96, 1, None, None),
                             (96, 128, 3, None, 1)))
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 16, 1, None, None),
                             (16, 32, 3, None, 1)))
        out.add(_make_branch('max', norm_layer, norm_kwargs,
                             (in_channels, pool_features, 1, None, None)))
    return out

def _make_Mixed_3b(in_channels, pool_features, prefix, norm_layer, norm_kwargs):
    out = HybridConcurrent(axis=1, prefix=prefix)
    with out.name_scope():
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 128, 1, None, None)))
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 128, 1, None, None),
                             (128, 192, 3, None, 1)))
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 32, 1, None, None),
                             (32, 96, 3, None, 1)))
        out.add(_make_branch('max', norm_layer, norm_kwargs,
                             (in_channels, pool_features, 1, None, None)))
    return out

def _make_Mixed_4a(in_channels, pool_features, prefix, norm_layer, norm_kwargs):
    out = HybridConcurrent(axis=1, prefix=prefix)
    with out.name_scope():
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 192, 1, None, None)))
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 96, 1, None, None),
                             (96, 208, 3, None, 1)))
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 16, 1, None, None),
                             (16, 48, 3, None, 1)))
        out.add(_make_branch('max', norm_layer, norm_kwargs,
                             (in_channels, pool_features, 1, None, None)))
    return out

def _make_Mixed_4b(in_channels, pool_features, prefix, norm_layer, norm_kwargs):
    out = HybridConcurrent(axis=1, prefix=prefix)
    with out.name_scope():
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 160, 1, None, None)))
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 112, 1, None, None),
                             (112, 224, 3, None, 1)))
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 24, 1, None, None),
                             (24, 64, 3, None, 1)))
        out.add(_make_branch('max', norm_layer, norm_kwargs,
                             (in_channels, pool_features, 1, None, None)))
    return out

def _make_Mixed_4c(in_channels, pool_features, prefix, norm_layer, norm_kwargs):
    out = HybridConcurrent(axis=1, prefix=prefix)
    with out.name_scope():
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 128, 1, None, None)))
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 128, 1, None, None),
                             (128, 256, 3, None, 1)))
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 24, 1, None, None),
                             (24, 64, 3, None, 1)))
        out.add(_make_branch('max', norm_layer, norm_kwargs,
                             (in_channels, pool_features, 1, None, None)))
    return out

def _make_Mixed_4d(in_channels, pool_features, prefix, norm_layer, norm_kwargs):
    out = HybridConcurrent(axis=1, prefix=prefix)
    with out.name_scope():
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 112, 1, None, None)))
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 144, 1, None, None),
                             (144, 288, 3, None, 1)))
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 32, 1, None, None),
                             (32, 64, 3, None, 1)))
        out.add(_make_branch('max', norm_layer, norm_kwargs,
                             (in_channels, pool_features, 1, None, None)))
    return out

def _make_Mixed_4e(in_channels, pool_features, prefix, norm_layer, norm_kwargs):
    out = HybridConcurrent(axis=1, prefix=prefix)
    with out.name_scope():
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 256, 1, None, None)))
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 160, 1, None, None),
                             (160, 320, 3, None, 1)))
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 32, 1, None, None),
                             (32, 128, 3, None, 1)))
        out.add(_make_branch('max', norm_layer, norm_kwargs,
                             (in_channels, pool_features, 1, None, None)))
    return out

def _make_Mixed_5a(in_channels, pool_features, prefix, norm_layer, norm_kwargs):
    out = HybridConcurrent(axis=1, prefix=prefix)
    with out.name_scope():
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 256, 1, None, None)))
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 160, 1, None, None),
                             (160, 320, 3, None, 1)))
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 32, 1, None, None),
                             (32, 128, 3, None, 1)))
        out.add(_make_branch('max', norm_layer, norm_kwargs,
                             (in_channels, pool_features, 1, None, None)))
    return out

def _make_Mixed_5b(in_channels, pool_features, prefix, norm_layer, norm_kwargs):
    out = HybridConcurrent(axis=1, prefix=prefix)
    with out.name_scope():
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 384, 1, None, None)))
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 192, 1, None, None),
                             (192, 384, 3, None, 1)))
        out.add(_make_branch(None, norm_layer, norm_kwargs,
                             (in_channels, 48, 1, None, None),
                             (48, 128, 3, None, 1)))
        out.add(_make_branch('max', norm_layer, norm_kwargs,
                             (in_channels, pool_features, 1, None, None)))
    return out

class I3D_InceptionV1(HybridBlock):
    r"""Inception v1 model from
    `"Going Deeper with Convolutions"
    <https://arxiv.org/abs/1409.4842>`_ paper.

    Inflated 3D model (I3D) from
    `"Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset"
    <https://arxiv.org/abs/1705.07750>`_ paper.
    Slight differences between this implementation and the original implementation due to padding.

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
        super(I3D_InceptionV1, self).__init__(**kwargs)

        self.num_segments = num_segments
        self.num_crop = num_crop
        self.feat_dim = 1024
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.feat_ext = feat_ext

        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')

            self.features.add(_make_basic_conv(in_channels=3, channels=64, kernel_size=7, strides=2, padding=3, norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            self.features.add(nn.MaxPool3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding=(0, 1, 1)))

            if partial_bn:
                if norm_kwargs is not None:
                    norm_kwargs['use_global_stats'] = True
                else:
                    norm_kwargs = {}
                    norm_kwargs['use_global_stats'] = True

            self.features.add(_make_basic_conv(in_channels=64, channels=64, kernel_size=1, norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            self.features.add(_make_basic_conv(in_channels=64, channels=192, kernel_size=3, padding=(1, 1, 1), norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            self.features.add(nn.MaxPool3D(pool_size=(1, 3, 3), strides=(1, 2, 2), padding=(0, 1, 1)))

            self.features.add(_make_Mixed_3a(192, 32, 'Mixed_3a_', norm_layer, norm_kwargs))
            self.features.add(_make_Mixed_3b(256, 64, 'Mixed_3b_', norm_layer, norm_kwargs))
            self.features.add(nn.MaxPool3D(pool_size=3, strides=(2, 2, 2), padding=(1, 1, 1)))

            self.features.add(_make_Mixed_4a(480, 64, 'Mixed_4a_', norm_layer, norm_kwargs))
            self.features.add(_make_Mixed_4b(512, 64, 'Mixed_4b_', norm_layer, norm_kwargs))
            self.features.add(_make_Mixed_4c(512, 64, 'Mixed_4c_', norm_layer, norm_kwargs))
            self.features.add(_make_Mixed_4d(512, 64, 'Mixed_4d_', norm_layer, norm_kwargs))
            self.features.add(_make_Mixed_4e(528, 128, 'Mixed_4e_', norm_layer, norm_kwargs))
            self.features.add(nn.MaxPool3D(pool_size=2, strides=(2, 2, 2)))

            self.features.add(_make_Mixed_5a(832, 128, 'Mixed_5a_', norm_layer, norm_kwargs))
            self.features.add(_make_Mixed_5b(832, 128, 'Mixed_5b_', norm_layer, norm_kwargs))
            self.features.add(nn.GlobalAvgPool3D())

            self.head = nn.HybridSequential(prefix='')
            self.head.add(nn.Dropout(rate=self.dropout_ratio))
            self.output = nn.Dense(units=nclass, in_units=self.feat_dim, weight_initializer=init.Normal(sigma=self.init_std))
            self.head.add(self.output)

            self.features.initialize(ctx=ctx)
            self.head.initialize(ctx=ctx)

            if pretrained_base and not pretrained:
                inceptionv1_2d = googlenet(pretrained=True)
                weights2d = inceptionv1_2d.collect_params()
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

def i3d_inceptionv1_kinetics400(nclass=400, pretrained=False, pretrained_base=True,
                                ctx=cpu(), root='~/.mxnet/models', use_tsn=False,
                                num_segments=1, num_crop=1, partial_bn=False,
                                feat_ext=False, **kwargs):
    r"""Inception v1 model trained on Kinetics400 dataset from
    `"Going Deeper with Convolutions"
    <https://arxiv.org/abs/1409.4842>`_ paper.

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

    model = I3D_InceptionV1(nclass=nclass,
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
        model.load_parameters(get_model_file('i3d_inceptionv1_kinetics400',
                                             tag=pretrained, root=root), ctx=ctx)
        from ...data import Kinetics400Attr
        attrib = Kinetics400Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)

    return model
