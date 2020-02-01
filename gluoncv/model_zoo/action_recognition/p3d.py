"""P3D, implemented in Gluon. https://arxiv.org/abs/1711.10305.
Code adapted from https://github.com/qijiezhao/pseudo-3d-pytorch."""
# pylint: disable=arguments-differ,unused-argument,line-too-long

__all__ = ['P3D', 'p3d_resnet50_kinetics400', 'p3d_resnet101_kinetics400']

from mxnet import init
from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm

def conv1x3x3(in_planes, out_planes, spatial_stride=1, temporal_stride=1, dilation=1):
    """1x3x3 convolution with padding"""
    return nn.Conv3D(in_channels=in_planes,
                     channels=out_planes,
                     kernel_size=(1, 3, 3),
                     strides=(temporal_stride, spatial_stride, spatial_stride),
                     padding=(0, dilation, dilation),
                     dilation=dilation,
                     use_bias=False)

def conv3x1x1(in_planes, out_planes, spatial_stride=1, temporal_stride=1, dilation=1):
    """3x1x1 convolution with padding"""
    return nn.Conv3D(in_channels=in_planes,
                     channels=out_planes,
                     kernel_size=(3, 1, 1),
                     strides=(temporal_stride, spatial_stride, spatial_stride),
                     padding=(dilation, 0, 0),
                     dilation=dilation,
                     use_bias=False)

class Bottleneck(HybridBlock):
    r"""ResBlock for P3D

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
    num_layers : int.
        The current depth of this layer.
    depth_3d : int.
        Number of 3D layers in the network. For example,
        a P3D with ResNet50 backbone has a depth_3d of 13, which is the sum of 3, 4 and 6.
    block_design : tuple of str.
        Different designs for each block, from 'A', 'B' or 'C'.
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
                 num_layers=0,
                 depth_3d=13,
                 block_design=('A', 'B', 'C'),
                 norm_layer=BatchNorm,
                 norm_kwargs=None,
                 layer_name='',
                 **kwargs):
        super(Bottleneck, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.spatial_tride = spatial_stride
        self.temporal_tride = temporal_stride
        self.dilation = dilation
        self.downsample = downsample
        self.num_layers = num_layers
        self.depth_3d = depth_3d
        self.block_design = block_design
        self.cycle = len(self.block_design)

        if self.downsample is not None:
            strides = (1, 2, 2)
        else:
            strides = spatial_stride

        with self.name_scope():
            # first block
            if self.num_layers < self.depth_3d:
                if self.num_layers == 0:
                    strides = 1
                self.conv1 = nn.Conv3D(in_channels=inplanes,
                                       channels=planes,
                                       kernel_size=1,
                                       strides=strides,
                                       use_bias=False)
                self.bn1 = norm_layer(in_channels=planes,
                                      **({} if norm_kwargs is None else norm_kwargs))
            else:
                if self.num_layers == self.depth_3d:
                    strides = 2
                else:
                    strides = 1
                self.conv1 = nn.Conv2D(in_channels=inplanes,
                                       channels=planes,
                                       kernel_size=1,
                                       strides=strides,
                                       use_bias=False)
                self.bn1 = norm_layer(in_channels=planes,
                                      **({} if norm_kwargs is None else norm_kwargs))

            # second block
            self.block_id = int(num_layers)
            self.design = list(self.block_design)[self.block_id % self.cycle]
            if self.num_layers < self.depth_3d:
                self.conv2 = conv1x3x3(planes, planes)
                self.bn2 = norm_layer(in_channels=planes,
                                      **({} if norm_kwargs is None else norm_kwargs))

                self.conv_temporal = conv3x1x1(planes, planes)
                self.bn_temporal = norm_layer(in_channels=planes,
                                              **({} if norm_kwargs is None else norm_kwargs))
            else:
                self.conv2 = nn.Conv2D(in_channels=planes, channels=planes, kernel_size=3,
                                       strides=1, padding=1, use_bias=False)
                self.bn2 = norm_layer(in_channels=planes,
                                      **({} if norm_kwargs is None else norm_kwargs))

            # third block
            if self.num_layers < self.depth_3d:
                self.conv3 = nn.Conv3D(in_channels=planes,
                                       channels=planes * self.expansion,
                                       kernel_size=1,
                                       use_bias=False)
                self.bn3 = norm_layer(in_channels=planes * self.expansion,
                                      **({} if norm_kwargs is None else norm_kwargs))
            else:
                self.conv3 = nn.Conv2D(in_channels=planes,
                                       channels=planes * self.expansion,
                                       kernel_size=1,
                                       use_bias=False)
                self.bn3 = norm_layer(in_channels=planes * self.expansion,
                                      **({} if norm_kwargs is None else norm_kwargs))

            self.relu = nn.Activation('relu')

    def P3DA(self, x):
        """P3D-A unit
        Stacked architecture by making temporal 1D filters (T)
        follow spatial 2D filters (S) in a cascaded manner.
        """
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv_temporal(x)
        x = self.bn_temporal(x)
        x = self.relu(x)
        return x

    def P3DB(self, x):
        """P3D-B unit
        Parallel architecture by making temporal 1D filters (T)
        and spatial 2D filters (S) in different pathways.
        """
        spatial_x = self.conv2(x)
        spatial_x = self.bn2(spatial_x)
        spatial_x = self.relu(spatial_x)

        temporal_x = self.conv_temporal(x)
        temporal_x = self.bn_temporal(temporal_x)
        temporal_x = self.relu(temporal_x)
        return spatial_x + temporal_x

    def P3DC(self, x):
        """P3D-C unit
        A compromise design between P3D-A and P3D-B.
        """
        spatial_x = self.conv2(x)
        spatial_x = self.bn2(spatial_x)
        spatial_x = self.relu(spatial_x)

        st_residual_x = self.conv_temporal(spatial_x)
        st_residual_x = self.bn_temporal(st_residual_x)
        st_residual_x = self.relu(st_residual_x)
        return spatial_x + st_residual_x

    def hybrid_forward(self, F, x):
        """Hybrid forward of a ResBlock in P3D."""
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        if self.block_id < self.depth_3d:
            if self.design == 'A':
                out = self.P3DA(out)
            elif self.design == 'B':
                out = self.P3DB(out)
            elif self.design == 'C':
                out = self.P3DC(out)
            else:
                print('We do not support %s building block for P3D networks. \
                      Please try A, B or C.' % self.design)
        else:
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out = F.Activation(out + identity, act_type='relu')
        return out

class P3D(HybridBlock):
    r"""
    The Pseudo 3D network (P3D).
    Learning Spatio-Temporal Representation with Pseudo-3D Residual Networks.
    ICCV, 2017. https://arxiv.org/abs/1711.10305

    Parameters
    ----------
    nclass : int
        Number of classes in the training dataset.
    block : Block, default is `Bottleneck`.
        Class for the residual block.
    layers : list of int
        Numbers of layers in each block
    block_design : tuple of str.
        Different designs for each block, from 'A', 'B' or 'C'.
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
    def __init__(self, nclass, block, layers, shortcut_type='B',
                 block_design=('A', 'B', 'C'), dropout_ratio=0.5,
                 num_segments=1, num_crop=1, feat_ext=False,
                 init_std=0.001, ctx=None, partial_bn=False,
                 norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(P3D, self).__init__()
        self.shortcut_type = shortcut_type
        self.block_design = block_design
        self.partial_bn = partial_bn
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.num_segments = num_segments
        self.num_crop = num_crop
        self.feat_ext = feat_ext
        self.inplanes = 64
        self.feat_dim = 512 * block.expansion

        with self.name_scope():
            self.conv1 = nn.Conv3D(in_channels=3, channels=64, kernel_size=(1, 7, 7),
                                   strides=(1, 2, 2), padding=(0, 3, 3), use_bias=False)
            self.bn1 = norm_layer(in_channels=64, **({} if norm_kwargs is None else norm_kwargs))
            self.relu = nn.Activation('relu')
            self.pool = nn.MaxPool3D(pool_size=(2, 3, 3), strides=2, padding=(0, 1, 1))
            self.pool2 = nn.MaxPool3D(pool_size=(2, 1, 1), strides=(2, 1, 1), padding=0)

            if self.partial_bn:
                if norm_kwargs is not None:
                    norm_kwargs['use_global_stats'] = True
                else:
                    norm_kwargs = {}
                    norm_kwargs['use_global_stats'] = True

            # 3D layers are only for (layers1, layers2 and layers3), layers4 is C2D
            self.depth_3d = sum(layers[:3])
            self.layer_cnt = 0

            self.layer1 = self._make_res_layer(block=block,
                                               planes=64,
                                               blocks=layers[0],
                                               layer_name='layer1_')
            self.layer2 = self._make_res_layer(block=block,
                                               planes=128,
                                               blocks=layers[1],
                                               spatial_stride=2,
                                               layer_name='layer2_')
            self.layer3 = self._make_res_layer(block=block,
                                               planes=256,
                                               blocks=layers[2],
                                               spatial_stride=2,
                                               layer_name='layer3_')
            self.layer4 = self._make_res_layer(block=block,
                                               planes=512,
                                               blocks=layers[3],
                                               spatial_stride=2,
                                               layer_name='layer4_')

            self.avgpool = nn.GlobalAvgPool2D()
            self.dropout = nn.Dropout(rate=self.dropout_ratio)
            self.fc = nn.Dense(in_units=self.feat_dim, units=nclass,
                               weight_initializer=init.Normal(sigma=self.init_std))

    def _make_res_layer(self,
                        block,
                        planes,
                        blocks,
                        shortcut_type='B',
                        block_design=('A', 'B', 'C'),
                        spatial_stride=1,
                        temporal_stride=1,
                        norm_layer=BatchNorm,
                        norm_kwargs=None,
                        layer_name=''):
        """Build each stage of a ResNet"""
        downsample = None

        if self.layer_cnt < self.depth_3d:
            if spatial_stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.HybridSequential(prefix=layer_name + 'downsample_')
                with downsample.name_scope():
                    downsample.add(nn.Conv3D(in_channels=self.inplanes,
                                             channels=planes * block.expansion,
                                             kernel_size=1,
                                             strides=(temporal_stride, spatial_stride, spatial_stride),
                                             use_bias=False))
                    downsample.add(norm_layer(in_channels=planes * block.expansion,
                                              **({} if norm_kwargs is None else norm_kwargs)))
        else:
            if spatial_stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.HybridSequential(prefix=layer_name + 'downsample_')
                with downsample.name_scope():
                    downsample.add(nn.Conv2D(in_channels=self.inplanes,
                                             channels=planes * block.expansion,
                                             kernel_size=1,
                                             strides=spatial_stride,
                                             use_bias=False))
                    downsample.add(norm_layer(in_channels=planes * block.expansion,
                                              **({} if norm_kwargs is None else norm_kwargs)))

        layers = nn.HybridSequential(prefix=layer_name)
        with layers.name_scope():
            layers.add(block(inplanes=self.inplanes,
                             planes=planes,
                             spatial_stride=spatial_stride,
                             temporal_stride=temporal_stride,
                             downsample=downsample,
                             num_layers=self.layer_cnt,
                             depth_3d=self.depth_3d,
                             block_design=block_design))

            self.layer_cnt += 1
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.add(block(inplanes=self.inplanes,
                                 planes=planes,
                                 spatial_stride=1,
                                 temporal_stride=1,
                                 num_layers=self.layer_cnt,
                                 depth_3d=self.depth_3d,
                                 block_design=block_design))
                self.layer_cnt += 1
        return layers

    def hybrid_forward(self, F, x):
        """Hybrid forward of P3D net"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.pool2(self.layer1(x))
        x = self.pool2(self.layer2(x))
        x = self.pool2(self.layer3(x))
        x = F.reshape(x, (-1, 0, 1, 0, 0))    # 0 keeps the original dimension
        x = F.squeeze(x, axis=2)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = F.squeeze(x, axis=(2, 3))

        # segmental consensus
        x = F.reshape(x, shape=(-1, self.num_segments * self.num_crop, self.feat_dim))
        x = F.mean(x, axis=1)

        if self.feat_ext:
            return x

        x = self.fc(self.dropout(x))
        return x

def p3d_resnet50_kinetics400(nclass=400, pretrained=False, pretrained_base=True,
                             root='~/.mxnet/models', num_segments=1, num_crop=1,
                             feat_ext=False, ctx=cpu(), **kwargs):
    r"""The Pseudo 3D network (P3D) with ResNet50 backbone trained on Kinetics400 dataset.

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
    feat_ext : bool.
        Whether to extract features before dense classification layer or
        do a complete forward pass.
    """

    model = P3D(nclass=nclass,
                block=Bottleneck,
                layers=[3, 4, 6, 3],
                num_segments=num_segments,
                num_crop=num_crop,
                feat_ext=feat_ext,
                ctx=ctx,
                **kwargs)
    model.initialize(init.MSRAPrelu(), ctx=ctx)

    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('p3d_resnet50_kinetics400',
                                             tag=pretrained, root=root), ctx=ctx)
        from ...data import Kinetics400Attr
        attrib = Kinetics400Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)

    return model

def p3d_resnet101_kinetics400(nclass=400, pretrained=False, pretrained_base=True,
                              root='~/.mxnet/models', num_segments=1, num_crop=1,
                              feat_ext=False, ctx=cpu(), **kwargs):
    r"""The Pseudo 3D network (P3D) with ResNet101 backbone trained on Kinetics400 dataset.

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
    feat_ext : bool.
        Whether to extract features before dense classification layer or
        do a complete forward pass.
    """

    model = P3D(nclass=nclass,
                block=Bottleneck,
                layers=[3, 4, 23, 3],
                num_segments=num_segments,
                num_crop=num_crop,
                feat_ext=feat_ext,
                ctx=ctx,
                **kwargs)
    model.initialize(init.MSRAPrelu(), ctx=ctx)

    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('p3d_resnet101_kinetics400',
                                             tag=pretrained, root=root), ctx=ctx)
        from ...data import Kinetics400Attr
        attrib = Kinetics400Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)

    return model
