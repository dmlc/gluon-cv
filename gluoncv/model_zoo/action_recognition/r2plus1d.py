# pylint: disable=arguments-differ,unused-argument,line-too-long
"""R2Plus1D, implemented in Gluon. https://arxiv.org/abs/1711.11248.
Code adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/video/resnet.py."""


__all__ = ['R2Plus1D', 'r2plus1d_resnet18_kinetics400',
           'r2plus1d_resnet34_kinetics400', 'r2plus1d_resnet50_kinetics400',
           'r2plus1d_resnet101_kinetics400', 'r2plus1d_resnet152_kinetics400']

from mxnet import init
from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm

def conv3x1x1(in_planes, out_planes, spatial_stride=1, temporal_stride=1, dilation=1):
    """3x1x1 convolution with padding"""
    return nn.Conv3D(in_channels=in_planes,
                     channels=out_planes,
                     kernel_size=(3, 1, 1),
                     strides=(temporal_stride, spatial_stride, spatial_stride),
                     padding=(dilation, 0, 0),
                     dilation=dilation,
                     use_bias=False)

class Conv2Plus1D(HybridBlock):
    r"""Building block of Conv2Plus1D

    Parameters
    ----------
    inplanes : int.
        Input channels of each block.
    planes : int.
        Output channels of each block.
    midplanes : int.
        Intermediate channels of each block.
    stride : int, default is 1.
        Stride in each dimension of 3D convolutional layers in a block.
    padding : int, default is 1.
        Padding in each dimension of the feature map.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    def __init__(self,
                 inplanes,
                 planes,
                 midplanes,
                 stride=1,
                 padding=1,
                 norm_layer=BatchNorm,
                 norm_kwargs=None,
                 **kwargs):
        super(Conv2Plus1D, self).__init__()

        with self.name_scope():
            self.conv1 = nn.Conv3D(in_channels=inplanes,
                                   channels=midplanes,
                                   kernel_size=(1, 3, 3),
                                   strides=(1, stride, stride),
                                   padding=(0, padding, padding),
                                   use_bias=False)
            self.bn1 = norm_layer(in_channels=midplanes,
                                  **({} if norm_kwargs is None else norm_kwargs))
            self.relu = nn.Activation('relu')
            self.conv2 = nn.Conv3D(in_channels=midplanes,
                                   channels=planes,
                                   kernel_size=(3, 1, 1),
                                   strides=(stride, 1, 1),
                                   padding=(padding, 0, 0),
                                   use_bias=False)

    def hybrid_forward(self, F, x):
        """Hybrid forward of a Conv2Plus1D block."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

class BasicBlock(HybridBlock):
    r"""ResNet Basic Block for R2Plus1D

    Parameters
    ----------
    inplanes : int.
        Input channels of each block.
    planes : int.
        Output channels of each block.
    stride : int, default is 1.
        Stride in each dimension of 3D convolutional layers in a block.
    downsample : bool.
        Whether to contain a downsampling layer in the block.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    layer_name : str, default is ''.
        Give a name to current block.
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 norm_layer=BatchNorm, norm_kwargs=None, layer_name='',
                 **kwargs):
        super(BasicBlock, self).__init__()
        self.downsample = downsample

        with self.name_scope():
            midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)
            self.conv1 = Conv2Plus1D(inplanes, planes, midplanes, stride)
            self.bn1 = norm_layer(in_channels=planes,
                                  **({} if norm_kwargs is None else norm_kwargs))
            self.relu = nn.Activation('relu')
            self.conv2 = Conv2Plus1D(planes, planes, midplanes)
            self.bn2 = norm_layer(in_channels=planes,
                                  **({} if norm_kwargs is None else norm_kwargs))

    def hybrid_forward(self, F, x):
        """Hybrid forward of a ResBlock in R2+1D."""
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
    r"""ResNet Bottleneck Block for R2Plus1D

    Parameters
    ----------
    inplanes : int.
        Input channels of each block.
    planes : int.
        Output channels of each block.
    stride : int, default is 1.
        Stride in each dimension of 3D convolutional layers in a block.
    downsample : bool.
        Whether to contain a downsampling layer in the block.
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

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 norm_layer=BatchNorm, norm_kwargs=None, layer_name='',
                 **kwargs):
        super(Bottleneck, self).__init__()
        self.downsample = downsample

        with self.name_scope():
            midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

            # 1x1x1
            self.conv1 = nn.Conv3D(in_channels=inplanes, channels=planes, kernel_size=1, use_bias=False)
            self.bn1 = norm_layer(in_channels=planes,
                                  **({} if norm_kwargs is None else norm_kwargs))
            self.relu = nn.Activation('relu')

            # Second kernel
            self.conv2 = Conv2Plus1D(planes, planes, midplanes, stride)
            self.bn2 = norm_layer(in_channels=planes,
                                  **({} if norm_kwargs is None else norm_kwargs))

            self.conv3 = nn.Conv3D(in_channels=planes, channels=planes * self.expansion,
                                   kernel_size=1, use_bias=False)
            self.bn3 = norm_layer(in_channels=planes * self.expansion,
                                  **({} if norm_kwargs is None else norm_kwargs))

    def hybrid_forward(self, F, x):
        """Hybrid forward of a ResBlock in R2+1D."""
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

class R2Plus1D(HybridBlock):
    r"""The R2+1D network.
    A Closer Look at Spatiotemporal Convolutions for Action Recognition.
    CVPR, 2018. https://arxiv.org/abs/1711.11248

    Parameters
    ----------
    nclass : int
        Number of classes in the training dataset.
    block : Block, default is `Bottleneck`.
        Class for the residual block.
    layers : list of int
        Numbers of layers in each block
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
    def __init__(self, nclass, block, layers, dropout_ratio=0.5,
                 num_segments=1, num_crop=1, feat_ext=False,
                 init_std=0.001, ctx=None, partial_bn=False,
                 norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(R2Plus1D, self).__init__()
        self.partial_bn = partial_bn
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.num_segments = num_segments
        self.num_crop = num_crop
        self.feat_ext = feat_ext
        self.inplanes = 64
        self.feat_dim = 512 * block.expansion

        with self.name_scope():
            self.conv1 = nn.Conv3D(in_channels=3, channels=45, kernel_size=(1, 7, 7),
                                   strides=(1, 2, 2), padding=(0, 3, 3), use_bias=False)
            self.bn1 = norm_layer(in_channels=45, **({} if norm_kwargs is None else norm_kwargs))
            self.relu = nn.Activation('relu')
            self.conv2 = conv3x1x1(in_planes=45, out_planes=64)
            self.bn2 = norm_layer(in_channels=64, **({} if norm_kwargs is None else norm_kwargs))

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

            self.avgpool = nn.GlobalAvgPool3D()
            self.dropout = nn.Dropout(rate=self.dropout_ratio)
            self.fc = nn.Dense(in_units=self.feat_dim, units=nclass,
                               weight_initializer=init.Normal(sigma=self.init_std))

    def hybrid_forward(self, F, x):
        """Hybrid forward of R2+1D net"""
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
        x = F.squeeze(x, axis=(2, 3, 4))

        # segmental consensus
        x = F.reshape(x, shape=(-1, self.num_segments * self.num_crop, self.feat_dim))
        x = F.mean(x, axis=1)

        if self.feat_ext:
            return x

        x = self.fc(self.dropout(x))
        return x

    def _make_res_layer(self,
                        block,
                        planes,
                        blocks,
                        stride=1,
                        norm_layer=BatchNorm,
                        norm_kwargs=None,
                        layer_name=''):
        """Build each stage of a ResNet"""
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.HybridSequential(prefix=layer_name + 'downsample_')
            with downsample.name_scope():
                downsample.add(nn.Conv3D(in_channels=self.inplanes,
                                         channels=planes * block.expansion,
                                         kernel_size=1,
                                         strides=(stride, stride, stride),
                                         use_bias=False))
                downsample.add(norm_layer(in_channels=planes * block.expansion,
                                          **({} if norm_kwargs is None else norm_kwargs)))

        layers = nn.HybridSequential(prefix=layer_name)
        with layers.name_scope():
            layers.add(block(inplanes=self.inplanes,
                             planes=planes,
                             stride=stride,
                             downsample=downsample))

            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.add(block(inplanes=self.inplanes, planes=planes))

        return layers

def r2plus1d_resnet18_kinetics400(nclass=400, pretrained=False, pretrained_base=True,
                                  root='~/.mxnet/models', num_segments=1, num_crop=1,
                                  feat_ext=False, ctx=cpu(), **kwargs):
    r"""R2Plus1D with ResNet18 backbone trained on Kinetics400 dataset.

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

    model = R2Plus1D(nclass=nclass,
                     block=BasicBlock,
                     layers=[2, 2, 2, 2],
                     num_segments=num_segments,
                     num_crop=num_crop,
                     feat_ext=feat_ext,
                     ctx=ctx,
                     **kwargs)
    model.initialize(init.MSRAPrelu(), ctx=ctx)

    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('r2plus1d_resnet18_kinetics400',
                                             tag=pretrained, root=root), ctx=ctx)
        from ...data import Kinetics400Attr
        attrib = Kinetics400Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)

    return model

def r2plus1d_resnet34_kinetics400(nclass=400, pretrained=False, pretrained_base=True,
                                  root='~/.mxnet/models', num_segments=1, num_crop=1,
                                  feat_ext=False, ctx=cpu(), **kwargs):
    r"""R2Plus1D with ResNet34 backbone trained on Kinetics400 dataset.

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

    model = R2Plus1D(nclass=nclass,
                     block=BasicBlock,
                     layers=[3, 4, 6, 3],
                     num_segments=num_segments,
                     num_crop=num_crop,
                     feat_ext=feat_ext,
                     ctx=ctx,
                     **kwargs)
    model.initialize(init.MSRAPrelu(), ctx=ctx)

    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('r2plus1d_resnet34_kinetics400',
                                             tag=pretrained, root=root), ctx=ctx)
        from ...data import Kinetics400Attr
        attrib = Kinetics400Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)

    return model

def r2plus1d_resnet50_kinetics400(nclass=400, pretrained=False, pretrained_base=True,
                                  root='~/.mxnet/models', num_segments=1, num_crop=1,
                                  feat_ext=False, ctx=cpu(), **kwargs):
    r"""R2Plus1D with ResNet50 backbone trained on Kinetics400 dataset.

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

    model = R2Plus1D(nclass=nclass,
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
        model.load_parameters(get_model_file('r2plus1d_resnet50_kinetics400',
                                             tag=pretrained, root=root), ctx=ctx)
        from ...data import Kinetics400Attr
        attrib = Kinetics400Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)

    return model

def r2plus1d_resnet101_kinetics400(nclass=400, pretrained=False, pretrained_base=True,
                                   root='~/.mxnet/models', num_segments=1, num_crop=1,
                                   feat_ext=False, ctx=cpu(), **kwargs):
    r"""R2Plus1D with ResNet101 backbone trained on Kinetics400 dataset.

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

    model = R2Plus1D(nclass=nclass,
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
        model.load_parameters(get_model_file('r2plus1d_resnet101_kinetics400',
                                             tag=pretrained, root=root), ctx=ctx)
        from ...data import Kinetics400Attr
        attrib = Kinetics400Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)

    return model

def r2plus1d_resnet152_kinetics400(nclass=400, pretrained=False, pretrained_base=True,
                                   root='~/.mxnet/models', num_segments=1, num_crop=1,
                                   feat_ext=False, ctx=cpu(), **kwargs):
    r"""R2Plus1D with ResNet152 backbone trained on Kinetics400 dataset.

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

    model = R2Plus1D(nclass=nclass,
                     block=Bottleneck,
                     layers=[3, 8, 36, 3],
                     num_segments=num_segments,
                     num_crop=num_crop,
                     feat_ext=feat_ext,
                     ctx=ctx,
                     **kwargs)
    model.initialize(init.MSRAPrelu(), ctx=ctx)

    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('r2plus1d_resnet152_kinetics400',
                                             tag=pretrained, root=root), ctx=ctx)
        from ...data import Kinetics400Attr
        attrib = Kinetics400Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)

    return model
