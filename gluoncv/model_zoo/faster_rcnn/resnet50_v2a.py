# pylint: disable=arguments-differ
"""Resnet50 v2a model which take original image with zero mean and uniform std."""
import mxnet as mx
from mxnet.gluon import nn, HybridBlock

__all__ = ['resnet50_v2a']

def _conv3x3(channels, stride, in_channels):
    """add conv 3x3 block."""
    return nn.Conv2D(channels, kernel_size=3, strides=stride, padding=1,
                     use_bias=False, in_channels=in_channels)


class BottleneckV2(HybridBlock):
    """Bottleneck V2 for internal use."""
    def __init__(self, channels, stride, downsample=False, in_channels=0, **kwargs):
        super(BottleneckV2, self).__init__(**kwargs)
        self.bn1 = nn.BatchNorm(epsilon=2e-5, use_global_stats=True)
        self.conv1 = nn.Conv2D(channels // 4, kernel_size=1, strides=1, use_bias=False)
        self.bn2 = nn.BatchNorm(epsilon=2e-5, use_global_stats=True)
        self.conv2 = _conv3x3(channels // 4, stride, channels // 4)
        self.bn3 = nn.BatchNorm(epsilon=2e-5, use_global_stats=True)
        self.conv3 = nn.Conv2D(channels, kernel_size=1, strides=1, use_bias=False)
        if downsample:
            self.downsample = nn.Conv2D(channels, 1, stride, use_bias=False,
                                        in_channels=in_channels)
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        """Custom forward."""
        residual = x
        x = self.bn1(x)
        x = F.Activation(x, act_type='relu')
        if self.downsample:
            residual = self.downsample(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = F.Activation(x, act_type='relu')
        x = self.conv2(x)

        x = self.bn3(x)
        x = F.Activation(x, act_type='relu')
        x = self.conv3(x)

        return x + residual


class Rescale(HybridBlock):
    """Rescale layer/block that restore the original by
    the default mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225].
    """
    def __init__(self, **kwargs):
        super(Rescale, self).__init__(**kwargs)
        with self.name_scope():
            init_scale = mx.nd.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1)) * 255
            self.init_scale = self.params.get_constant('init_scale', init_scale)
            init_mean = mx.nd.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1)) * 255
            self.init_mean = self.params.get_constant('init_mean', init_mean)

    def hybrid_forward(self, F, x, init_scale, init_mean):
        """Restore original image scale."""
        x = F.broadcast_mul(x, init_scale)  # restore std
        x = F.broadcast_add(x, init_mean)  # restore mean
        return x


class ResNet50V2(HybridBlock):
    """Resnet v2(a) for Faster-RCNN.

    Please ignore this if you are looking for model for other tasks.
    """
    def __init__(self, **kwargs):
        super(ResNet50V2, self).__init__(**kwargs)
        with self.name_scope():
            self.rescale = nn.HybridSequential(prefix='')
            self.rescale.add(Rescale(prefix=''))
            self.layer0 = nn.HybridSequential(prefix='')
            self.layer0.add(nn.BatchNorm(scale=False, epsilon=2e-5, use_global_stats=True))
            self.layer0.add(nn.Conv2D(64, 7, 2, 3, use_bias=False))
            self.layer0.add(nn.BatchNorm(epsilon=2e-5, use_global_stats=True))
            self.layer0.add(nn.Activation('relu'))
            self.layer0.add(nn.MaxPool2D(3, 2, 1))

            self.layer1 = self._make_layer(stage_index=1, layers=3, in_channels=64,
                                           channels=256, stride=1)
            self.layer2 = self._make_layer(stage_index=2, layers=4, in_channels=256,
                                           channels=512, stride=2)
            self.layer3 = self._make_layer(stage_index=3, layers=6, in_channels=512,
                                           channels=1024, stride=2)
            self.layer4 = self._make_layer(stage_index=4, layers=3, in_channels=1024,
                                           channels=2048, stride=2)

            self.layer4.add(nn.BatchNorm(epsilon=2e-5, use_global_stats=True))
            self.layer4.add(nn.Activation('relu'))
            # self.layer4.add(nn.GlobalAvgPool2D())
            # self.layer4.add(nn.Flatten())

    def _make_layer(self, stage_index, layers, channels, stride, in_channels=0):
        layer = nn.HybridSequential(prefix='stage%d_' % stage_index)
        with layer.name_scope():
            layer.add(BottleneckV2(channels, stride, channels != in_channels,
                                   in_channels=in_channels, prefix=''))
            for _ in range(layers - 1):
                layer.add(BottleneckV2(channels, 1, False, in_channels=channels, prefix=''))
        return layer

    # pylint: disable=unused-argument
    def hybrid_forward(self, F, x):
        """Custom forward."""
        x = self.rescale(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

def resnet50_v2a(pretrained=False, root='~/.mxnet/models', ctx=mx.cpu(0), **kwargs):
    """Constructs a ResNet50-v2a model.

    Please ignore this if you are looking for model for other tasks.

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    ctx : Context, default mx.cpu(0)
        The context in which to load the pretrained weights.
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
    """
    model = ResNet50V2(prefix='', **kwargs)
    if pretrained:
        from ..model_store import get_model_file
        model.load_params(get_model_file('resnet%d_v%da'%(50, 2),
                                         root=root), ctx=ctx, allow_missing=True)
        for v in model.collect_params(select='init_scale|init_mean').values():
            v.initialize(force_reinit=True, ctx=ctx)
    return model
