import mxnet as mx
from mxnet.gluon import nn
from mxnet.context import cpu

from .resnet_encoder import ResnetEncoder
from .depth_decoder import DepthDecoder


class MonoDepth2(nn.HybridBlock):
    def __init__(self, backbone, pretrained_base, num_input_images=1,
                 scales=range(4), num_output_channels=1, use_skips=True, ctx=cpu(), **kwargs):
        super(MonoDepth2, self).__init__()

        self.encoder = ResnetEncoder(backbone, pretrained_base, num_input_images, ctx=ctx)
        self.decoder = DepthDecoder(self.encoder.num_ch_enc, scales, num_output_channels, use_skips)

        self.decoder.initialize(init=mx.init.MSRAPrelu(), ctx=ctx)

    def hybrid_forward(self, F, x):
        features = self.encoder(x)
        outputs = self.decoder(features)

        return outputs

    def demo(self, x):
        return self.predict(x)

    def predict(self, x):
        features = self.encoder(x)
        outputs = self.decoder(features)

        return outputs


def get_monodepth2(backbone='resnet18', pretrained_base=True, num_input_images=1,
                   scales=range(4), num_output_channels=1, use_skips=True,
                   root='~/.mxnet/models', ctx=cpu(0),
                   pretrained=False, pretrained_model='kitti_stereo_640x192', **kwargs):
    r"""MonoDepth2

    Parameters
    ----------
    backbone : string, default:'resnet18'
        Pre-trained dilated backbone network type
        ('resnet18', 'resnet34', 'resnet50', 'resnet101' or 'resnet152').
    pretrained_base : bool or str, default: True
        This will load pretrained backbone network, that was trained on ImageNet.
    num_input_images : int, default: 1
        The number of input images. 1 for depth encoder, larger than 1 for pose encoder.

    scales: list, default: range(4)
        The scales used in the loss.
    num_output_channels: int, default: 1
        The number of output channels.
    use_skips: bool, default: True
        This will use skip architecture in the network.

    ctx : Context, default: CPU
        The context in which to load the pretrained weights.
    root : str, default: '~/.mxnet/models'
        Location for keeping the model parameters.
    pretrained : bool or str, default: False
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_model : string, default: kitti_stereo_640x192
        The dataset that model pretrained on.

    """
    acronyms = {
        'kitti_stereo_640x192': 'kitti_stereo_640x192',
    }

    model = MonoDepth2(backbone=backbone, pretrained_base=pretrained_base,
                       num_input_images=num_input_images, scales=scales,
                       num_output_channels=num_output_channels, use_skips=use_skips,
                       ctx=ctx, **kwargs)

    if pretrained:
        from gluoncv.model_zoo.model_store import get_model_file
        model.load_parameters(
            get_model_file('monodepth2_%s_%s' % (backbone, acronyms[pretrained_model]),
                           tag=pretrained, root=root),
            ctx=ctx
        )
    return model


def get_monodepth2_resnet18_kitti_stereo_640x192(**kwargs):
    r"""Monodepth2

    Parameters
    ----------
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet18').

    """
    return get_monodepth2(backbone='resnet18', pretrained_model='kitti_stereo_640x192', **kwargs)
