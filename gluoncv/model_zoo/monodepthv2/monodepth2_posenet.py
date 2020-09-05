"""Monodepth
Digging Into Self-Supervised Monocular Depth Estimation, ICCV 2019
https://arxiv.org/abs/1806.01260
"""
from mxnet.gluon import nn
from mxnet.context import cpu

from .resnet_encoder import ResnetEncoder
from .pose_decoder import PoseDecoder


class MonoDepth2PoseNet(nn.HybridBlock):
    r"""Monodepth2

    Parameters
    ----------
    backbone : string
        Pre-trained dilated backbone network type ('resnet18', 'resnet34', 'resnet50',
        'resnet101' or 'resnet152').
    pretrained_base : bool or str
        Refers to if the backbone is pretrained or not. If `True`,
        model weights of a model that was trained on ImageNet is loaded.
    num_input_images : int
        The number of input sequences. 1 for depth encoder, larger than 1 for pose encoder.
        (Default: 2)
    num_input_features : int
        The number of input feature maps from posenet encoder. (Default: 1)
    num_frames_to_predict_for: int
        The number of output pose between frames; If None, it equals num_input_features - 1.
        (Default: 2)
    stride: int
        The stride number for Conv in pose decoder. (Default: 1)

    Reference:

        Clement Godard, Oisin Mac Aodha, Michael Firman, Gabriel Brostow.
        "Digging Into Self-Supervised Monocular Depth Estimation." ICCV, 2019

    Examples
    --------
    >>> model = MonoDepth2PoseNet(backbone='resnet18', pretrained_base=True)
    >>> print(model)
    """
    # pylint: disable=unused-argument
    def __init__(self, backbone, pretrained_base, num_input_images=2, num_input_features=1,
                 num_frames_to_predict_for=2, stride=1, ctx=cpu(), **kwargs):
        super(MonoDepth2PoseNet, self).__init__()

        with self.name_scope():
            self.encoder = ResnetEncoder(backbone, pretrained_base,
                                         num_input_images=num_input_images, ctx=ctx)
            if not pretrained_base:
                self.encoder.initialize(ctx=ctx)
            self.decoder = PoseDecoder(self.encoder.num_ch_enc,
                                       num_input_features=num_input_features,
                                       num_frames_to_predict_for=num_frames_to_predict_for,
                                       stride=stride)
            self.decoder.initialize(ctx=ctx)

    def hybrid_forward(self, F, x):
        # pylint: disable=unused-argument
        features = [self.encoder(x)]
        axisangle, translation = self.decoder(features)

        return axisangle, translation

    def demo(self, x):
        return self.predict(x)

    def predict(self, x):
        features = [self.encoder.predict(x)]
        axisangle, translation = self.decoder.predict(features)

        return axisangle, translation


def get_monodepth2posenet(backbone='resnet18', pretrained_base=True, num_input_images=2,
                          num_input_features=1, num_frames_to_predict_for=2, stride=1,
                          root='~/.mxnet/models', ctx=cpu(0), pretrained=False,
                          pretrained_model='kitti_stereo_640x192', **kwargs):
    r"""Monodepth2

    Parameters
    ----------
    backbone : string
        Pre-trained dilated backbone network type ('resnet18', 'resnet34', 'resnet50',
        'resnet101' or 'resnet152').
    pretrained_base : bool or str
        Refers to if the backbone is pretrained or not. If `True`,
        model weights of a model that was trained on ImageNet is loaded.
    num_input_images : int
        The number of input sequences. 1 for depth encoder, larger than 1 for pose encoder.
        (Default: 2)
    num_input_features : int
        The number of input feature maps from posenet encoder. (Default: 1)
    num_frames_to_predict_for: int
        The number of output pose between frames; If None, it equals num_input_features - 1.
        (Default: 2)
    stride: int
        The stride number for Conv in pose decoder. (Default: 1)

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

    model = MonoDepth2PoseNet(
        backbone=backbone, pretrained_base=pretrained_base,
        num_input_images=num_input_images, num_input_features=num_input_features,
        num_frames_to_predict_for=num_frames_to_predict_for, stride=stride,
        ctx=ctx, **kwargs)

    if pretrained:
        from ...model_zoo.model_store import get_model_file
        model.load_parameters(
            get_model_file('monodepth2_%s_%s' % (backbone, pretrained_model),
                           tag=pretrained, root=root),
            ctx=ctx
        )
    return model


def get_monodepth2_resnet18_posenet_kitti_mono_640x192(**kwargs):
    r"""Monodepth2 PoseNet

    Parameters
    ----------
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet18').

    """
    return get_monodepth2posenet(backbone='resnet18',
                                 pretrained_model='posenet_kitti_mono_640x192', **kwargs)


def get_monodepth2_resnet18_posenet_kitti_mono_stereo_640x192(**kwargs):
    r"""Monodepth2 PoseNet

    Parameters
    ----------
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet18').

    """
    return get_monodepth2posenet(backbone='resnet18',
                                 pretrained_model='posenet_kitti_mono_stereo_640x192', **kwargs)
