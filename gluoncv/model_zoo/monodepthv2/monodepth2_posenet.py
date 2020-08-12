"""Monodepth
Digging Into Self-Supervised Monocular Depth Estimation, ICCV 2019
https://arxiv.org/abs/1806.01260
"""
import mxnet as mx
from mxnet.gluon import nn
from mxnet.context import cpu

from .resnet_encoder import ResnetEncoder
from .pose_decoder import PoseDecoder


class MonoDepth2PoseNet(nn.HybridBlock):
    # pylint: disable=unused-argument
    def __init__(self, backbone, pretrained_base, num_input_images=1, num_input_features=1,
                 num_frames_to_predict_for=None, stride=1, ctx=cpu(), **kwargs):
        super(MonoDepth2PoseNet, self).__init__()

        with self.name_scope():
            self.encoder = ResnetEncoder(backbone, pretrained_base,
                                         num_input_images=num_input_images, ctx=ctx)
            self.decoder = PoseDecoder(self.encoder.num_ch_enc,
                                       num_input_features=num_input_features,
                                       num_frames_to_predict_for=num_frames_to_predict_for,
                                       stride=stride)
            self.decoder.initialize(ctx=ctx)

    def hybrid_forward(self, F, x):
        # pylint: disable=unused-argument
        features = [self.encoder(x)]
        outputs = self.decoder(features)

        return outputs

    def demo(self, x):
        return self.predict(x)

    def predict(self, x):
        features = [self.encoder.predict(x)]
        outputs = self.decoder.predict(features)

        return outputs
