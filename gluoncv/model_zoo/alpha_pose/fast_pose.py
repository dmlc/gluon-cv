"""Fast pose network for alpha pose"""
import mxnet as mx
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
from mxnet import initializer


class PixelShuffle(HybridBlock):
    """PixelShuffle layer for re-org channel to spatial dimention.

    Parameters
    ----------
    upscale_factor : int
        Upscaling factor for input->output spatially.

    """
    def __init__(self, upscale_factor):
        super(PixelShuffle, self).__init__()
        self.upscale_factor = upscale_factor

    def hybrid_forward(self, F, x):
        f1, f2 = self.upscale_factor, self.upscale_factor
        # (N, f1*f2*C, H, W)
        x = F.reshape(x, (0, -4, -1, f1 * f2, 0, 0))  # (N, C, f1*f2, H, W)
        x = F.reshape(x, (0, 0, -4, f1, f2, 0, 0))    # (N, C, f1, f2, H, W)
        x = F.transpose(x, (0, 1, 4, 2, 5, 3))        # (N, C, H, f1, W, f2)
        x = F.reshape(x, (0, 0, -3, -3))              # (N, C, H*f1, W*f2)
        return x


class DUC(HybridBlock):
    def __init__(self, planes, upscale_factor=2, norm_layer=nn.BatchNorm, **kwargs):
        super(DUC, self).__init__()
        with self.name_scope():
            self.conv = nn.Conv2D(planes, kernel_size=3, padding=1, use_bias=False)
            self.bn = norm_layer(**kwargs)
            self.relu = nn.Activation('relu')
            self.pixel_shuffle = PixelShuffle(upscale_factor)

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)

        return x


class FastPose(HybridBlock):
    def __init__(self, deconv_dim=256, **kwargs):
        super(FastPose, self).__init__(**kwargs)
