import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon.nn import HybridBlock
# pylint: disable-all

__all__ = ['PAM_Module', 'CAM_Module']


class PAM_Module(HybridBlock):
    r""" Position attention module
    from the paper `"Dual Attention Network for Scene Segmentation"
    <https://arxiv.org/abs/1809.02983>`
    PAM_Module captures long-range spatial contextual information.
    """
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2D(in_channels=in_dim, channels=in_dim//8, kernel_size=(1, 1))
        self.key_conv = nn.Conv2D(in_channels=in_dim, channels=in_dim//8, kernel_size=(1, 1))
        self.value_conv = nn.Conv2D(in_channels=in_dim, channels=in_dim, kernel_size=(1, 1))
        self.gamma = self.params.get('gamma', shape=(1,), init=mx.init.Zero())

    def hybrid_forward(self, F, x, **kwargs):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        gamma = kwargs['gamma']
        proj_query = F.reshape(self.query_conv(x), (0, 0, -1))
        proj_key = F.reshape(self.key_conv(x), (0, 0, -1))
        energy = F.batch_dot(proj_query, proj_key, transpose_a=True)
        attention = F.softmax(energy)
        proj_value = F.reshape(self.value_conv(x), (0, 0, -1))
        out = F.batch_dot(proj_value, attention, transpose_b=True)
        out = F.reshape_like(out, x, lhs_begin=2, lhs_end=None, rhs_begin=2, rhs_end=None)

        out = F.broadcast_mul(gamma, out) + x

        return out


class CAM_Module(HybridBlock):
    r""" Channel attention module
    from the paper `"Dual Attention Network for Scene Segmentation"
    <https://arxiv.org/abs/1809.02983>`
    CAM_Module explicitly models interdependencies between channels.
    """
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = self.params.get('gamma', shape=(1,), init=mx.init.Zero())

    def hybrid_forward(self, F, x, **kwargs):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        gamma = kwargs['gamma']
        proj_query = F.reshape(x, (0, 0, -1))
        proj_key = F.reshape(x, (0, 0, -1))
        energy = F.batch_dot(proj_query, proj_key, transpose_b=True)
        energy_new = F.max(energy, -1, True).broadcast_like(energy) - energy
        attention = F.softmax(energy_new)
        proj_value = F.reshape(x, (0, 0, -1))

        out = F.batch_dot(attention, proj_value)
        out = F.reshape_like(out, x, lhs_begin=2, lhs_end=None, rhs_begin=2, rhs_end=None)

        out = F.broadcast_mul(gamma, out) + x
        return out