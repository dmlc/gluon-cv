"""Non-local block for video action recognition"""
# pylint: disable=line-too-long,too-many-lines,missing-docstring,arguments-differ,unused-argument
from mxnet.gluon.block import HybridBlock
from mxnet import init
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm

def build_nonlocal_block(cfg):
    """ Build nonlocal block from
    `"Non-local Neural Networks"
    <https://arxiv.org/abs/1711.07971>`_ paper.
    Code adapted from mmaction.
    """
    assert isinstance(cfg, dict)
    cfg_ = cfg.copy()
    return NonLocal(**cfg_)

class NonLocal(HybridBlock):
    r"""Non-local block

    Parameters
    ----------
    in_channels : int.
        Input channels of each block.
    nonlocal_type : str.
        Types of design for non-local block.
    dim : int, default 3.
        2D or 3D non-local block.
    embed_dim : int.
        Intermediate number of channels.
    sub_sample : bool.
        Whether to downsample the feature map to save computation.
    use_bn : bool.
        Whether to use batch normalization layer inside a non-local block.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    ctx : Context, default CPU.
        The context in which to load the pretrained weights.
    """
    def __init__(self, in_channels=1024, nonlocal_type="gaussian", dim=3,
                 embed=True, embed_dim=None, sub_sample=False, use_bn=True,
                 norm_layer=BatchNorm, norm_kwargs=None, ctx=None, **kwargs):
        super(NonLocal, self).__init__()

        assert nonlocal_type in ['gaussian', 'dot', 'concat']
        self.nonlocal_type = nonlocal_type
        self.embed = embed
        self.embed_dim = embed_dim if embed_dim is not None else in_channels // 2
        self.sub_sample = sub_sample
        self.use_bn = use_bn

        with self.name_scope():
            if self.embed:
                if dim == 2:
                    self.theta = nn.Conv2D(in_channels=in_channels, channels=self.embed_dim, kernel_size=(1, 1),
                                           strides=(1, 1), padding=(0, 0), weight_initializer=init.MSRAPrelu())
                    self.phi = nn.Conv2D(in_channels=in_channels, channels=self.embed_dim, kernel_size=(1, 1),
                                         strides=(1, 1), padding=(0, 0), weight_initializer=init.MSRAPrelu())
                    self.g = nn.Conv2D(in_channels=in_channels, channels=self.embed_dim, kernel_size=(1, 1),
                                       strides=(1, 1), padding=(0, 0), weight_initializer=init.MSRAPrelu())
                elif dim == 3:
                    self.theta = nn.Conv3D(in_channels=in_channels, channels=self.embed_dim, kernel_size=(1, 1, 1),
                                           strides=(1, 1, 1), padding=(0, 0, 0), weight_initializer=init.MSRAPrelu())
                    self.phi = nn.Conv3D(in_channels=in_channels, channels=self.embed_dim, kernel_size=(1, 1, 1),
                                         strides=(1, 1, 1), padding=(0, 0, 0), weight_initializer=init.MSRAPrelu())
                    self.g = nn.Conv3D(in_channels=in_channels, channels=self.embed_dim, kernel_size=(1, 1, 1),
                                       strides=(1, 1, 1), padding=(0, 0, 0), weight_initializer=init.MSRAPrelu())

            if self.nonlocal_type == 'concat':
                if dim == 2:
                    self.concat_proj = nn.HybridSequential()
                    self.concat_proj.add(nn.Conv2D(in_channels=self.embed_dim * 2, channels=1, kernel_size=(1, 1),
                                                   strides=(1, 1), padding=(0, 0), weight_initializer=init.MSRAPrelu()))
                    self.concat_proj.add(nn.Activation('relu'))
                elif dim == 3:
                    self.concat_proj = nn.HybridSequential()
                    self.concat_proj.add(nn.Conv3D(in_channels=self.embed_dim * 2, channels=1, kernel_size=(1, 1, 1),
                                                   strides=(1, 1, 1), padding=(0, 0, 0), weight_initializer=init.MSRAPrelu()))
                    self.concat_proj.add(nn.Activation('relu'))

            if sub_sample:
                if dim == 2:
                    self.max_pool = nn.MaxPool2D(pool_size=(2, 2))
                elif dim == 3:
                    self.max_pool = nn.MaxPool3D(pool_size=(1, 2, 2))
                self.sub_phi = nn.HybridSequential()
                self.sub_phi.add(self.phi)
                self.sub_phi.add(self.max_pool)
                self.sub_g = nn.HybridSequential()
                self.sub_g.add(self.g)
                self.sub_g.add(self.max_pool)

            if dim == 2:
                self.W = nn.Conv2D(in_channels=self.embed_dim, channels=in_channels, kernel_size=(1, 1),
                                   strides=(1, 1), padding=(0, 0), weight_initializer=init.MSRAPrelu())
            elif dim == 3:
                self.W = nn.Conv3D(in_channels=self.embed_dim, channels=in_channels, kernel_size=(1, 1, 1),
                                   strides=(1, 1, 1), padding=(0, 0, 0), weight_initializer=init.MSRAPrelu())

            if use_bn:
                self.bn = norm_layer(in_channels=in_channels, gamma_initializer='zeros', **({} if norm_kwargs is None else norm_kwargs))
                self.W_bn = nn.HybridSequential()
                self.W_bn.add(self.W)
                self.W_bn.add(self.bn)

    def hybrid_forward(self, F, x):
        if self.embed:
            theta = self.theta(x)
            if self.sub_sample:
                phi = self.sub_phi(x)
                g = self.sub_g(x)
            else:
                phi = self.phi(x)
                g = self.g(x)
        else:
            theta = x
            phi = x
            g = x

        if self.nonlocal_type == 'gaussian':
            # reshape [BxCxTxHxW] to [BxCxTHW]
            theta = F.reshape(theta, (0, 0, -1))
            phi = F.reshape(phi, (0, 0, -1))
            g = F.reshape(g, (0, 0, -1))
            # Direct transpose is slow, merge it into `batch_dot` operation.
            # theta_phi = nd.batch_dot(F.transpose(theta, axes=(0, 2, 1)), phi)
            theta_phi = F.batch_dot(theta, phi, transpose_a=True)
            # Normalizing the affinity tensor theta_phi before softmax.
            theta_phi = theta_phi * (self.embed_dim ** -0.5)
            attn = F.softmax(theta_phi, axis=2)
        elif self.non_local_type == 'concat':
            raise NotImplementedError
        elif self.non_local_type == 'dot':
            raise NotImplementedError
        else:
            raise NotImplementedError

        y = F.batch_dot(g, attn, transpose_b=True)
        y = F.reshape_like(y, x, lhs_begin=2, lhs_end=None, rhs_begin=2, rhs_end=None)

        if self.use_bn:
            z = self.W_bn(y) + x
        else:
            z = self.W(y) + x
        return z
