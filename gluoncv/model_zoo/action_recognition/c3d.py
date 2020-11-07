"""C3D, implemented in Gluon. https://arxiv.org/abs/1412.0767"""
# pylint: disable=arguments-differ,unused-argument

__all__ = ['C3D', 'c3d_kinetics400']

from mxnet import init
from mxnet.context import cpu
from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn

class C3D(HybridBlock):
    r"""
    The Convolutional 3D network (C3D).
    Learning Spatiotemporal Features with 3D Convolutional Networks.
    ICCV, 2015. https://arxiv.org/abs/1412.0767

    Parameters
    ----------
    nclass : int
        Number of classes in the training dataset.
    num_segments : int, default is 1.
        Number of segments used to evenly divide a video.
    num_crop : int, default is 1.
        Number of crops used during evaluation, choices are 1, 3 or 10.
    feat_ext : bool.
        Whether to extract features before dense classification layer or
        do a complete forward pass.
    dropout_ratio : float
        Dropout value used in the dropout layers after dense layers to avoid overfitting.
    init_std : float
        Default standard deviation value for initializing dense layers.
    ctx : str
        Context, default CPU. The context in which to load the pretrained weights.
    """

    def __init__(self, nclass, dropout_ratio=0.5,
                 num_segments=1, num_crop=1, feat_ext=False,
                 init_std=0.001, ctx=None, **kwargs):
        super(C3D, self).__init__()
        self.num_segments = num_segments
        self.num_crop = num_crop
        self.feat_ext = feat_ext
        self.feat_dim = 8192

        with self.name_scope():
            self.conv1 = nn.Conv3D(in_channels=3, channels=64,
                                   kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool1 = nn.MaxPool3D(pool_size=(1, 2, 2), strides=(1, 2, 2))

            self.conv2 = nn.Conv3D(in_channels=64, channels=128,
                                   kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool2 = nn.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2))

            self.conv3a = nn.Conv3D(in_channels=128, channels=256,
                                    kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.conv3b = nn.Conv3D(in_channels=256, channels=256,
                                    kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool3 = nn.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2))

            self.conv4a = nn.Conv3D(in_channels=256, channels=512,
                                    kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.conv4b = nn.Conv3D(in_channels=512, channels=512,
                                    kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool4 = nn.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2))

            self.conv5a = nn.Conv3D(in_channels=512, channels=512,
                                    kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.conv5b = nn.Conv3D(in_channels=512, channels=512,
                                    kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool5 = nn.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding=(0, 1, 1))

            self.fc6 = nn.Dense(in_units=8192, units=4096,
                                weight_initializer=init.Normal(sigma=init_std))
            self.fc7 = nn.Dense(in_units=4096, units=4096,
                                weight_initializer=init.Normal(sigma=init_std))
            self.fc8 = nn.Dense(in_units=4096, units=nclass,
                                weight_initializer=init.Normal(sigma=init_std))
            self.dropout = nn.Dropout(rate=dropout_ratio)
            self.relu = nn.Activation('relu')

    def hybrid_forward(self, F, x):
        """Hybrid forward of C3D net"""
        x = self.relu(self.conv1(x))
        x = self.pool1(x)

        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)

        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)

        # segmental consensus
        x = F.reshape(x, shape=(-1, self.num_segments * self.num_crop, self.feat_dim))
        x = F.mean(x, axis=1)

        x = self.relu(self.fc6(x))
        x = self.dropout(x)

        if self.feat_ext:
            return x

        x = self.relu(self.fc7(x))
        x = self.dropout(x)
        x = self.fc8(x)
        return x

def c3d_kinetics400(nclass=400, pretrained=False, ctx=cpu(),
                    root='~/.mxnet/models', num_segments=1, num_crop=1,
                    feat_ext=False, **kwargs):
    r"""The Convolutional 3D network (C3D) trained on Kinetics400 dataset.
    Learning Spatiotemporal Features with 3D Convolutional Networks.
    ICCV, 2015. https://arxiv.org/abs/1412.0767

    Parameters
    ----------
    nclass : int.
        Number of categories in the dataset.
    pretrained : bool or str.
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
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

    model = C3D(nclass=nclass, ctx=ctx, num_segments=num_segments,
                num_crop=num_crop, feat_ext=feat_ext, **kwargs)
    model.initialize(init.MSRAPrelu(), ctx=ctx)

    if pretrained:
        from ..model_store import get_model_file
        model.load_parameters(get_model_file('c3d_kinetics400',
                                             tag=pretrained, root=root), ctx=ctx)
        from ...data import Kinetics400Attr
        attrib = Kinetics400Attr()
        model.classes = attrib.classes
    model.collect_params().reset_ctx(ctx)

    return model
