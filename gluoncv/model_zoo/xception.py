# code adapted from https://github.com/jfzhang95/pytorch-deeplab-xception/
"""Xception, implemented in Gluon."""
__all__ = ['Xception', 'get_xcetption']
from mxnet.context import cpu
import mxnet.gluon.nn as nn

def fixed_padding(inputs, F, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (0, 0, 0, 0, pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs

class SeparableConv2d(nn.HybridBlock):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1,
                 dilation=1, bias=False, norm_layer=None, norm_kwargs=None):
        super(SeparableConv2d, self).__init__()
        norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, 0, dilation,
                               groups=inplanes, bias=bias)
        self.bn = norm_layer(inplanes, **norm_kwargs)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def hybrid_forward(self, F, x):
        x = fixed_padding(x, F, self.conv1.kernel_size[0], dilation=self.conv1.dilation[0])
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x

class Block(nn.HybridBlock):
    def __init__(self, inplanes, planes, reps, stride=1, dilation=1, norm_layer=None,
                 start_with_relu=True, grow_first=True, is_last=False):
        super(Block, self).__init__()
        norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
        if planes != inplanes or stride != 1:
            self.skip = nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False)
            self.skipbn = norm_layer(planes, **norm_kwargs)
        else:
            self.skip = None
        self.relu = nn.ReLU(inplace=True)
        self.rep = nn.Sequential()
        filters = inplanes
        if grow_first:
            if start_with_relu:
                self.rep.add(self.relu)
            self.rep.add(SeparableConv2d(inplanes, planes, 3, 1, dilation, norm_layer=norm_layer,
                         norm_kwargs=norm_kwargs))
            self.rep.add(norm_layer(planes, **norm_kwargs))
            filters = planes
        for i in range(reps - 1):
            if grow_first or start_with_relu:
                self.rep.add(self.relu)
            self.rep.add(SeparableConv2d(filters, filters, 3, 1, dilation, norm_layer=norm_layer,
                         norm_kwargs=norm_kwargs))
            self.rep.add(norm_layer(filters, **norm_kwargs))
        if not grow_first:
            self.rep.add(self.relu)
            self.rep.add(SeparableConv2d(inplanes, planes, 3, 1, dilation, norm_layer=norm_layer,
                         norm_kwargs=norm_kwargs))
            self.rep.add(norm_layer(planes, **norm_kwargs))
        if stride != 1:
            self.rep.add(self.relu)
            self.rep.add(SeparableConv2d(planes, planes, 3, 2, norm_layer=norm_layer,
                                         norm_kwargs=norm_kwargs))
            self.rep.add(norm_layer(planes, **norm_kwargs))
        elif is_last:
            self.rep.add(self.relu)
            self.rep.add(SeparableConv2d(planes, planes, 3, 1, norm_layer=norm_layer,
                                         norm_kwargs=norm_kwargs))
            self.rep.add(norm_layer(planes, **norm_kwargs))

    def hybrid_forward(self, F, inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x = x + skip
        return x

class Xception(nn.HybridBlock):
    """Modified Aligned Xception
    """
    def __init__(self, output_stride=, norm_layer=nn.BatchNorm,
                 norm_kwargs=None):
        super(Xception, self).__init__()
        norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
        if output_stride == 8:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = (1, 1)
        if output_stride == 16:
            entry_block3_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = (1, 2)
        elif output_stride == 8:
            entry_block3_stride = 1
            middle_block_dilation = 2
            exit_block_dilations = (2, 4)
        else:
            raise NotImplementedError
        # Entry flow
        with self.name_scope():
            self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False)
            self.bn1 = norm_layer(32, **norm_kwargs)
            self.relu = nn.ReLU(inplace=True)

            self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False)
            self.bn2 = norm_layer(64)

            self.block1 = Block(64, 128, reps=2, stride=2, norm_layer=norm_layer,
                                norm_kwargs=norm_kwargs,
                                start_with_relu=False)
            self.block2 = Block(128, 256, reps=2, stride=2, norm_layer=norm_layer,
                                norm_kwargs=norm_kwargs,
                                start_with_relu=False,
                                grow_first=True)
            self.block3 = Block(256, 728, reps=2, stride=entry_block3_stride,
                                norm_layer=norm_layer,
                                norm_kwargs=norm_kwargs,
                                start_with_relu=True, grow_first=True, is_last=True)
            # Middle flow
            self.midflow = nn.Sequential()
            for i in range(4, 20):
                self.midflow.add(('block%d'%i, Block(728, 728, reps=3, stride=1,
                                                     dilation=middle_block_dilation,
                                                     norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                                                     start_with_relu=True, grow_first=True)))

            # Exit flow
            self.block20 = Block(728, 1024, reps=2, stride=1, dilation=exit_block_dilations[0],
                                 norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                                 start_with_relu=True, grow_first=False,
                                 is_last=True)

            self.conv3 = SeparableConv2d(1024, 1536, 3, stride=1, dilation=exit_block_dilations[1],
                                         norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.bn3 = norm_layer(1536, **norm_kwargs)

            self.conv4 = SeparableConv2d(1536, 1536, 3, stride=1, dilation=exit_block_dilations[1],
                                         norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.bn4 = norm_layer(1536, **norm_kwargs)

            self.conv5 = SeparableConv2d(1536, 2048, 3, stride=1, dilation=exit_block_dilations[1],
                                         norm_layer=norm_layer, norm_kwargs=norm_kwargs)
            self.bn5 = norm_layer(2048, **norm_kwargs)

        def hybrid_forward(self, F, x):
            # Entry flow
            x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.block1(x)
        # add relu here
        x = self.relu(x)
        low_level_feat = x
        x = self.block2(x)
        x = self.block3(x)

        # Middle flow
        x = self.midflow(x)

        # Exit flow
        x = self.block20(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        return x, low_level_feat


# Constructor
def get_xcetption(pretrained=False, ctx=cpu(),
                  root='~/.mxnet/models', **kwargs):
    r"""Xception model from

    Parameters
    ----------
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default $MXNET_HOME/models
        Location for keeping the model parameters.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    net = Xception(**kwargs)
    if pretrained:
        from .model_store import get_model_file
        net.load_parameters(get_model_file('exception',
                                           tag=pretrained, root=root), ctx=ctx)
        from ..data import ImageNet1kAttr
        attrib = ImageNet1kAttr()
        net.synset = attrib.synset
        net.classes = attrib.classes
        net.classes_long = attrib.classes_long
    return net
