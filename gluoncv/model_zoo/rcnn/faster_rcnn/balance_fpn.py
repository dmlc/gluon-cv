from mxnet import gluon, nd
from mxnet.gluon import nn
import mxnet as mx
from mxnet.gluon.nn import BatchNorm
from ...action_recognition.non_local import NonLocal

def _conv2dsym(channel, kernel, padding, stride, norm_layer=BatchNorm, norm_kwargs=None):
    """A common conv-bn-leakyrelu cell"""
    cell = nn.HybridSequential(prefix='')
    cell.add(nn.Conv2D(channel, kernel_size=kernel,
                       strides=stride, padding=padding, use_bias=False))
    cell.add(norm_layer(epsilon=1e-5, momentum=0.9, **({} if norm_kwargs is None else norm_kwargs)))
    cell.add(nn.LeakyReLU(0.1))
    return cell

def _conv2d(channel, kernel, padding, stride):
    """A common conv-bn-leakyrelu cell"""
    cell = nn.HybridSequential(prefix='')
    cell.add(nn.Conv2D(channel, kernel_size=kernel,
                       strides=stride, padding=padding, use_bias=False))
    return cell


def _upsample(x, stride=2):

    return x.repeat(axis=-1, repeats=stride).repeat(axis=-2, repeats=stride)

class BFPNBlock(gluon.HybridBlock):
    '''
        dict(
            type='FPN',
            num_ins=4,
            out_channels=256,
            num_outs=5),
    '''
    def __init__(self,
                 out_channels,
                 num_ins,
                 num_outs):
        super(BFPNBlock, self).__init__()

        self.out_channels = out_channels  # 256,
        self.num_ins = num_ins  # 4, len(stages)
        self.num_outs = num_outs  # 5
        self.refine_level = 2
        self.refine_type = 'non_local'

        self.lateral_convs = nn.HybridSequential('L_')
        with self.lateral_convs.name_scope():
            for i in range(self.num_ins):  # 0, 4
                l_conv = _conv2d(out_channels, 1, 0, 1)
                self.lateral_convs.add(l_conv)

        self.fpn_convs = nn.HybridSequential('P_')
        with self.fpn_convs.name_scope():
            for i in range(self.num_ins):  # 0, 4
                fpn_conv = _conv2d(out_channels, 3, 1, 1)
                self.fpn_convs.add(fpn_conv)

        self.extra_convs = nn.HybridSequential('E_')
        with self.extra_convs.name_scope():
            if self.num_outs > self.num_ins:
                for i in range(self.num_outs - self.num_ins):
                    extra_conv = _conv2d(out_channels, 3, 1, 2)
                    self.extra_convs.add(extra_conv)

        self.refine = nn.HybridSequential('R_')
        with self.refine.name_scope():
            if self.refine_type == 'conv':
                self.refine.add(_conv2d(channel=out_channels, kernel=3, padding=1, stride=1))
            elif self.refine_type == 'non_local':
                no_local_block = NonLocal(in_channels=out_channels, dim=2, use_bn=False)
                self.refine.add(no_local_block)

    def hybrid_forward(self, F, inputs):
        assert len(inputs) == self.num_ins
        # step 1: gather multi-level features by resize and average

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)]  # 1*1

        # build top-down path
        for i in range(self.num_ins - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.contrib.BilinearResize2D(data=laterals[i],
                                                                           like=laterals[i - 1],
                                                                           mode='like')
        fpn_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(len(laterals))  # 3*3,1
            ]

        if self.num_outs > self.num_ins:
            for i in range(self.num_outs - self.num_ins):
                # change outs to laterals
                out = self.extra_convs[i](laterals[-1])
                fpn_outs.append(out)

        # gather all level features.
        feats = []
        for i in range(self.num_ins + 1):
            gathered = F.contrib.BilinearResize2D(data=fpn_outs[i],
                                                  like=fpn_outs[self.refine_level],
                                                  mode='like')
            feats.append(gathered)

        bsf = sum(feats) / len(feats)

        # step 2: refine gathered features
        if self.refine_type is not None:
            bsf = self.refine(bsf)

        # step 3: scatter refined features to multi-levels by a residual path
        outs = []
        for i in range(self.num_ins + 1):
            residual = F.contrib.BilinearResize2D(data=bsf,
                                                  like=fpn_outs[i],
                                                  mode='like')
            out = residual + fpn_outs[i]
            outs.append(out)

        return tuple(outs)


class BFPN_feature(gluon.HybridBlock):
    '''
    balance fpn
    '''
    def __init__(self, stages, out_channels=256, use_p6=True, **kwargs):
        super(BFPN_feature, self).__init__(**kwargs)
        self.stages = stages
        if use_p6:
            num_outs = len(self.stages)+1
        else:
            num_outs = len(self.stages)
        self.fpn_blocks = BFPNBlock(out_channels, len(self.stages), num_outs=num_outs)

    def hybrid_forward(self, x):
        routes = []
        for stage in self.stages:
            x = stage(x)
            routes.append(x)
        routes_bfp = self.fpn_blocks(routes)
        return routes_bfp
