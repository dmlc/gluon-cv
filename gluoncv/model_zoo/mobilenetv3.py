from __future__ import division

from mxnet.gluon.block import HybridBlock
from mxnet.gluon import nn
import numpy as np

def make_divisible(x, divisible_by=8):
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)

class ReLU6(HybridBlock):
    def __init__(self, **kwargs):
        super(ReLU6, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return F.clip(x, 0, 6)

class HardSigmoid(HybridBlock):
    def __init__(self, **kwargs):
        super(HardSigmoid, self).__init__(**kwargs)
        self.act = ReLU6()

    def hybrid_forward(self, F, x):
        return self.act(x + 3.) / 6.

class HardSwish(HybridBlock):
    def __init__(self, **kwargs):
        super(HardSwish, self).__init__(**kwargs)
        self.act = HardSigmoid()

    def hybrid_forward(self, F, x):
        return x * self.act(x)

class Activation(HybridBlock):
    def __init__(self, act_func, **kwargs):
        super(Activation, self).__init__(**kwargs)
        if act_func == "relu":
            self.act = nn.Activation('relu')
        elif act_func == "relu6":
            self.act = ReLU6()
        elif act_func == "hard_sigmoid":
            self.act = HardSigmoid()
        elif act_func == "swish":
            self.act = nn.Swish()
        elif act_func == "hard_swish":
            self.act = HardSwish()
        elif act_func == "leaky":
            self.act = nn.LeakyReLU(alpha=0.375)
        else:
            raise NotImplemented

    def hybrid_forward(self, F, x):
        return self.act(x)

class _SE(HybridBlock):
    def __init__(self, num_out, ratio=4, act_func=["relu", "hard_sigmoid"], use_bn=False, prefix='', **kwargs):
        super(_SE, self).__init__(**kwargs)
        self.use_bn = use_bn
        num_mid = make_divisible(num_out // ratio)
        self.pool = nn.GlobalAvgPool2D()
        self.conv1 = nn.Conv2D(channels=num_mid, kernel_size=1, use_bias=True, prefix=('%s_fc1_' % prefix))
        self.act1 = Activation(act_func[0])
        self.conv2 = nn.Conv2D(channels=num_out, kernel_size=1, use_bias=True, prefix=('%s_fc2_' % prefix))
        self.act2 = Activation(act_func[1])

    def hybrid_forward(self, F, x):
        out = self.pool(x)
        out = self.conv1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.act2(out)
        return F.broadcast_mul(x, out)

class _Unit(HybridBlock):
    def __init__(self, num_out, kernel_size=1, strides=1, pad=0, num_groups=1,
                       use_act=True, use_bn=True, act_type="relu", prefix='', **kwargs):
        super(_Unit, self).__init__(**kwargs)
        self.use_act = use_act
        self.conv = nn.Conv2D(channels=num_out, kernel_size=kernel_size, strides=strides, padding=pad, groups=num_groups, use_bias=False, prefix='%s-conv2d_'%prefix)
        self.bn = nn.BatchNorm(prefix='%s-batchnorm_'%prefix)
        if use_act == True:
            self.act = Activation(act_type)

    def hybrid_forward(self, F, x):
        out = self.conv(x)
        out = self.bn(out)
        if self.use_act:
            out = self.act(out)
        return out

class _ResUnit(HybridBlock):
    def __init__(self, num_in, num_mid, num_out, kernel_size, act_type="relu", use_se=False, strides=1, prefix='', **kwargs):
        super(_ResUnit, self).__init__(**kwargs)
        self.use_se = use_se
        self.first_conv = True if num_out != num_mid else False
        self.use_short_cut_conv = True
        if self.first_conv:
            self.expand = _Unit(num_mid, kernel_size=1, strides=1, pad=0, act_type=act_type, prefix='%s-exp'%prefix)
        self.conv1 = _Unit(num_mid, kernel_size=kernel_size, strides=strides,
                           pad=self._get_pad(kernel_size), act_type=act_type, num_groups=num_mid, prefix='%s-depthwise'%prefix)
        if use_se:
            self.se = _SE(num_mid, prefix='%s-se'%prefix) #, act_func=["hard_swish", "hard_sigmoid"])
        self.conv2 = _Unit(num_out, kernel_size=1, strides=1, pad=0, act_type=act_type, use_act=False, prefix='%s-linear'%prefix)
        if num_in != num_out or strides != 1:
            self.use_short_cut_conv = False

    def hybrid_forward(self, F, x):
        out = self.expand(x) if self.first_conv else x
        out = self.conv1(out)
        if self.use_se:
            out = self.se(out)
        out = self.conv2(out)
        if self.use_short_cut_conv:
            return x + out
        else:
            return out

    def _get_pad(self, kernel_size):
        if kernel_size == 1:
            return 0
        elif kernel_size == 3:
            return 1
        elif kernel_size == 5:
            return 2
        elif kernel_size == 7:
            return 3


class _MobileNetV3(HybridBlock):
    # pylint: disable=unused-variable
    def __init__(self, cfg, cls_ch_squeeze, cls_ch_expand, multiplier=1.,
                 classes=1000,
                 norm_kwargs=None, last_gamma=False,
                 final_drop=0., use_global_stats=False,
                 name_prefix='', **kwargs):
        super(_MobileNetV3, self).__init__(prefix=name_prefix)
        norm_kwargs = norm_kwargs if norm_kwargs is not None else {}
        if use_global_stats:
            norm_kwargs['use_global_stats'] = True
        # initialize residual networks
        k = multiplier
        self.last_gamma = last_gamma
        self.norm_kwargs = norm_kwargs
        self.inplanes = 16

        with self.name_scope():
            self.conv = nn.HybridSequential()
            self.conv.add(nn.Conv2D(channels=make_divisible(k*self.inplanes), kernel_size=3, padding=1, strides=2,
                                         use_bias=False, prefix='first-3x3-conv-conv2d_'))
            self.conv.add(nn.BatchNorm(prefix='first-3x3-conv-batchnorm_'))
            self.conv.add(HardSwish())
            i = 0
            for layer_cfg in cfg:
                layer = self._make_layer(kernel_size=layer_cfg[0],
                                         exp_ch=make_divisible(k * layer_cfg[1]),
                                         out_channel=make_divisible(k * layer_cfg[2]),
                                         use_se=layer_cfg[3],
                                         act_func=layer_cfg[4],
                                         stride=layer_cfg[5],
                                         prefix='seq-%d'%i,
                                        )
                self.conv.add(layer)
                i += 1
            self.cls = nn.HybridSequential()
            self.cls.add(nn.Conv2D(channels=make_divisible(k*cls_ch_squeeze), kernel_size=1, padding=0, strides=1,
                                   use_bias=False, prefix='last-1x1-conv1-conv2d_'))
            self.cls.add(nn.BatchNorm(prefix='last-1x1-conv1-batchnorm_',
                                         **({} if norm_kwargs is None else norm_kwargs)))
            self.cls.add(HardSwish())
            self.cls.add(nn.GlobalAvgPool2D())
            self.cls.add(nn.Conv2D(channels=cls_ch_expand, kernel_size=1, padding=0, strides=1,
                                   use_bias=False, prefix='last-1x1-conv2-conv2d_'))
            self.cls.add(HardSwish())

            if final_drop > 0:
                self.cls.add(nn.Dropout(final_drop))
            self.cls.add(nn.Conv2D(in_channels=cls_ch_expand, channels=classes,
                                       kernel_size=1, prefix='fc_'))

            self.flat = nn.Flatten()

    def _make_layer(self, kernel_size, exp_ch, out_channel, use_se, act_func, stride=1, prefix=''):

        mid_planes = exp_ch
        out_planes = out_channel
        layer = _ResUnit(self.inplanes, mid_planes, out_planes, kernel_size, act_func, strides=stride, use_se=use_se, prefix=prefix)
        self.inplanes = out_planes
        return layer

    def hybrid_forward(self, F, x):
        x = self.conv(x)
        x = self.cls(x)
        x = self.flat(x)
        return x

def get_model(model_name, multiplier=1., **kwargs):
    if model_name == "large":
        cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  False, 'relu', 1],
                [3, 64,  24,  False, 'relu', 2],
                [3, 72,  24,  False, 'relu', 1],
                [5, 72,  40,  True,  'relu', 2],
                [5, 120, 40,  True,  'relu', 1],
                [5, 120, 40,  True,  'relu', 1],
                [3, 240, 80,  False, 'hard_swish', 2],
                [3, 200, 80,  False, 'hard_swish', 1],
                [3, 184, 80,  False, 'hard_swish', 1],
                [3, 184, 80,  False, 'hard_swish', 1],
                [3, 480, 112, True,  'hard_swish', 1],
                [3, 672, 112, True,  'hard_swish', 1],
                [5, 672, 160, True,  'hard_swish', 2],
                [5, 960, 160, True,  'hard_swish', 1],
                [5, 960, 160, True,  'hard_swish', 1],
            ]
        cls_ch_squeeze = 960
        cls_ch_expand = 1280

    elif model_name == "deep":
        cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  0,  'relu', 1],
                [3, 64,  24,  0,  'relu', 2], #stage2
                [3, 72,  24,  0,  'relu', 1],
                [3, 72,  24,  0,  'relu', 1],
                [5, 72,  40,  1,  'relu', 2], #stage3
                [5, 120, 40,  1,  'relu', 1],
                [5, 120, 40,  1,  'relu', 1],
                [5, 120, 40,  1,  'relu', 1],
                [3, 200, 80,  0,  'hard_swish', 2], #stage4
                [3, 200, 80,  0,  'hard_swish', 1],
                [3, 200, 80,  0,  'hard_swish', 1],
                [3, 184, 80,  0,  'hard_swish', 1],
                [3, 184, 80,  0,  'hard_swish', 1],
                [3, 184, 80,  0,  'hard_swish', 1],
                [3, 480, 112, 1,  'hard_swish', 1],
                [3, 480, 112, 1,  'hard_swish', 1],
                [3, 480, 112, 1,  'hard_swish', 1],
                [3, 480, 112, 1,  'hard_swish', 1],
                [3, 480, 112, 1,  'hard_swish', 1],
                [3, 672, 112, 1,  'hard_swish', 1],
                [3, 672, 112, 1,  'hard_swish', 1],
                [3, 672, 112, 1,  'hard_swish', 1],
                [3, 672, 112, 1,  'hard_swish', 1],
                [5, 672, 160, 1,  'hard_swish', 2], #stage5
                [5, 960, 160, 1,  'hard_swish', 1],
                [5, 960, 160, 1,  'hard_swish', 1],
            ]
        cls_ch_squeeze = 960
        cls_ch_expand = 1280

    elif model_name == "small":
        cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16,  16,  True,  'relu', 2],
                [3, 72,  24,  False, 'relu', 2],
                [3, 88,  24,  False, 'relu', 1],
                [5, 96,  40,  True,  'hard_swish', 2],
                [5, 240, 40,  True,  'hard_swish', 1],
                [5, 240, 40,  True,  'hard_swish', 1],
                [5, 120, 48,  True,  'hard_swish', 1],
                [5, 144, 48,  True,  'hard_swish', 1],
                [5, 288, 96,  True,  'hard_swish', 2],
                [5, 576, 96,  True,  'hard_swish', 1],
                [5, 576, 96,  True,  'hard_swish', 1],
            ]
        cls_ch_squeeze = 576
        cls_ch_expand = 1280
    else:
        raise NotImplementedError
    return _MobileNetV3(cfg, cls_ch_squeeze, cls_ch_expand, multiplier=multiplier, final_drop=0.2, **kwargs)
