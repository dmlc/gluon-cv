"""Pyramid Scene Parsing Network"""
from mxnet.gluon import nn
from mxnet.context import cpu
from mxnet.gluon.nn import HybridBlock
from mxnet import gluon
from .resnet import get_resnet
from .segbase import SegBaseModel
from .fcn import _FCNHead
from .attention import PAM_Module, CAM_Module
# pylint: disable-all

__all__ = ['DANet', 'get_danet', 'get_danet_coco']

class DANet(SegBaseModel):
    r"""Fully Convolutional Networks for Semantic Segmentation
    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
    Reference:
        Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks
        for semantic segmentation." *CVPR*, 2015
    """

    def __init__(self, nclass, backbone='resnet50', aux=False, ctx=cpu(), pretrained_base=True,
                height=None, width=None,base_size=576, crop_size=512, dilated=True, **kwargs):
        super(DANet, self).__init__(nclass, aux, backbone, ctx=ctx, base_size=base_size,
                crop_size=crop_size, pretrained_base=pretrained_base, **kwargs)
        self.aux=aux
        height = height if height is not None else crop_size
        width = width if width is not None else crop_size

        with self.name_scope():
            self.head = DANetHead(2048, nclass, **kwargs)
            self.head.initialize(ctx=ctx)

        if self.aux:
            self.auxlayer = _FCNHead(1024, nclass, **kwargs)
            self.auxlayer.initialize(ctx=ctx)
            self.auxlayer.collect_params().setattr('lr_mult', 10)

        self._up_kwargs = {'height': height, 'width': width}


    def hybrid_forward(self, F, x):
        # imsize = x.size()[2:]
        c3, c4 = self.base_forward(x)

        x = self.head(c4)
        x = list(x)
        x[0] = F.contrib.BilinearResize2D(x[0], **self._up_kwargs)
        x[1] = F.contrib.BilinearResize2D(x[1], **self._up_kwargs)
        x[2] = F.contrib.BilinearResize2D(x[2], **self._up_kwargs)


        outputs = [x[0]]
        outputs.append(x[1])
        outputs.append(x[2])

        if self.aux:

            auxout = self.auxlayer(c3)
            auxout = F.contrib.BilinearResize2D(auxout, **self._up_kwargs)
            outputs.append(auxout)

        return tuple(outputs)
        
class DANetHead(HybridBlock):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm, norm_kwargs=None,**kwargs):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.HybridSequential()
        self.conv5a.add(nn.Conv2D(in_channels=in_channels, channels=inter_channels, kernel_size=3, 
                                                            padding=1, use_bias=False))
        self.conv5a.add(norm_layer(in_channels=inter_channels, **({} if norm_kwargs is None else norm_kwargs)))
        self.conv5a.add(nn.Activation('relu'))
        # self.conv5a.initialize(ctx=ctx)

        self.conv5c = nn.HybridSequential()
        self.conv5c.add(nn.Conv2D(in_channels=in_channels, channels=inter_channels, kernel_size=3, 
                                                            padding=1, use_bias=False))
        self.conv5c.add(norm_layer(in_channels=inter_channels, **({} if norm_kwargs is None else norm_kwargs)))
        self.conv5c.add(nn.Activation('relu'))
        # self.conv5c.initialize(ctx=ctx)
        
        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.HybridSequential()
        self.conv51.add(nn.Conv2D(in_channels=inter_channels, channels=inter_channels, kernel_size=3, 
                                                            padding=1, use_bias=False))
        self.conv51.add(norm_layer(in_channels=inter_channels, **({} if norm_kwargs is None else norm_kwargs)))
        self.conv51.add(nn.Activation('relu'))
        # self.conv51.initialize(ctx=ctx)

        self.conv52 = nn.HybridSequential()
        self.conv52.add(nn.Conv2D(in_channels=inter_channels, channels=inter_channels, kernel_size=3, 
                                                            padding=1, use_bias=False))
        self.conv52.add(norm_layer(in_channels=inter_channels, **({} if norm_kwargs is None else norm_kwargs)))
        self.conv52.add(nn.Activation('relu'))

        self.conv6=nn.HybridSequential()
        self.conv6.add( nn.Conv2D(in_channels=512, channels=out_channels, kernel_size=1))
        self.conv6.add( nn.Dropout(0.1))

        self.conv7=nn.HybridSequential()
        self.conv7.add( nn.Conv2D(in_channels=512, channels=out_channels, kernel_size=1))
        self.conv7.add( nn.Dropout(0.1))

        self.conv8=nn.HybridSequential()
        self.conv8.add( nn.Conv2D(in_channels=512, channels=out_channels, kernel_size=1))
        self.conv8.add( nn.Dropout(0.1))


    def hybrid_forward(self, F, x):

        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv+sc_conv
        
        sasc_output = self.conv8(feat_sum)

        output = [sasc_output]
        output.append(sa_output)
        output.append(sc_output)
        return tuple(output)

    def predict(self, x):
        h, w = x.shape[2:]
        self._up_kwargs['height'] = h
        self._up_kwargs['width'] = w
        c3, c4 = self.base_forward(x)
        x = self.head.demo(c4)
        import mxnet.ndarray as F
        pred = F.contrib.BilinearResize2D(x, **self._up_kwargs)
        return pred


def get_danet(dataset='pascal_voc', backbone='resnet50', pretrained=False,
            root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    r"""DANet model from the paper `"Dual Attention Network for Scene Segmentation"
    <https://arxiv.org/abs/1809.02983.pdf>`
    """
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_aug': 'voc',
        'ade20k': 'ade',
        'coco': 'coco',
        'citys': 'citys',
    }
    
    from ..data import datasets
    # infer number of classes
    model = DANet(nclass=datasets[dataset].NUM_CLASS, backbone=backbone, ctx=ctx, **kwargs)
    model.classes = datasets[dataset].classes
    if pretrained:
        from .model_store import get_model_file
        model.load_parameters(get_model_file('danet_%s_%s'%(backbone, acronyms[dataset]),
                                             tag=pretrained, root=root), ctx=ctx)
    return model

