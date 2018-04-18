"""Base Model for Semantic Segmentation"""
from mxnet.gluon.nn import HybridBlock
from ..utils.metrics import voc_segmentation
from .dilated import dilatedresnetv0
# pylint: disable=abstract-method

class SegBaseModel(HybridBlock):
    r"""Base Model for Semantic Segmentation

    Parameters
    ----------
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : Block
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    """
    # pylint : disable=arguments-differ
    def __init__(self, nclass, aux, backbone='resnet50', **kwargs):
        super(SegBaseModel, self).__init__()
        self.aux = aux
        self.nclass = nclass
        with self.name_scope():
            if backbone == 'resnet50':
                pretrained = dilatedresnetv0.dilated_resnet50(pretrained=True, **kwargs)
            elif backbone == 'resnet101':
                pretrained = dilatedresnetv0.dilated_resnet101(pretrained=True, **kwargs)
            elif backbone == 'resnet152':
                pretrained = dilatedresnetv0.dilated_resnet152(pretrained=True, **kwargs)
            else:
                raise RuntimeError('unknown backbone: {}'.format(backbone))
            self.conv1 = pretrained.conv1
            self.bn1 = pretrained.bn1
            self.relu = pretrained.relu
            self.maxpool = pretrained.maxpool
            self.layer1 = pretrained.layer1
            self.layer2 = pretrained.layer2
            self.layer3 = pretrained.layer3
            self.layer4 = pretrained.layer4

    def base_forward(self, x):
        """forwarding pre-trained network"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        c3 = self.layer3(x)
        c4 = self.layer4(c3)
        return c3, c4

    def evaluate(self, x, target=None, bg=False):
        """evaluating network with inputs and targets"""
        pred = self.forward(x)
        if self.aux:
            pred = pred[0]
        if target is None:
            return pred
        correct, labeled = voc_segmentation.batch_pix_accuracy( \
            pred, target, bg)
        inter, union = voc_segmentation.batch_intersection_union(
            pred, target, self.nclass, bg)

        return correct, labeled, inter, union


class SegEvalModel(object):
    """Segmentation Eval Module"""
    def __init__(self, module, bg=False):
        self.module = module
        self.bg = bg

    def __call__(self, *inputs, **kwargs):
        return self.module.evaluate(*inputs, bg=self.bg, **kwargs)

    def collect_params(self):
        return self.module.collect_params()
