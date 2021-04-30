# pylint: disable=missing-function-docstring, unused-argument
"""
C2D video action recognition models based on ResNet
"""
import torch
import torch.nn as nn
import torchvision


__all__ = ['ActionRecResNetV1b', 'resnet18_v1b_kinetics400', 'resnet34_v1b_kinetics400',
           'resnet50_v1b_kinetics400', 'resnet101_v1b_kinetics400', 'resnet152_v1b_kinetics400',
           'resnet50_v1b_sthsthv2', 'resnet50_v1b_custom']


class ActionRecResNetV1b(nn.Module):
    r"""ResNet models for video action recognition
    Deep Residual Learning for Image Recognition, CVPR 2016
    https://arxiv.org/abs/1512.03385

    Parameters
    ----------
    depth : int, default is 50.
        Depth of ResNet, from {18, 34, 50, 101, 152}.
    num_classes : int
        Number of classes in the training dataset.
    pretrained_base : bool, default is True.
        Load pretrained base network (backbone), the extra layers are randomized.
    feat_ext : bool, default is False.
        Whether to extract features from backbone network or perform a standard network forward.
    partial_bn : bool, default is False.
        Freeze all batch normalization layers during training except the first one.
    dropout_ratio : float, default is 0.5.
        The dropout rate of a dropout layer.
        The larger the value, the more strength to prevent overfitting.
    init_std : float, default is 0.01.
        Standard deviation value when initialize a fully connected layer.
    num_segment : int, default is 1.
        Number of segments used to evenly divide a video.
    num_crop : int, default is 1.
        Number of crops used during training.

    Input : a single video frame or N images from N segments when num_segment > 1
    Output : a single predicted action label
    """
    def __init__(self, depth, num_classes, pretrained_base=True,
                 dropout_ratio=0.5, init_std=0.01,
                 num_segment=1, num_crop=1, feat_ext=False,
                 partial_bn=False, **kwargs):
        super(ActionRecResNetV1b, self).__init__()

        self.depth = depth
        self.num_classes = num_classes
        self.pretrained_base = pretrained_base
        self.feat_ext = feat_ext

        if self.depth == 18:
            C2D = torchvision.models.resnet18(pretrained=self.pretrained_base, progress=True)
            self.expansion = 1
        elif self.depth == 34:
            C2D = torchvision.models.resnet34(pretrained=self.pretrained_base, progress=True)
            self.expansion = 1
        elif self.depth == 50:
            C2D = torchvision.models.resnet50(pretrained=self.pretrained_base, progress=True)
            self.expansion = 4
        elif self.depth == 101:
            C2D = torchvision.models.resnet101(pretrained=self.pretrained_base, progress=True)
            self.expansion = 4
        elif self.depth == 152:
            C2D = torchvision.models.resnet152(pretrained=self.pretrained_base, progress=True)
            self.expansion = 4
        else:
            raise RuntimeError("We do not support ResNet with depth %d." % (self.depth))

        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        self.num_segment = num_segment
        self.num_crop = num_crop
        self.feat_dim = 512 * self.expansion

        self.conv1 = C2D.conv1
        self.bn1 = C2D.bn1
        self.relu = C2D.relu
        self.maxpool = C2D.maxpool
        self.layer1 = C2D.layer1
        self.layer2 = C2D.layer2
        self.layer3 = C2D.layer3
        self.layer4 = C2D.layer4
        self.avgpool = C2D.avgpool

        self.drop = nn.Dropout(self.dropout_ratio)
        self.fc = nn.Linear(in_features=self.feat_dim, out_features=num_classes)
        nn.init.normal_(self.fc.weight, 0, self.init_std)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        bs, ch, tm, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4)
        x = x.contiguous().view(bs * tm, ch, h, w)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.drop(x)
        x = torch.flatten(x, 1)

        # segmental consensus
        x = x.view(bs, tm, self.feat_dim)
        x = torch.mean(x, dim=1)

        if self.feat_ext:
            return x

        x = self.fc(x)
        return x


def resnet18_v1b_kinetics400(cfg):
    model = ActionRecResNetV1b(depth=18,
                               num_classes=cfg.CONFIG.DATA.NUM_CLASSES,
                               pretrained_base=cfg.CONFIG.MODEL.PRETRAINED_BASE,
                               feat_ext=cfg.CONFIG.INFERENCE.FEAT,
                               partial_bn=cfg.CONFIG.MODEL.PARTIAL_BN,
                               num_segment=cfg.CONFIG.DATA.NUM_SEGMENT,
                               num_crop=cfg.CONFIG.DATA.NUM_CROP,
                               dropout_ratio=0.5,
                               init_std=0.01)

    if cfg.CONFIG.MODEL.PRETRAINED:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('resnet18_v1b_kinetics400',
                                                        tag=cfg.CONFIG.MODEL.PRETRAINED)))
    return model


def resnet34_v1b_kinetics400(cfg):
    model = ActionRecResNetV1b(depth=34,
                               num_classes=cfg.CONFIG.DATA.NUM_CLASSES,
                               pretrained_base=cfg.CONFIG.MODEL.PRETRAINED_BASE,
                               feat_ext=cfg.CONFIG.INFERENCE.FEAT,
                               partial_bn=cfg.CONFIG.MODEL.PARTIAL_BN,
                               num_segment=cfg.CONFIG.DATA.NUM_SEGMENT,
                               num_crop=cfg.CONFIG.DATA.NUM_CROP,
                               dropout_ratio=0.5,
                               init_std=0.01)

    if cfg.CONFIG.MODEL.PRETRAINED:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('resnet34_v1b_kinetics400',
                                                        tag=cfg.CONFIG.MODEL.PRETRAINED)))
    return model


def resnet50_v1b_kinetics400(cfg):
    model = ActionRecResNetV1b(depth=50,
                               num_classes=cfg.CONFIG.DATA.NUM_CLASSES,
                               pretrained_base=cfg.CONFIG.MODEL.PRETRAINED_BASE,
                               feat_ext=cfg.CONFIG.INFERENCE.FEAT,
                               partial_bn=cfg.CONFIG.MODEL.PARTIAL_BN,
                               num_segment=cfg.CONFIG.DATA.NUM_SEGMENT,
                               num_crop=cfg.CONFIG.DATA.NUM_CROP,
                               dropout_ratio=0.5,
                               init_std=0.01)

    if cfg.CONFIG.MODEL.PRETRAINED:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('resnet50_v1b_kinetics400',
                                                        tag=cfg.CONFIG.MODEL.PRETRAINED)))
    return model


def resnet101_v1b_kinetics400(cfg):
    model = ActionRecResNetV1b(depth=101,
                               num_classes=cfg.CONFIG.DATA.NUM_CLASSES,
                               pretrained_base=cfg.CONFIG.MODEL.PRETRAINED_BASE,
                               feat_ext=cfg.CONFIG.INFERENCE.FEAT,
                               partial_bn=cfg.CONFIG.MODEL.PARTIAL_BN,
                               num_segment=cfg.CONFIG.DATA.NUM_SEGMENT,
                               num_crop=cfg.CONFIG.DATA.NUM_CROP,
                               dropout_ratio=0.5,
                               init_std=0.01)

    if cfg.CONFIG.MODEL.PRETRAINED:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('resnet101_v1b_kinetics400',
                                                        tag=cfg.CONFIG.MODEL.PRETRAINED)))
    return model


def resnet152_v1b_kinetics400(cfg):
    model = ActionRecResNetV1b(depth=152,
                               num_classes=cfg.CONFIG.DATA.NUM_CLASSES,
                               pretrained_base=cfg.CONFIG.MODEL.PRETRAINED_BASE,
                               feat_ext=cfg.CONFIG.INFERENCE.FEAT,
                               partial_bn=cfg.CONFIG.MODEL.PARTIAL_BN,
                               num_segment=cfg.CONFIG.DATA.NUM_SEGMENT,
                               num_crop=cfg.CONFIG.DATA.NUM_CROP,
                               dropout_ratio=0.5,
                               init_std=0.01)

    if cfg.CONFIG.MODEL.PRETRAINED:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('resnet152_v1b_kinetics400',
                                                        tag=cfg.CONFIG.MODEL.PRETRAINED)))
    return model


def resnet50_v1b_sthsthv2(cfg):
    model = ActionRecResNetV1b(depth=50,
                               num_classes=cfg.CONFIG.DATA.NUM_CLASSES,
                               pretrained_base=cfg.CONFIG.MODEL.PRETRAINED_BASE,
                               feat_ext=cfg.CONFIG.INFERENCE.FEAT,
                               partial_bn=cfg.CONFIG.MODEL.PARTIAL_BN,
                               num_segment=cfg.CONFIG.DATA.NUM_SEGMENT,
                               num_crop=cfg.CONFIG.DATA.NUM_CROP,
                               dropout_ratio=0.5,
                               init_std=0.01)

    if cfg.CONFIG.MODEL.PRETRAINED:
        from ..model_store import get_model_file
        model.load_state_dict(torch.load(get_model_file('resnet50_v1b_sthsthv2',
                                                        tag=cfg.CONFIG.MODEL.PRETRAINED)))
    return model


def resnet50_v1b_custom(cfg):
    model = ActionRecResNetV1b(depth=50,
                               num_classes=cfg.CONFIG.DATA.NUM_CLASSES,
                               pretrained_base=cfg.CONFIG.MODEL.PRETRAINED_BASE,
                               feat_ext=cfg.CONFIG.INFERENCE.FEAT,
                               partial_bn=cfg.CONFIG.MODEL.PARTIAL_BN,
                               num_segment=cfg.CONFIG.DATA.NUM_SEGMENT,
                               num_crop=cfg.CONFIG.DATA.NUM_CROP,
                               dropout_ratio=0.5,
                               init_std=0.01)

    if cfg.CONFIG.MODEL.PRETRAINED:
        from ..model_store import get_model_file
        state_dict = torch.load(get_model_file('resnet50_v1b_kinetics400', tag=cfg.CONFIG.MODEL.PRETRAINED))
        for k in list(state_dict.keys()):
            # retain only backbone up to before the classification layer
            if k.startswith('fc'):
                del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {'fc.weight', 'fc.bias'}
        print("=> Initialized from a ResNet50 model pretrained on Kinetcis400 dataset")
    return model
