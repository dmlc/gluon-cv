"""Faster RCNN Model"""
from mxnet import cpu
import mxnet.ndarray as F

from .rpn import RPN, RegionProposal
from .rcnn import RCNN_ResNet
# pylint: disable=arguments-differ, unused-variable, dangerous-default-value

__all__ = ['FasterRCNN', 'get_faster_rccn', 'get_faster_rcnn_resnet50_coco',
           'get_faster_rcnn_resnet101_voc']

class FasterRCNN(RCNN_ResNet):
    r""" Faster RCNN Model
    Parameters
    ----------

    classes : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    train : bool
        Training model (default: False)
    dilated: bool
        Applying dilation to the backbone.
    roi_mode : string
        ROI pooling type (default: 'align'; 'align' or 'pool')
    roi_size : tuple of int
        ROI feature size (default: (14, 14))
    anchor_ratios : iterable of list
        Aspect ratios of anchors in the RPN. (default: [0.5, 1, 2])
    anchor_scales : iterable of list (default: [32, 64, 128, 256, 512])
        Scales of anchors in the RPN.
    nms_thresh : float, default is 0.7.
        Non-maximum suppression threshold. You can speficy < 0 or > 1 to disable NMS.
    nms_topK : int, default is 1000
        Apply NMS to top k detection results in NMS.
    pre_nms_topN : int, default is 6000
        Pick topN entries before NMS.
    """
    def __init__(self, classes, backbone, train=False, roi_mode='align', roi_size=(14, 14),
                 anchor_ratios=[0.5, 1, 2], anchor_scales=[32, 64, 128, 256, 512],
                 nms_thresh=0.7, nms_topK=1000, pre_nms_topN=6000, **kwargs):
        super(FasterRCNN, self).__init__(classes, backbone, **kwargs)
        self.classes = classes
        self.roi_mode = roi_mode
        self.roi_size = roi_size
        self.rpn = RPN(1024, num_anchors=15)
        self.region_proposal = RegionProposal(
            train=train, stride=self.stride, ratios=F.array(anchor_ratios),
            scales=F.array(anchor_scales), nms_thresh=nms_thresh, nms_topK=nms_topK,
            pre_nms_topN=pre_nms_topN)
        self.rpn.initialize()

    def forward(self, x, scaling_factor):
        image_shape = x.shape
        base_feat = self.base_forward(x)
        # Region Proposal for ROIs
        rpn_cls, rpn_reg = self.rpn(base_feat)
        # ROIs
        rpn_scores, rois = self.region_proposal(
            rpn_cls, rpn_reg, base_feat.shape, image_shape, scaling_factor)
        batches = rois.shape[0]
        num_rois = rois.shape[1]
        # Prepare Data for ROI Pooling, assuming batch=1 for now (FIXME)
        roi_batchid = F.concatenate([batchid * F.ones((num_rois, 1), rois.context)
                                     for batchid in range(batches)], axis=0)
        roi_batchid = roi_batchid.reshape(-1, 1)
        rois = rois.reshape(-1, 4)
        roi_with_batchid = F.concatenate([roi_batchid, rois], axis=1)
        # ROI Features (#ROIs, channel, h, w)
        pooled_feat = self.roi_feature(base_feat, roi_with_batchid)
        # RCNN Output
        rcnn_cls, rcnn_reg = self.top_forward(pooled_feat)
        rcnn_cls = F.softmax(rcnn_cls, axis=1)
        return rcnn_cls, rcnn_reg, rois, base_feat

    def roi_feature(self, base_feat, rois):
        if self.roi_mode == 'pool':
            return F.ROIPooling(base_feat, rois, self.roi_size, 1.0/self.stride)
        elif self.roi_mode == 'align':
            return  F.contrib.ROIAlign(base_feat, rois, self.roi_size, 1.0/self.stride)
        else:
            raise NotImplementedError


def get_faster_rccn(dataset='pascal_voc', backbone='resnet101', pretrained=False,
                    root='~/.mxnet/models', ctx=cpu(0), **kwargs):
    r"""Faster RCNN model from the paper
    "Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster r-cnn: Towards
    real-time object detection with region proposal networks"

    Parameters
    ----------
    dataset : str, default pascal_voc
        The dataset that model pretrained on. (pascal_voc, ade20k)
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_faster_rccn(dataset='pascal_voc', backbone='resnet50', pretrained=False)
    >>> print(model)
    """
    acronyms = {
        'pascal_voc': 'voc',
        'mscoco': 'coco',
    }
    num_class = {
        'pascal_voc': 21,
        'mscoco': 81,
    }
    # infer number of classes
    model = FasterRCNN(num_class[dataset], backbone=backbone, **kwargs)
    if pretrained:
        from ..model_store import get_model_file
        model.load_params(get_model_file('faster_rcnn_%s_%s'%(backbone, acronyms[dataset]),
                                         root=root), ctx=ctx)
    return model


def get_faster_rcnn_resnet50_coco(**kwargs):
    r"""Faster RCNN model from the paper
    "Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster r-cnn: Towards
    real-time object detection with region proposal networks"

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_faster_rcnn_resnet50_coco(pretrained=True)
    >>> print(model)
    """
    return get_faster_rccn('mscoco', 'resnet50', **kwargs)


def get_faster_rcnn_resnet101_voc(**kwargs):
    r"""Faster RCNN model from the paper
    "Ren, S., He, K., Girshick, R., & Sun, J. (2015). Faster r-cnn: Towards
    real-time object detection with region proposal networks"

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.

    Examples
    --------
    >>> model = get_faster_rcnn_resnet101_voc(pretrained=True)
    >>> print(model)
    """
    return get_faster_rccn('pascal_voc', 'resnet101', **kwargs)
