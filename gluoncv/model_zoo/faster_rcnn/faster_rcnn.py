"""Faster RCNN Model"""
from mxnet import cpu
import mxnet.ndarray as F

from .rpn import RPN, RegionProposal
from .rcnn import RCNN_ResNet
# pylint: disable=arguments-differ, unused-variable, invalid-sequence-index

__all__ = ['FasterRCNN', 'get_faster_rccn', 'get_faster_rcnn_resnet50_coco',
           'get_faster_rcnn_resnet101_voc']

class FasterRCNN(RCNN_ResNet):
    """ faster RCNN """
    def __init__(self, classes, backbone, roi_mode='align', **kwargs):
        super(FasterRCNN, self).__init__(classes, backbone, **kwargs)
        self.classes = classes
        self.roi_mode = roi_mode
        self.roi_size = (14, 14)
        self.rpn = RPN(1024, num_anchors=15)
        self.region_proposal = RegionProposal()
        self.rpn.initialize()

    def forward(self, x):
        # TODO FIXME padding? for different image shape
        image_shape = x.shape
        # B, C, H, W
        base_feat = self.base_forward(x)
        # Region Proposal for ROIs
        rpn_cls, rpn_reg = self.rpn(base_feat)
        # B, N, 4
        rpn_scores, rois = self.region_proposal(rpn_cls, rpn_reg, base_feat.shape, image_shape)

        batches = rois.shape[0]
        num_rois = rois.shape[1]
        # ROI Pooling
        roi_batchid = F.concatenate([batchid * F.ones((num_rois, 1), rois.context)
                                     for batchid in range(batches)], axis=0)
        roi_batchid = roi_batchid.reshape(-1, 1)
        rois = rois.reshape(-1, 4)
        rpn_batchid = F.concatenate([roi_batchid, rois], axis=1)
        # B*N, C, 7, 7
        pooled_feat = self.roi_feature(base_feat, rpn_batchid)
        # RCNN predict
        rcnn_cls, rcnn_reg = self.top_forward(pooled_feat)
        rcnn_cls = F.softmax(rcnn_cls, axis=1)
        return rcnn_cls, rcnn_reg, rois, base_feat

    def roi_feature(self, base_feat, rois):
        if self.roi_mode == 'pool':
            return F.ROIPooling(base_feat, rois, self.roi_size, 1.0/self.stride)
        elif self.roi_mode == 'align':
            # ROI Align Layer with AVG Pooling
            return  F.contrib.ROIAlign(base_feat, rois, self.roi_size, 1.0/self.stride)
        else:
            raise NotImplementedError

    @staticmethod
    def rcnn_nms(rcnn_cls, bbox_pred, thresh=0.5, pre_nms_topN=-1, topK=100):
        """RCNN NMS"""
        nclass = rcnn_cls.shape[1]
        nboxs = bbox_pred.shape[0]
        ids = F.concatenate([classid * F.ones((nboxs, 1), rcnn_cls.context)
                             for classid in range(nclass)], axis=1)
        rcnn_cls = rcnn_cls.reshape(nboxs * nclass, 1)
        bbox_pred = bbox_pred.reshape(nboxs * nclass, 4)
        data = F.concat(ids.reshape(nboxs * nclass, 1), rcnn_cls, bbox_pred, dim=1)
        nms_pred = F.contrib.box_nms(data, thresh, pre_nms_topN, coord_start=2,
                                     score_index=1, id_index=0, force_suppress=False)
        # topK with effective rois
        effect = int(F.sum(nms_pred[:, 0] >= 0).asscalar())
        topK = topK if effect > topK else effect
        # N,1
        rcnn_ids = nms_pred[:topK, 0]
        rcnn_scores = nms_pred[:topK, 1]
        # N, 4
        rcnn_bbox = nms_pred[:topK, 2:]
        return rcnn_ids, rcnn_scores, rcnn_bbox


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
