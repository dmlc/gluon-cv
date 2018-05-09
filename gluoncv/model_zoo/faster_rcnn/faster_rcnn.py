import mxnet as mx
from mxnet import gluon
import mxnet.ndarray as F
import mxnet.gluon.nn as nn
from mxnet.gluon import Block

from .rpn import RPN, RegionProposal
from .rcnn import RCNN_ResNet
from .bbox import bbox_inverse_transform, bbox_clip

class FasterRCNN(RCNN_ResNet):
    """ faster RCNN """
    def __init__(self, classes, backbone, roi_mode='align', **kwargs):
        super(FasterRCNN, self).__init__(classes, backbone, **kwargs)
        self.classes = classes
        self.roi_mode = roi_mode
        self.rpn = RPN(1024, num_anchors=9)
        self.region_proposal = RegionProposal(self.stride)
        self.rpn.initialize()

    def forward(self, x):
        # TODO FIXME padding? for different image shape
        image_shape = x.shape
        # B, C, H, W
        base_feat = self.base_forward(x)
        # Region Proposal for ROIs
        rpn_cls, rpn_reg = self.rpn(base_feat)
        # B, N, 4
        # B, sample_size, 4, TODO: Change to contrib.Proposal operator
        rois = self.region_proposal(rpn_cls, rpn_reg, base_feat.shape, image_shape)
        rois = rois.reshape(-1, 4)
        # ROI Pooling
        # B*N, C, 7, 7
        # FIXME batch ID
        rpn_batchid = F.concatenate([mx.nd.zeros((rois.shape[0], 1), rois.context), rois], axis=1)
        pooled_feat = self.roi_feature(base_feat, rpn_batchid)
        # RCNN predict
        # rcnn_cls \in R^(B*N, nclass, 7, 7) && rcnn_reg \in R^(B*N, 4*nclass, 7, 7)
        rcnn_cls, rcnn_reg = self.top_forward(pooled_feat)
        rcnn_cls = mx.nd.softmax(rcnn_cls, axis=1)
        # BBox
        rcnn_bbox_pred = mx.nd.zeros(rcnn_reg.shape)
        for i in range(self.classes):
            transform_bbox = bbox_inverse_transform(rois, rcnn_reg[:, i*4:(i+1)*4])
            rcnn_bbox_pred[:, i*4:(i+1)*4] = bbox_clip(
                transform_bbox, image_shape[2], image_shape[3])
        # rcnn_cls \in R^(B*N x nclass) && rcnn_reg \in R^(B*N x nclass*4)
        # reshape
        rcnn_cls = rcnn_cls.reshape(image_shape[0], -1, self.classes)
        rcnn_bbox_pred = rcnn_bbox_pred.reshape(image_shape[0], -1, self.classes, 4)
        return rcnn_cls, rcnn_bbox_pred

    def roi_feature(self, base_feat, rois):
        if self.roi_mode == 'pool':
            return F.ROIPooling(base_feat, rois, (7, 7), 1.0/self.stride)
        elif self.roi_mode == 'align':
            # ROI Align Layer with AVG Pooling
            x = F.contrib.ROIAlign(base_feat, rois, (8, 8), 1.0/self.stride)
            return F.Pooling(x, kernel= (2, 2), pool_type='avg')
        else:
            raise NotImplemented


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
    }
    num_class = {
        'pascal_voc': 21,
    }
    # infer number of classes
    model = FasterRCNN(num_class['pascal_voc'], backbone=backbone, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_params(get_model_file('faster_rccn_%s_%s'%(backbone, acronyms[dataset]),
                                         root=root), ctx=ctx)
    return model


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
