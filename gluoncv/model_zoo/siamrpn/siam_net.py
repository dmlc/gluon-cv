"""SiamRPN network"""
# pylint: disable=arguments-differ,unused-argument
from mxnet.gluon.block import HybridBlock
from gluoncv.model_zoo.siamrpn.siam_alexnet import alexnetlegacy
from gluoncv.model_zoo.siamrpn.siam_rpn import DepthwiseRPN

class SiamrpnNet(HybridBlock):
    """
        SiamrpnNet
    """
    def __init__(self):
        super(SiamrpnNet, self).__init__()
        self.backbone = alexnetlegacy()
        self.rpn_head = DepthwiseRPN()
        self.zbranch = None
        self.xbranch = None


    def template(self, zinput):
        """
        template z branch
        """
        zbranch = self.backbone(zinput)
        self.zbranch = zbranch

    def track(self, xinput):
        """
        track x branch
        """
        xbranch = self.backbone(xinput)
        cls, loc = self.rpn_head(self.zbranch, xbranch)
        return {
            'cls': cls,
            'loc': loc,
            }

    def mask_refine(self, pos):
        return self.refine_head(self.xf, self.mask_corr_feature, pos)

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def hybrid_forward(self, F, data):
        """ only used in training """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        label_loc_weight = data['label_loc_weight'].cuda()
        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)

        cls, loc = self.rpn_head(zf, xf)
        # get loss
        cls = self.log_softmax(cls)
        cls_loss = select_cross_entropy_loss(cls, label_cls)
        loc_loss = weight_l1_loss(loc, label_loc, label_loc_weight)

        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
